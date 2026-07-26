#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t stat = (call);                                            \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                     \
            std::cerr << "cuBLAS error: " << stat                                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// input:  [B, S, E], where E = H * D
// output: [B, H, S, D]
__global__ void split_heads_kernel(const float* input, float* output,
                                   int B, int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (S * H * D);

    int out_idx = ((b * H + h) * S + s) * D + d;
    output[out_idx] = input[idx];
}

// input:  [B, H, S, D]
// output: [B, S, E], where E = H * D
__global__ void merge_heads_kernel(const float* input, float* output,
                                   int B, int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int s = (idx / D) % S;
    int h = (idx / (D * S)) % H;
    int b = idx / (D * S * H);

    int out_idx = ((b * S + s) * H + h) * D + d;
    output[out_idx] = input[idx];
}

// One CUDA block handles one row of the [B, H, S, S] attention matrix.
template <int BLOCK_SIZE>
__global__ void softmax_rows_kernel(float* scores, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float* row_ptr = scores + row * cols;

    __shared__ float shared[BLOCK_SIZE];

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_ptr[col]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float v = expf(row_ptr[col] - max_val);
        row_ptr[col] = v;
        local_sum += v;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / shared[0];

    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        row_ptr[col] *= inv_sum;
    }
}

static void row_major_gemm(cublasHandle_t handle,
                           cublasOperation_t transA,
                           cublasOperation_t transB,
                           int M, int N, int K,
                           const float* A,
                           const float* B,
                           float* C,
                           float alpha = 1.0f,
                           float beta = 0.0f) {
    int lda = (transA == CUBLAS_OP_N) ? K : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K;
    int ldc = N;
    cublasOperation_t cu_transA = (transB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cu_transB = (transA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;

    // cuBLAS is column-major. Reversing A/B computes C^T in the same memory
    // buffer that row-major C uses.
    CHECK_CUBLAS(cublasSgemm(handle,
                             cu_transA, cu_transB,
                             N, M, K,
                             &alpha,
                             B, ldb,
                             A, lda,
                             &beta,
                             C, ldc));
}

static void row_major_gemm_strided_batched(cublasHandle_t handle,
                                           cublasOperation_t transA,
                                           cublasOperation_t transB,
                                           int M, int N, int K,
                                           const float* A,
                                           long long strideA,
                                           const float* B,
                                           long long strideB,
                                           float* C,
                                           long long strideC,
                                           int batch_count,
                                           float alpha = 1.0f,
                                           float beta = 0.0f) {
    int lda = (transA == CUBLAS_OP_N) ? K : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K;
    int ldc = N;
    cublasOperation_t cu_transA = (transB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cu_transB = (transA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;

    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                           cu_transA, cu_transB,
                                           N, M, K,
                                           &alpha,
                                           B, ldb, strideB,
                                           A, lda, strideA,
                                           &beta,
                                           C, ldc, strideC,
                                           batch_count));
}

static void fill_random(std::vector<float>& x, float scale = 1.0f) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float& v : x) {
        v = dist(gen);
    }
}

int main() {
    const int B = 2;
    const int S = 4;
    const int E = 8;
    const int H = 2;
    const int D = E / H;
    const int tokens = B * S;
    const int head_batches = B * H;

    std::vector<float> h_x(tokens * E);
    std::vector<float> h_wq(E * E), h_wk(E * E), h_wv(E * E), h_wo(E * E);
    fill_random(h_x, 1.0f);
    fill_random(h_wq, 0.2f);
    fill_random(h_wk, 0.2f);
    fill_random(h_wv, 0.2f);
    fill_random(h_wo, 0.2f);

    float *d_x, *d_wq, *d_wk, *d_wv, *d_wo;
    float *d_q_linear, *d_k_linear, *d_v_linear;
    float *d_q, *d_k, *d_v;
    float *d_scores, *d_head_out, *d_context, *d_y;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wq, h_wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wk, h_wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wv, h_wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wo, h_wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_linear, tokens * E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_linear, tokens * E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_linear, tokens * E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q, B * H * S * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, B * H * S * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, B * H * S * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, B * H * S * S * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_out, B * H * S * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_context, tokens * E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, tokens * E * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wq, h_wq.data(), h_wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wk, h_wk.data(), h_wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wv, h_wv.data(), h_wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wo, h_wo.data(), h_wo.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 1) Q = X Wq, K = X Wk, V = X Wv
    row_major_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tokens, E, E, d_x, d_wq, d_q_linear);
    row_major_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tokens, E, E, d_x, d_wk, d_k_linear);
    row_major_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tokens, E, E, d_x, d_wv, d_v_linear);

    // 2) [B, S, E] -> [B, H, S, D]
    const int block = 256;
    const int total = B * S * E;
    split_heads_kernel<<<CEIL_DIV(total, block), block>>>(d_q_linear, d_q, B, S, H, D);
    split_heads_kernel<<<CEIL_DIV(total, block), block>>>(d_k_linear, d_k, B, S, H, D);
    split_heads_kernel<<<CEIL_DIV(total, block), block>>>(d_v_linear, d_v, B, S, H, D);
    CHECK_CUDA(cudaGetLastError());

    // 3) scores = Q K^T / sqrt(D), independently for each (batch, head)
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    row_major_gemm_strided_batched(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_T,
                                   S, S, D,
                                   d_q, S * D,
                                   d_k, S * D,
                                   d_scores, S * S,
                                   head_batches,
                                   scale);

    // 4) attn = softmax(scores), in-place
    softmax_rows_kernel<256><<<B * H * S, 256>>>(d_scores, B * H * S, S);
    CHECK_CUDA(cudaGetLastError());

    // 5) head_out = attn V
    row_major_gemm_strided_batched(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   S, D, S,
                                   d_scores, S * S,
                                   d_v, S * D,
                                   d_head_out, S * D,
                                   head_batches);

    // 6) [B, H, S, D] -> [B, S, E], then final output projection Y = context Wo
    merge_heads_kernel<<<CEIL_DIV(total, block), block>>>(d_head_out, d_context, B, S, H, D);
    CHECK_CUDA(cudaGetLastError());
    row_major_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tokens, E, E, d_context, d_wo, d_y);

    std::vector<float> h_attn(B * H * S * S);
    std::vector<float> h_y(tokens * E);
    CHECK_CUDA(cudaMemcpy(h_attn.data(), d_scores, h_attn.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "first attention row: ";
    float row_sum = 0.0f;
    for (int i = 0; i < S; ++i) {
        std::cout << h_attn[i] << " ";
        row_sum += h_attn[i];
    }
    std::cout << "\nrow sum: " << row_sum << std::endl;

    std::cout << "first output token: ";
    for (int i = 0; i < E; ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_wq));
    CHECK_CUDA(cudaFree(d_wk));
    CHECK_CUDA(cudaFree(d_wv));
    CHECK_CUDA(cudaFree(d_wo));
    CHECK_CUDA(cudaFree(d_q_linear));
    CHECK_CUDA(cudaFree(d_k_linear));
    CHECK_CUDA(cudaFree(d_v_linear));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_head_out));
    CHECK_CUDA(cudaFree(d_context));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
