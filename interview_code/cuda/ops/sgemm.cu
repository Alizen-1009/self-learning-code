#include <common.cuh>

template <const int BLOCKSIZE = 32>
__global__ void matmul_kernel_simple(const float* A, const float* B, float* C, int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BLOCKSIZE) {
        const int a_col = k0 + tx;
        const int b_row = k0 + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TY = 8, const int TX = 8>
__global__ void matmul_kernel_v0(const float* __restrict__ A_ptr, const float* __restrict__ B_ptr,
    float* __restrict__ C_ptr, const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ float a_shared[BK][BM];
    __shared__ float b_shared[BK][BN];
    float acc_m[TY][TX] = { 0.f };
    float reg_a[TY] = { 0.f };
    float reg_b[TX] = { 0.f };
    float ldg_a[4] = { 0.f };

    const float* A_start_ptr = A_ptr + blockIdx.y * BM * K;
    const float* B_start_ptr = B_ptr + blockIdx.x * BN;

    int A_row_per_thread = BK / 4;
    int B_row_per_thread = BN / 4;

    int A_chunk_y = tid / A_row_per_thread;
    int A_chunk_x = tid % A_row_per_thread;
    int B_chunk_y = tid / B_row_per_thread;
    int B_chunk_x = tid % B_row_per_thread;

    for (int tile = 0; tile < K; tile += BK) {
        FLOAT4(ldg_a[0]) = FETCH_CFLOAT4(A_start_ptr[A_chunk_y * K + A_chunk_x * 4 + tile]);
        FLOAT4(b_shared[B_chunk_y][B_chunk_x * 4]) =
            FETCH_CFLOAT4(B_start_ptr[(B_chunk_y + tile) * N + B_chunk_x * 4]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a_shared[A_chunk_x * 4 + i][A_chunk_y] = ldg_a[i];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k++) {
            FLOAT4(reg_a[0]) = FLOAT4(a_shared[k][ty * TY]);
            FLOAT4(reg_a[4]) = FLOAT4(a_shared[k][ty * TY + 4]);
            FLOAT4(reg_b[0]) = FLOAT4(b_shared[k][tx * TX]);
            FLOAT4(reg_b[4]) = FLOAT4(b_shared[k][tx * TX + 4]);
#pragma unroll
            for (int i = 0; i < TY; i++) {
#pragma unroll
                for (int j = 0; j < TX; j++) {
                    acc_m[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    float* C_ptr_start = C_ptr + N * by * BM + bx * BN;
    for (int i = 0; i < TY; i++) {
        FLOAT4(C_ptr_start[N * (ty * TY + i) + tx * TX]) = FLOAT4(acc_m[i][0]);
        FLOAT4(C_ptr_start[N * (ty * TY + i) + tx * TX + 4]) = FLOAT4(acc_m[i][4]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TY = 8, const int TX = 8>
__global__ void matmul_kernel_v1(const float* __restrict__ A_ptr, const float* __restrict__ B_ptr,
    float* __restrict__ C_ptr, const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ float a_shared[2][BK][BM];
    __shared__ float b_shared[2][BK][BN];
    float acc_m[TY][TX] = { 0.f };
    float reg_a[TY] = { 0.f };
    float reg_b[TX] = { 0.f };
    float ldg_a[4] = { 0.f };

    const float* A_start_ptr = A_ptr + blockIdx.y * BM * K;
    const float* B_start_ptr = B_ptr + blockIdx.x * BN;

    int A_row_per_thread = BK / 4;
    int B_row_per_thread = BN / 4;

    int A_chunk_y = tid / A_row_per_thread;
    int A_chunk_x = tid % A_row_per_thread;
    int B_chunk_y = tid / B_row_per_thread;
    int B_chunk_x = tid % B_row_per_thread;

    FLOAT4(ldg_a[0]) = FETCH_CFLOAT4(A_start_ptr[A_chunk_y * K + A_chunk_x * 4]);
    FLOAT4(b_shared[0][B_chunk_y][B_chunk_x * 4]) = FETCH_CFLOAT4(B_start_ptr[B_chunk_y * N + B_chunk_x * 4]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a_shared[0][A_chunk_x * 4 + i][A_chunk_y] = ldg_a[i];
    }
    __syncthreads();

    int write_stage_idx = 1;
    for (int tile = BK; tile < K; tile += BK) {
        FLOAT4(ldg_a[0]) = FETCH_CFLOAT4(A_start_ptr[A_chunk_y * K + A_chunk_x * 4 + tile]);
        FLOAT4(b_shared[write_stage_idx][B_chunk_y][B_chunk_x * 4]) =
            FETCH_CFLOAT4(B_start_ptr[(B_chunk_y + tile) * N + B_chunk_x * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            a_shared[write_stage_idx][A_chunk_x * 4 + i][A_chunk_y] = ldg_a[i];
        }

        write_stage_idx ^= 1;
#pragma unroll
        for (int k = 0; k < BK; k++) {
            FLOAT4(reg_a[0]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY]);
            FLOAT4(reg_a[4]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY + 4]);
            FLOAT4(reg_b[0]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX]);
            FLOAT4(reg_b[4]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX + 4]);
#pragma unroll
            for (int i = 0; i < TY; i++) {
#pragma unroll
                for (int j = 0; j < TX; j++) {
                    acc_m[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    write_stage_idx ^= 1;
#pragma unroll
    for (int k = 0; k < BK; k++) {
        FLOAT4(reg_a[0]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY]);
        FLOAT4(reg_a[4]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY + 4]);
        FLOAT4(reg_b[0]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX]);
        FLOAT4(reg_b[4]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX + 4]);

#pragma unroll
        for (int i = 0; i < TY; i++) {
#pragma unroll
            for (int j = 0; j < TX; j++) {
                acc_m[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    float* C_ptr_start = C_ptr + N * by * BM + bx * BN;
    for (int i = 0; i < TY; i++) {
        FLOAT4(C_ptr_start[N * (ty * TY + i) + tx * TX]) = FLOAT4(acc_m[i][0]);
        FLOAT4(C_ptr_start[N * (ty * TY + i) + tx * TX + 4]) = FLOAT4(acc_m[i][4]);
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid(CEIL(N, BLOCKSIZE), CEIL(M, BLOCKSIZE));
    matmul_kernel_simple<BLOCKSIZE> << <grid, block >> > (A, B, C, M, N, K);

    cudaDeviceSynchronize();
}

__global__ vodi matmul(float* A, float* B, float* C)
