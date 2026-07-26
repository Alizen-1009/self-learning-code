#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
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

#define CHECK_CUDNN(call)                                                       \
    do {                                                                        \
        cudnnStatus_t stat = (call);                                             \
        if (stat != CUDNN_STATUS_SUCCESS) {                                      \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(stat)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

static void print_matrix(const char* name, const std::vector<float>& x,
                         int rows, int cols) {
    std::cout << name << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << x[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

static void run_cublas_gemm_demo() {
    // Row-major: A is [2, 4], B is [4, 3], C is [2, 3].
    const int M = 2;
    const int N = 3;
    const int K = 4;

    std::vector<float> h_a = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    std::vector<float> h_b = {
        1, 0, 2,
        0, 1, 2,
        1, 1, 0,
        2, 0, 1,
    };
    std::vector<float> h_c(M * N, 0.0f);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major layout. Reversing A/B computes:
    // row-major C = A * B  <=>  column-major C^T = B^T * A^T.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_b, N,
                             d_a, K,
                             &beta,
                             d_c, N));

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    print_matrix("cuBLAS GEMM C = A * B:", h_c, M, N);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

static void run_cudnn_conv_demo() {
    // NCHW input: [1, 1, 3, 3].
    std::vector<float> h_x = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    // KCRS filter: [1, 1, 2, 2].
    std::vector<float> h_w = {
        1, 0,
        0, -1,
    };
    std::vector<float> h_y(1 * 1 * 2 * 2, 0.0f);

    float *d_x, *d_w, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, 1, 1, 3, 3));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, 1, 1, 2, 2));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                0, 0,  // pad_h, pad_w
                                                1, 1,  // stride_h, stride_w
                                                1, 1,  // dilation_h, dilation_w
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc, &out_n, &out_c, &out_h, &out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));

    const cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_bytes));

    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(handle,
                                        &alpha,
                                        x_desc, d_x,
                                        w_desc, d_w,
                                        conv_desc,
                                        algo,
                                        workspace, workspace_bytes,
                                        &beta,
                                        y_desc, d_y));

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    print_matrix("cuDNN Conv2D output:", h_y, out_h, out_w);

    if (workspace != nullptr) {
        CHECK_CUDA(cudaFree(workspace));
    }
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
}

int main() {
    run_cublas_gemm_demo();
    run_cudnn_conv_demo();
    return 0;
}
