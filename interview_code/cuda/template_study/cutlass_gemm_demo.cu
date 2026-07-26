#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

#define CHECK_CUTLASS(status)                                                   \
    do {                                                                        \
        cutlass::Status s = (status);                                            \
        if (s != cutlass::Status::kSuccess) {                                    \
            std::cerr << "CUTLASS error code: " << int(s)                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

static void fill_random(std::vector<float>& x) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : x) v = dist(gen);
}

int main() {
    const int M = 64;
    const int N = 64;
    const int K = 64;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    fill_random(h_A);
    fill_random(h_B);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice));

    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, RowMajor,
        float, RowMajor,
        float, RowMajor>;

    CutlassGemm gemm;

    float alpha = 1.0f;
    float beta = 0.0f;
    CutlassGemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {alpha, beta});

    CHECK_CUTLASS(gemm.can_implement(args));
    CHECK_CUTLASS(gemm(args));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "C[0,0] = " << h_C[0] << "\n";
    std::cout << "C[0,1] = " << h_C[1] << "\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
