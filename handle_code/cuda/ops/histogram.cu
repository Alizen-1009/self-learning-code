#include <cuda_runtime.h>

// 每个 block 先统计到共享内存，再合并到全局直方图。
// hist 必须预先清零；只统计 [0, bins) 内的整数，计数使用 int。
__global__ void histogram_kernel(const int* data, int* hist, int n, int bins) {
    extern __shared__ int s_hist[];
    int tid = threadIdx.x;

    for (int b = tid; b < bins; b += blockDim.x)
        s_hist[b] = 0;
    __syncthreads();

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < static_cast<size_t>(n); i += stride) {
        int value = data[i];
        if (value >= 0 && value < bins)
            atomicAdd(&s_hist[value], 1);
    }
    __syncthreads();

    for (int b = tid; b < bins; b += blockDim.x)
        atomicAdd(&hist[b], s_hist[b]);
}

// data、hist 均为 GPU 指针，hist 长度为 bins。
// bins * sizeof(int) 必须在设备的单 block 共享内存限额内。
cudaError_t histogram(const int* data, int* hist, int n, int bins,
                      cudaStream_t stream = 0) {
    if (n < 0 || bins <= 0) return cudaErrorInvalidValue;
    size_t shared_bytes = static_cast<size_t>(bins) * sizeof(int);
    cudaError_t err = cudaMemsetAsync(hist, 0, shared_bytes, stream);
    if (err != cudaSuccess || n == 0) return err;

    constexpr int threads = 256;
    int blocks = 1 + (n - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    histogram_kernel<<<blocks, threads, shared_bytes, stream>>>(data, hist, n, bins);
    return cudaGetLastError();
}
