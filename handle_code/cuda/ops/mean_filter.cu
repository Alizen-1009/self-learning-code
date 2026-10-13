#include <cuda_runtime.h>

// 单通道二维均值滤波，越界坐标取最近的边界像素（复制边界）。
__global__ void mean_filter_kernel(const float* in, float* out,
                                   int width, int height, int radius) {
    extern __shared__ float tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x0 = blockIdx.x * blockDim.x;
    int y0 = blockIdx.y * blockDim.y;
    int tile_w = blockDim.x + 2 * radius;
    int tile_h = blockDim.y + 2 * radius;

    // 所有线程协作加载输出区域及周围的 halo。
    for (int j = ty; j < tile_h; j += blockDim.y) {
        for (int i = tx; i < tile_w; i += blockDim.x) {
            int sx = min(max(x0 + i - radius, 0), width - 1);
            int sy = min(max(y0 + j - radius, 0), height - 1);
            tile[j * tile_w + i] = in[static_cast<size_t>(sy) * width + sx];
        }
    }
    __syncthreads();

    int x = x0 + tx;
    int y = y0 + ty;
    // 必须在同步之后退出，边界 block 的线程也要参与加载。
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int window = 2 * radius + 1;
    for (int j = 0; j < window; ++j)
        for (int i = 0; i < window; ++i)
            sum += tile[(ty + j) * tile_w + tx + i];
    out[static_cast<size_t>(y) * width + x] = sum / (window * window);
}

// in、out 为不同的 GPU 缓冲区，存储连续的 height * width 个 float。
// 适合小半径；(16 + 2 * radius)^2 * sizeof(float) 必须在共享内存限额内。
cudaError_t mean_filter(const float* in, float* out, int width, int height,
                        int radius, cudaStream_t stream = 0) {
    if (width <= 0 || height <= 0 || radius < 0)
        return cudaErrorInvalidValue;

    dim3 block(16, 16);
    dim3 grid(1 + (width - 1) / 16, 1 + (height - 1) / 16);
    size_t tile_side = 16 + 2 * static_cast<size_t>(radius);
    size_t shared_bytes = tile_side * tile_side * sizeof(float);
    mean_filter_kernel<<<grid, block, shared_bytes, stream>>>(
        in, out, width, height, radius);
    return cudaGetLastError();
}
