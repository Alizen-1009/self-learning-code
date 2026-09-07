#pragma once
#include <cuda_runtime.h>

// 完整 warp 参与，只有 lane 0 得到总和。
__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// x、y、weight 均为 [N]，只启动一个 block 处理整个向量。
// 假设 N > 0，索引不溢出；一维 block，线程数为 32 的倍数。
// 调用：rms_norm<<<1, 256>>>(x, weight, y, 1e-6f, N);
__global__ void rms_norm(const float* x, const float* weight, float* y,
                         float eps, int N) {
    int tid = threadIdx.x;
    int warpid = tid / 32;
    int laneid = tid % 32;
    int num_warps = blockDim.x / 32;

    __shared__ float sdata[32];
    __shared__ float inv_rms;

    // 1. 每个线程累加平方和。
    float sum = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }

    // 2. warp 内归约，lane 0 写入共享内存。
    sum = warp_sum(sum);
    if (laneid == 0) sdata[warpid] = sum;
    __syncthreads();

    // 3. 第一个 warp 汇总，线程 0 计算归一化系数。
    if (warpid == 0) {
        sum = laneid < num_warps ? sdata[laneid] : 0.f;
        sum = warp_sum(sum);
        if (laneid == 0) inv_rms = rsqrtf(sum / N + eps);
    }
    __syncthreads();

    // 4. 所有线程使用同一个系数计算输出。
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}
