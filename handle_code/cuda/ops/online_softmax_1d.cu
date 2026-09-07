#include <cuda_runtime.h>
#include <float.h>

// 合并整个 warp 的 (最大值 m, 指数和 d)，只有 lane 0 得到完整结果。
__device__ __forceinline__ void warp_reduce_md(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_m = __shfl_down_sync(0xffffffffu, m, offset);
        float other_d = __shfl_down_sync(0xffffffffu, d, offset);
        float new_m = fmaxf(m, other_m);
        d = d * expf(m - new_m) + other_d * expf(other_m - new_m);
        m = new_m;
    }
}

// x、y 均为一维 FP32 数组 [N]，只启动一个 block。
// 假设 N > 0、x 为有限数、int 索引不溢出。
// block 为一维，线程数是 32 的倍数，且不超过 1024。
// 调用：online_softmax_1d<<<1, 256>>>(x, y, N);
__global__ void online_softmax_1d(const float* x, float* y, int N) {
    int tid = threadIdx.x;
    int laneid = tid % 32;
    int warpid = tid / 32;
    int num_warps = blockDim.x / 32;

    __shared__ float shared_m[32];
    __shared__ float shared_d[32];

    // 1. 一次遍历同时维护最大值和指数和。
    // d 始终表示已处理元素的 sum(exp(x[i] - m))。
    // 使用 -FLT_MAX，使空线程之间合并时不会出现 -inf - (-inf)。
    float m = -FLT_MAX;
    float d = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = x[i];
        float new_m = fmaxf(m, v);
        d = d * expf(m - new_m) + expf(v - new_m);
        m = new_m;
    }

    // 2. warp 内合并 (m, d)，不能分别对 m 求 max、对 d 求 sum。
    warp_reduce_md(m, d);
    if (laneid == 0) {
        shared_m[warpid] = m;
        shared_d[warpid] = d;
    }
    __syncthreads();

    // 3. 第一个 warp 合并所有 warp 的统计量。
    if (warpid == 0) {
        m = laneid < num_warps ? shared_m[laneid] : -FLT_MAX;
        d = laneid < num_warps ? shared_d[laneid] : 0.f;
        warp_reduce_md(m, d);
        if (laneid == 0) {
            shared_m[0] = m;
            shared_d[0] = d;
        }
    }
    __syncthreads();

    // 4. 已知全局统计量后，再遍历一次输入，写出 softmax。
    m = shared_m[0];
    d = shared_d[0];
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - m) / d;
    }
}
