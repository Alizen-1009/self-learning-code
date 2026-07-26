#include <float.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float WarpReduceMax(float value) {
    for (int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
    }
    return value;
}

template <const int THREADS_PER_BLOCK = 1024, const int WARP_SIZE = 32>
__global__ void reduce_max_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int warpid = tid / WARP_SIZE;
    int laneid = tid % WARP_SIZE;
    int WARP_NUM = CEIL(blockDim.x, WARP_SIZE);

    __shared__ float dshard[32];
    float value = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        value = fmaxf(value, input[i]);
    }
    value = WarpReduceMax(value);
    if (laneid == 0)
        dshard[warpid] = value;
    __syncthreads();

    if (warpid == 0) {
        value = (laneid < WARP_NUM) ? dshard[laneid] : -FLT_MAX;
        value = WarpReduceMax(value);
    }

    if (threadIdx.x == 0)
        *output = value;
}
