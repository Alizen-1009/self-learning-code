#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;
struct __align__(8) MD {
    float mx;
    float sum;
};

__device__ __forceinline__ MD warp_reduce_softmax(MD val) {
    for (int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        MD other;
        other.mx = __shfl_down_sync(0xffffffff, val.mx, mask);
        other.sum = __shfl_down_sync(0xffffffff, val.sum, mask);
        float pre_mx = val.mx;
        val.mx = max(val.mx, other.mx);
        val.sum = val.sum * __expf(pre_mx - val.mx) + other.sum * __expf(other.mx - val.mx);
    }
    return val;
}
__global__ void onlinesoftmax(float* in, int N) {
    int tid = threadIdx.x;

    int warpid = tid / WARP_SIZE;
    int laneid = tid % WARP_SIZE;

    int WARP_NUM = blockDim.x / WARP_SIZE;
    __shared__ MD sdata[WARP_SIZE];

    MD val{-FLT_MAX, 0.0f};
    for (int i = tid; i < N; i += blockDim.x) {
        float tmp_mx = in[i];
        float tmp_sum = 1;

        float pre_mx = val.mx;
        val.mx = max(val.mx, tmp_mx);
        val.sum = val.sum * __expf(pre_mx - val.mx) + tmp_sum * __expf(tmp_mx - val.mx);
    }

    val = warp_reduce_softmax(val);
    if (laneid == 0) sdata[warpid] = val;

    __syncthreads();
    if (warpid == 0) {
        val = laneid < WARP_NUM ? sdata[laneid] : MD{-FLT_MAX, 0.0f};
        val = warp_reduce_softmax(val);
        if (tid == 0) sdata[0] = val;
    }
    __syncthreads();

    float scale = 1.0f / sdata[0].sum;
    float mx = sdata[0].mx;
    for (int i = tid; i < N; i += blockDim.x) {
        in[i] = __expf(in[i] - mx) * scale;
    }
}

int main() {
    const int N = 4;
    float h[N] = {1, 2, 3, 4};
    float* d;

    cudaMalloc((void**)&d, sizeof(h));
    cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice);

    onlinesoftmax<<<1, BLOCK_SIZE>>>(d, N);

    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) printf("%.6f ", h[i]);
    printf("\n");

    cudaFree(d);
    return 0;
}
