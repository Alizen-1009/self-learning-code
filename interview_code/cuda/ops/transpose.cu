#include <common.cuh>
template<const int BLOCK_SIZE = 32>
__global__ void transpose(float* input, float* output, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1];

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;

    if (x < N && y < M) {
        sdata[ty][tx] = input[y * N + x];
    }
    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (x < M && y < N) {
        output[y * M + x] = sdata[tx][ty];
    }
}

__global__ void transpose_swizzle(float* input, float* output, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE];

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;

    if (x < N && y < M) {
        sdata[ty][ty ^ tx] = input[y * N + x];
    }
    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (x < M && y < N) {
        output[y * M + x] = sdata[tx][tx ^ ty];
    }
}



dim3 block(32, 32);
dim3 grid(N / BLOCK_SIZE, M / BLOCK_SIZE)
transpose << <grid, block >> > (in, out, M, N)