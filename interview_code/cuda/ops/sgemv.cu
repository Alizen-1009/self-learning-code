template <const int WARP_SIZE = 32>
__global__ void sgemv_32_kernel(const float* A, const float* x, float* res, int M, int N) {
    int bid = blockIdx.x;
    int row = bid * blockDim.y + threadIdx.y;

    if (row < M) {
        float sum = 0;
        for (int col = threadIdx.x; col < N; col += blockDim.x) {
            sum += A[row * N + col] * x[col];
        }
        sum = DynamicWarpReduceSum<WARP_SIZE>(sum);
        if (threadIdx.x == 0) res[row] = sum;
    }
}


const int WARP_SIZE = 32;
const int THREAD_PER_BLOCK = 128;
const int WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
const int ROW_PER_BLOCK = WARP_PER_BLOCK;
dim3 grid(CEIL(M, ROW_PER_BLOCK));
dim3 block(WARP_SIZE, THREAD_PER_BLOCK / WARP_SIZE);
sgemv_32_kernel << <grid, block >> > (A, x, res, M, N);