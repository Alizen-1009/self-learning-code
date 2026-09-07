template<const int BLOCK_SIZE = 32>
__global__ void matmul(float* A, float* B, float* C, int N, int M, int K) {
    int bx = blockDim.x, by = blockDim.y;
    int tx = blockDim.x, ty = blcokDim.y;

    int col = bx * BLOCK_SIZE + tx;
    int row = by * BLOCK_SIZE + ty;

    __sharded__ As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __sharded__ Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    float acc = 0.0f;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        float a_col = k + tx;
        float b_row = k + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * M + col] : 0;

        __syncthreads();


        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc += A[ty][k] * B[k][tx];
        }
        __syncthreads();
    }
    if (row < M * *col < N) c[row * N + col] = acc;
}