// template<const int BLOCK_SIZE = 32>
// __global__ void matmul(float* A, float* B, float* C, int N, int M, int K) {
//     int bx = blockDim.x, by = blockDim.y;
//     int tx = blockDim.x, ty = blcokDim.y;

//     int col = bx * BLOCK_SIZE + tx;
//     int row = by * BLOCK_SIZE + ty;

//     __sharded__ As[BLOCK_SIZE][BLOCK_SIZE + 1];
//     __sharded__ Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

//     float acc = 0.0f;
//     for (int k = 0; k < K; k += BLOCK_SIZE) {
//         float a_col = k + tx;
//         float b_row = k + ty;

//         As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
//         Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * M + col] : 0;

//         __syncthreads();


//         for (int i = 0; i < BLOCK_SIZE; i++) {
//             acc += A[ty][k] * B[k][tx];
//         }
//         __syncthreads();
//     }
//     if (row < M * *col < N) c[row * N + col] = acc;
// }




#include <sys/types.h>
const int N = 1024, M = 1024;
int ceil(int a, int b) {
    return (a + b - 1) / b;
}
float A[M + 1][N + 1], B[N + 1][M + 1];
template<const int BLOCK_SIZE = 32>
__global__ void transpose(float* A, float* B, int N, int M) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = BlockDim.x, by = BlockDim.y;

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;

    __shard__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1];

    if (x < N && y < M) {
        sdata[tx][ty] = A[y * N + x];
    }
    __syncthreads();

    int x2 = bx * BLOCK_SIZE + ty;
    int y2 = by * BLOCK_SIZE + tx;
    B[x2 * M + y2] = sdata[ty][tx];
}

int main() {
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            A[i][j] = i + j;
        }
    }
    dim3 grid(ceil(M, BLOCK_SIZE), ceil(N, BLOCK_SIZE));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    transpose << <grid, block >> > (&A, &B, N, M);

}

