#include <cuda_runtime.h>

using Value = long long;  // 假设前缀和不会溢出
constexpr int BLOCK = 256;

// 每个 block 做局部 inclusive scan，并写出本块总和。
__global__ void scan_blocks(const Value* in, Value* out,
    Value* sums, int n) {
    __shared__ Value data[BLOCK_SIZE];
    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + t;
    data[t] = i < n ? in[i] : 0;
    __syncthreads();

    for (int d = 1; d < BLOCK_SIZE; d *= 2) {
        Value x = t >= d ? data[t - d] : 0;
        __syncthreads();         // 确保本轮读取先完成
        data[t] += x;
        __syncthreads();         // 确保本轮写入先完成
    }

    if (i < n) out[i] = data[t];
    if (t == BLOCK_SIZE - 1) sums[blockIdx.x] = data[t];
}

// 把前面所有 block 的累计和加到当前 block。
__global__ void add_offsets(Value* out, const Value* scanned_sums, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n && blockIdx.x > 0)
        out[i] += scanned_sums[blockIdx.x - 1];
}

// 设备指针；流程：局部 scan -> 递归 scan 块总和 -> 加回偏移。
void prefix_sum(const Value* in, Value* out, int n) {
    if (n <= 0) return;

    int blocks = 1 + (n - 1) / BLOCK_SIZE;
    Value* sums;
    cudaMalloc((void**)&sums, blocks * sizeof(Value));
    scan_blocks << <blocks, BLOCK_SIZE >> > (in, out, sums, n);

    if (blocks > 1) {
        Value* scanned_sums;
        cudaMalloc((void**)&scanned_sums, blocks * sizeof(Value));
        prefix_sum(sums, scanned_sums, blocks);
        add_offsets << <blocks, BLOCK_SIZE >> > (out, scanned_sums, n);
        cudaDeviceSynchronize();
        cudaFree(scanned_sums);
    }
    else {
        cudaDeviceSynchronize();
    }

    cudaFree(sums);
}
