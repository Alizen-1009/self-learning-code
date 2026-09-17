#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 教学版 inclusive scan：out[i] = in[0] + ... + in[i]。
// 使用 long long；要求所有中间加法均不溢出。
using Value = long long;
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 每个 block 负责一段连续数据，计算局部前缀和与本块总和。
__global__ void scan_blocks(const Value* in, Value* out,
                            Value* block_sums, int n) {
    __shared__ Value data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;
    data[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    // Hillis-Steele scan：依次累加距离为 1、2、4、8……的前驱。
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        Value previous = (tid >= offset) ? data[tid - offset] : 0;
        // 所有线程先读完上一轮的数据，再允许任何线程修改它。
        __syncthreads();
        data[tid] += previous;
        // 等本轮全部写完，才能开始下一轮读取。
        __syncthreads();
    }

    if (i < n) out[i] = data[tid];
    // 末尾无效位置填了 0，所以最后一个槽位仍是本块总和。
    if (tid == BLOCK_SIZE - 1) block_sums[blockIdx.x] = data[tid];
}

__global__ void add_offsets(Value* out, const Value* scanned_sums, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    // scanned_sums 是 inclusive scan，前一块的值就是本块偏移。
    if (i < n && blockIdx.x > 0) {
        out[i] += scanned_sums[blockIdx.x - 1];
    }
}

// 主机函数：参数是设备指针，n >= 0；所有 kernel 使用同一默认 stream。
// 数据分块 -> 递归扫描块总和 -> 加回偏移。单块时递归结束。
void prefix_sum(const Value* d_in, Value* d_out, int n) {
    if (n <= 0) return;
    int blocks = 1 + (n - 1) / BLOCK_SIZE;
    Value* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums, static_cast<size_t>(blocks) * sizeof(Value)));
    scan_blocks<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (blocks > 1) {
        Value* d_scanned_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scanned_sums,
                              static_cast<size_t>(blocks) * sizeof(Value)));
        prefix_sum(d_sums, d_scanned_sums, blocks);
        add_offsets<<<blocks, BLOCK_SIZE>>>(d_out, d_scanned_sums, n);
        CUDA_CHECK(cudaGetLastError());
        // 教学版显式等待，保证临时缓冲区使用结束后再释放。
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_scanned_sums));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_sums));
}

bool check(const std::vector<Value>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) {
        prefix_sum(nullptr, nullptr, 0);
        std::printf("N=0: PASS\n");
        return true;
    }

    size_t bytes = input.size() * sizeof(Value);
    Value *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));
    prefix_sum(d_in, d_out, n);
    std::vector<Value> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    Value expected = 0;
    for (int i = 0; i < n; i++) {
        expected += input[i];
        if (output[i] != expected) {
            std::fprintf(stderr, "N=%d, i=%d: got %lld, expected %lld\n",
                         n, i, output[i], expected);
            return false;
        }
    }
    std::printf("N=%d: PASS\n", n);
    if (n == 8) {
        for (Value value : output) std::printf("%lld ", value);
        std::printf("\n");
    }
    return true;
}

int main() {
    if (!check({1, 2, 3, 4, 5, 6, 7, 8})) return EXIT_FAILURE;
    // 覆盖空输入、单元素、块边界、多层递归，以及非整块尾部。
    for (int n : {0, 1, 255, 256, 257, 65536, 65537, 1000003}) {
        std::vector<Value> input(n);
        for (int i = 0; i < n; i++) input[i] = i % 17 - 8;
        if (!check(input)) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
