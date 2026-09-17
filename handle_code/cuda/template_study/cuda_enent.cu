// CUDA Event：用一个完成标记，让两个 stream 按需要的顺序执行。
// 编译：nvcc cuda_enent.cu -o /tmp/cuda_event_demo
// 运行：/tmp/cuda_event_demo
//
// stream1: 写 x = 42 -> record(ready)
//                              |
// stream2:                wait(ready) -> 读取 x
//
// Event 不锁住 x，只限制等待它的 stream 中后续工作的执行顺序。
// stream wait 不需要 CPU 等待，也不是启动一个 CUDA 线程占着 SM 自旋。
// start/stop 通常只是事件的变量名，没有 cudaEventStart() 这个 API。

#include <cuda_runtime.h>
#include <cstdio>

__global__ void write_value(int* x) {
    *x = 42;
}

__global__ void read_value(const int* x) {
    printf("x = %d\n", *x);  // 预期输出：x = 42
}

int main() {
    int* x;
    cudaMalloc(&x, sizeof(int));

    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

    cudaEvent_t ready;
    cudaEventCreate(&ready);

    // 1. 在 stream1 写数据，然后记录完成点。
    write_value<<<1, 1, 0, stream1>>>(x);
    cudaEventRecord(ready, stream1);

    // 2. stream2 后续的读取必须等 ready 完成；CPU 可以继续往下走。
    // 注意：CPU 端先调用 record，再调用 wait。
    cudaStreamWaitEvent(stream2, ready, 0);
    read_value<<<1, 1, 0, stream2>>>(x);

    // 3. CPU 等 stream2 完成，保证读取结束后才释放资源。
    cudaStreamSynchronize(stream2);
    // 如果只想让 CPU 等写入完成，可以用 cudaEventSynchronize(ready)。
    // 但 ready 只覆盖写入，不能用它来保证 stream2 的读取已完成。

    cudaEventDestroy(ready);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(x);
    return 0;
}
