
const int RADIX_BITS = 4;
const int RADIX_SIZE = 1 << (RADIX_BITS);
const int RADIX_MASK = RADIX_SIZE - 1;
template<const int THREAD_PER_BLOCK>
__global__ void TopK(const int* in, int N, int K, int* ans) {
    __shared__ int kth;
    __shared__ int count[RADIX_SIZE];
    __shared__ unsigned int  prefix, prefixMask;

    int tid = threadIdx.x;
    if (tid == 0) {
        kth = K;
        prefix = 0;
        prefixMask = 0;
    }
    __syncthreads();

    for (int shift = sizeof(int) * 8 - RADIX_BITS; ~shift; shift -= RADIX_BITS) {
        if (tid < RADIX_SIZE) {
            count[tid] = 0;
        }
        __syncthread();
        for (int i = tid; i < N; i += THREAD_PER_BLOCK) {
            int x = in[i];
            if ((x & prefixMask) == prefix) {
                int dight = (x >> shift) & RADIX_MASK;
                atomicAdd(&count[digit], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            for (int i = RADIX_SIZE - 1; ~i; i--) {
                int c = count[i];
                if (kth > c) {
                    kth -= c;
                    continue;
                }

                prefix |= ((unsigned int)i << shift);
                prefixMask |= (RADIX_MASK << shift);
                break;
            }
        }
        __syncthreads();
    }
    if (tid == 0) *ans = prefix;
}