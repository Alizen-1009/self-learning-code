__device__ int find_root(int* parent, int x) {
    int p = parent[x];

    // path halving
    while (p != parent[p]) {
        int gp = parent[p];
        parent[x] = gp;   // 可以非原子，优化路径压缩；正确性主要靠 union 里的 atomicCAS
        x = gp;
        p = parent[x];
    }

    return p;
}

__device__ void unite(int* parent, int a, int b) {
    while (true) {
        int ra = find_root(parent, a);
        int rb = find_root(parent, b);

        if (ra == rb) return;


        // 为了避免环，固定让大 root 指向小 root
        int high = max(ra, rb);
        int low = min(ra, rb);

        // 只有当 parent[high] 仍然是 high 时，才把它挂到 low
        int old = atomicCAS(&parent[high], high, low);

        if (old == high) {
            return;  // union 成功
        }

        // 否则说明别的线程改过 parent[high]，重试
    }
}



__global__ void compress_all(int* parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        parent[i] = find_root(parent, i);
    }
}
