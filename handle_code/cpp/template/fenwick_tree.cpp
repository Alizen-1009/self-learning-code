/*
题目：树状数组模板及逆序对示例
【功能】支持单点增加、前缀和、区间和。数组位置必须从 1 开始；对位置 0 做更新
会使 lowbit(0)=0，指针无法前进。
【方法】lowbit(x)=x&-x 是当前节点负责的区间长度。更新时反复加 lowbit，查询时
反复减 lowbit。示例将原数组先离散化成从 1 开始的排名，再从右往左扫描，查询
“已插入且比当前值小”的数量并累加，得到严格逆序对数。答案可能为 O(n^2)，用
long long 保存。复杂度 O(n log n) 时间、O(n) 空间。

English: Fenwick tree template supporting point addition, prefix sums, and range
sums. Updates move upward by lowbit; queries accumulate while moving downward.
The demo coordinate-compresses values to positive ranks, then scans from right
to left to count strictly smaller suffix values. O(n log n) time, O(n) space.
*/
#include <bits/stdc++.h>
using namespace std;
class FenwickTree {
    vector<long long> tree;

public:
    explicit FenwickTree(int n) : tree(n + 1, 0) {}

    void add(int index, long long delta) {
        for (int n = static_cast<int>(tree.size()); index < n;
             index += index & -index) {
            tree[index] += delta;
        }
    }

    long long prefixSum(int index) const {
        long long result = 0;
        for (; index > 0; index -= index & -index) {
            result += tree[index];
        }
        return result;
    }

    long long rangeSum(int left, int right) const {
        return prefixSum(right) - prefixSum(left - 1);
    }
};

int main() {
    vector<int> a(10);
    for (int i = 0; i < 10; i++) a[i] = i ^ 10;
    vector<int> sorted = a;
    sort(sorted.begin(), sorted.end());
    sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());

    FenwickTree tree(static_cast<int>(sorted.size()));
    long long inversions = 0;
    for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
        const int rank = static_cast<int>(
            lower_bound(sorted.begin(), sorted.end(), a[i]) - sorted.begin()) + 1;
        inversions += tree.prefixSum(rank - 1);
        tree.add(rank, 1);
    }
    cout << inversions << '\n';
    return 0;
}
