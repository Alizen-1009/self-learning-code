/*
题目：区间加、区间求和线段树模板

【功能】给定一个整数数组，支持两种闭区间操作：
1. add(l,r,delta)：把 [l,r] 中每个数都加上 delta；
2. sum(l,r)：返回 [l,r] 中所有数的和。

【方法】每个节点保存负责区间的和，以及尚未传给子节点的 lazy 增量。整段被覆盖时，
节点和增加 delta*区间长度，同时累计 lazy。访问部分子区间前先 push，把标记传给
两个孩子，再递归处理，最后用孩子的和更新父节点。

原草稿的 build 在叶子没有 return，会继续递归；modify 对“区间和”取 max 并没有
明确定义，且缺少下传标记。这里把接口统一成可验证的“区间加、区间求和”。

复杂度：建树 O(n)，每次更新或查询 O(log n)，空间 O(n)。main 使用一个小例子演示。

English: A lazy segment tree for range addition and range sums. Every node
stores its segment sum and a pending increment. Full-cover updates add
delta*length; partial operations first propagate the lazy value. Build is
O(n), each operation O(log n), and storage O(n).
*/

#include <iostream>
#include <vector>

using namespace std;
using int64 = long long;

class SegmentTree {
    struct Node {
        int64 sum = 0;
        int64 lazy = 0;
    };

    int size_;
    vector<Node> tree_;

    void apply(int node, int left, int right, int64 delta) {
        tree_[node].sum += delta * (right - left + 1);
        tree_[node].lazy += delta;
    }

    void push(int node, int left, int right) {
        if (tree_[node].lazy == 0 || left == right) return;
        const int middle = left + (right - left) / 2;
        apply(node * 2, left, middle, tree_[node].lazy);
        apply(node * 2 + 1, middle + 1, right, tree_[node].lazy);
        tree_[node].lazy = 0;
    }

    void build(int node, int left, int right, const vector<int64>& values) {
        if (left == right) {
            tree_[node].sum = values[left];
            return;
        }
        const int middle = left + (right - left) / 2;
        build(node * 2, left, middle, values);
        build(node * 2 + 1, middle + 1, right, values);
        tree_[node].sum = tree_[node * 2].sum + tree_[node * 2 + 1].sum;
    }

    void add(int node, int left, int right,
             int queryLeft, int queryRight, int64 delta) {
        if (queryLeft <= left && right <= queryRight) {
            apply(node, left, right, delta);
            return;
        }
        push(node, left, right);
        const int middle = left + (right - left) / 2;
        if (queryLeft <= middle)
            add(node * 2, left, middle, queryLeft, queryRight, delta);
        if (queryRight > middle)
            add(node * 2 + 1, middle + 1, right, queryLeft, queryRight, delta);
        tree_[node].sum = tree_[node * 2].sum + tree_[node * 2 + 1].sum;
    }

    int64 sum(int node, int left, int right,
              int queryLeft, int queryRight) {
        if (queryLeft <= left && right <= queryRight)
            return tree_[node].sum;
        push(node, left, right);
        const int middle = left + (right - left) / 2;
        int64 result = 0;
        if (queryLeft <= middle)
            result += sum(node * 2, left, middle, queryLeft, queryRight);
        if (queryRight > middle)
            result += sum(node * 2 + 1, middle + 1, right, queryLeft, queryRight);
        return result;
    }

public:
    explicit SegmentTree(const vector<int64>& values)
        : size_(static_cast<int>(values.size())), tree_(size_ * 4) {
        if (size_ > 0) build(1, 0, size_ - 1, values);
    }

    void add(int left, int right, int64 delta) {
        add(1, 0, size_ - 1, left, right, delta);
    }

    int64 sum(int left, int right) {
        return sum(1, 0, size_ - 1, left, right);
    }
};

int main() {
    SegmentTree tree({1, 2, 3, 4});
    cout << tree.sum(0, 3) << '\n';  // 10
    tree.add(1, 2, 5);
    cout << tree.sum(0, 3) << '\n';  // 20
    cout << tree.sum(1, 2) << '\n';  // 15
    return 0;
}
