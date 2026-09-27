/*
中文说明：这是一个线段树练习草稿，目标是维护区间和，并提供区间查询和区间 chmax
形式的更新接口；当前版本不是可直接使用的正确模板。
现有结构：node 保存区间边界与 val，operator+ 合并区间和，query 递归查询。
重要问题：build 在叶子节点赋值后缺少 return，会继续无限递归；modify 对整段只改
父节点 val、没有下传懒标记，之后查询子区间会不一致。使用前必须修复这两点。
正确线段树通常建树 O(n)，单次查询/合适的带标记更新 O(log n)，空间 O(n)。

English: An unfinished segment-tree exercise intended for range sums, queries,
and range-chmax-like updates; it is not currently a correct reusable template.
build lacks a return at leaves, and modify changes a parent without lazy
propagation, making child queries inconsistent. Fix both before use. A correct
tree normally builds in O(n), uses O(log n) per supported operation, and O(n) space.
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 2e5 + 7;
int a[N];
struct node {
    int l, r;
    int val;
}t[N << 2];
node operator + (const node& A, const node& B) {
    node C;
    C.l = A.l, C.r = B.r;
    C.val = A.val + B.val;
    return C;
}
void build(int l, int r, int x = 1) {
    if (l == r) {
        t[x].l = l, t[x].r = r;
        t[x].val = a[l];
    }
    int mid = l + r >> 1;
    build(l, mid, x << 1);
    build(mid + 1, r, x << 1 | 1);
    t[x] = t[x << 1] + t[x << 1 | 1];
}
void modify(int l, int r, int c, int x = 1) {
    if (l <= t[x].l && t[x].r <= r) {
        t[x].val = max(t[x].val, c);
        return;
    }
    int mid = t[x].l + t[x].r >> 1;
    if (l <= mid) modify(l, r, c, x << 1);
    if (r > mid) modify(l, r, c, x << 1 | 1);
    t[x] = t[x << 1] + t[x << 1 | 1];
}
node query(int l, int r, int x = 1) {
    if (l <= t[x].l && t[x].r <= r) return t[x];
    int mid = t[x].l + t[x].r >> 1;
    if (l > mid) return query(l, r, x << 1 | 1);
    else if (r <= mid) return query(l, r, x << 1);
    return query(l, mid, x << 1) + query(mid + 1, r, x << 1 | 1);
}
void solve() {
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
