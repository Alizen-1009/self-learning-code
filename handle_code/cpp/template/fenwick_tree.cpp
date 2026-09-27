/*
中文说明：树状数组（Fenwick Tree）模板，支持单点加法、前缀和与区间和查询。
解题方法：lowbit(x)=x&-x 表示节点管理区间的长度；更新时不断加 lowbit 向上走，
查询时不断减 lowbit 汇总前缀。示例从右向左插入 a[i]，统计比它小的已插入元素，
从而计算逆序关系数量。单次更新/查询 O(log N)，数组空间 O(N)。

English: Fenwick tree template supporting point addition, prefix sums, and range
sums. Updates move upward by lowbit; queries accumulate while moving downward.
The demo scans from right to left and counts inserted values smaller than a[i].
Each operation is O(log N), with O(N) storage.
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 2e5 + 7;
int t[N];
int lowbit(int x) {
    return x & -x;
}
void add(int x, int c) {
    while (x < N) {
        t[x] += c;
        x += lowbit(x);
    }
}
int get(int x) {
    int res = 0;
    while (x) {
        res += t[x];
        x -= lowbit(x);
    }
    return res;
}
int query(int l, int r) {
    return get(r) - get(l - 1);
}
void solve() {
    vector<int> a(10);
    for (int i = 0; i < 10; i++) a[i] = i ^ 10;
    int ans = 0;
    for (int i = 9; i >= 0; i--) {
        int k = a[i];
        int tmp = get(k - 1);
        add(k, 1);
        ans += tmp;
    }
    cout << ans << '\n';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
