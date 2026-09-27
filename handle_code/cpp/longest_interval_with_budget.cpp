/*
中文说明：给定若干已覆盖的整数区间和预算 m，求一个最长连续区间，使其中未覆盖的
整数点数量不超过 m。
解题方法：先排序并合并相交或相邻区间，再对区间间空隙做前缀和。二分答案长度，
枚举每个合并区间作为起点，并二分终点落在哪个区间，从而计算所需填补的空隙。
复杂度：约 O(n log n log V)，空间 O(n)，V 为答案坐标范围。

English: Given covered integer intervals and a budget m, find the longest
continuous interval containing at most m uncovered points. Merge intervals,
prefix-sum the gaps, then binary-search the length and test every start using a
second binary search. About O(n log n log V) time and O(n) space.
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 1e5 + 7;
struct node {
    ll l, r;
    bool operator < (const node& rhs) const {
        if (rhs.l == l) return r < rhs.r;
        return l < rhs.l;
    }
}a[N];
ll n, m;
ll psum[N];
void merge_intervals() {
    int cnt = 0;
    for (int i = 1; i <= n; i++) {
        if (cnt == 0 || a[i].l > a[cnt].r + 1) {
            a[++cnt] = a[i];
        }
        else {
            a[cnt].r = max(a[cnt].r, a[i].r);
        }
    }
    n = cnt;
}
void init() {
    psum[1] = 0;
    for (int i = 2; i <= n; i++) {
        psum[i] = psum[i - 1] + a[i].l - a[i - 1].r - 1;
    }
}
ll calc_len(int lidx, int ridx) {
    if (lidx > ridx) return 0;
    return psum[ridx] - psum[lidx - 1];
}
ll find_len(ll st, ll ed, int id) {
    int lidx = id;
    int ridx = n + 1;
    while (lidx < ridx) {
        int mid = (lidx + ridx) >> 1;
        if (a[mid].r >= ed) ridx = mid;
        else lidx = mid + 1;
    }
    if (lidx == n + 1) {
        return calc_len(id, n) + ed - a[n].r;
    }
    if (ed < a[lidx].l) {
        return calc_len(id, lidx - 1) + ed - a[lidx - 1].r;
    }
    return calc_len(id, lidx);

}
bool check(ll mid) {
    for (int i = 1; i <= n; i++) {
        ll st = a[i].l;
        ll ed = st + mid - 1;
        ll res = find_len(st, ed, i);
        if (res <= m)  return true;
    }
    return false;
}
void solve() {

    cin >> n >> m;
    for (int i = 1; i <= n; i++) cin >> a[i].l >> a[i].r;
    sort(a + 1, a + 1 + n);
    merge_intervals();

    init();
    ll lans = m, rans = 1e10;
    while (lans < rans) {
        ll mid = (lans + rans + 1) >> 1;
        // pf(mid);
        if (check(mid)) lans = mid;
        else rans = mid - 1;
    }
    cout << lans << '\n';



}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
