/*
题目：把所有 0 替换为区间值后的最小/最大数组和
【题意】数组中非 0 元素保持不变。每次查询给出允许替换的值域 [l,r]；数组里的
每个 0 可以独立选取其中任意值。输出这次查询能得到的最小总和和最大总和。
【方法】只需预处理原数组和 sum，以及 0 的个数 cnt。为了使总和最小，所有 0
都取 l；为了最大，所有 0 都取 r。因此答案分别是 sum+l*cnt、sum+r*cnt。
例如 [2,0,0] 与 [3,5] 的输出为 8、12。预处理 O(n)，单次查询 O(1)，空间 O(1)。

English: Each zero may be replaced by a value in query range [l,r]. Output the
minimum and maximum possible array sum. Precompute the original sum and zero
count; answers are sum+l*cnt and sum+r*cnt. O(n) preprocessing, O(1) per query.
*/
#include <bits/stdc++.h>
#define ll long long
#define pf(x) cout << "(" << __LINE__ << ")" << #x << "=" << x << endl
using namespace std;
int n, q;
void solve() {
    ll cnt = 0;
    cin >> n >> q;
    ll sum = 0;
    for (int i = 1; i <= n; i++) {
        ll x;
        cin >> x;
        sum += x;
        if (!x)
            cnt++;
    }
    while (q--) {
        ll l, r;
        cin >> l >> r;
        cout << sum + l * cnt << " " << sum + r * cnt << '\n';
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    // cin >> T;
    while (T--)
        solve();

    return 0;
}
