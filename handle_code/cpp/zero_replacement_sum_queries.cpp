/*
中文说明：数组中的每个 0 可独立替换为查询区间 [l,r] 内的数；对每次查询输出
替换后数组总和可能达到的最小值和最大值。
解题方法：预处理非替换状态下的总和 sum 和 0 的数量 cnt。所有 0 都取 l 时最小，
都取 r 时最大，所以答案分别是 sum+l·cnt 与 sum+r·cnt。
复杂度：预处理 O(n)，每次查询 O(1)，空间 O(1)。

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
