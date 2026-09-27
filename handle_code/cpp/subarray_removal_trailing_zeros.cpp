/*
中文说明：统计删除一个连续子数组后，剩余所有数的乘积末尾至少有 k 个 0 的删除方案。
解题方法：把每个数中的完整因子 10、剩余因子 2 和因子 5 分别做前缀计数；删除
[i,mid] 后，剩余乘积的零数为 cnt10+min(cnt2,cnt5)。对每个左端点二分最大的合法
右端点并累加方案数。复杂度 O(n log n+总质因子分解次数)，空间 O(n)。

English: Count contiguous subarrays whose removal leaves a product with at least
k trailing zeros. Prefix-count factors 10, residual 2, and residual 5; for each
left endpoint, binary-search the farthest removable right endpoint satisfying
cnt10+min(cnt2,cnt5)>=k. O(n log n) plus factor extraction, O(n) space.
*/
#include <bits/stdc++.h>
#define ll long long
#define pf(x) cout << "(" << __LINE__ << ")" << #x << "=" << x << endl
using namespace std;
const int N = 1e5 + 7;
int cnt2[N];
int cnt5[N];
int cnt0[N];
void solve() {
    int n, k;
    cin >> n >> k;
    for (int i = 1; i <= n; i++) {
        int x;
        cin >> x;
        cnt2[i] = cnt2[i - 1];
        cnt0[i] = cnt0[i - 1];
        cnt5[i] = cnt5[i - 1];
        while (x % 10 == 0) {
            cnt0[i]++;
            x /= 10;
        }
        while (x % 2 == 0) {
            cnt2[i]++;
            x /= 2;
        }
        while (x % 5 == 0) {
            cnt5[i]++;
            x /= 5;
        }
    }
    ll ans = 0;
    for (int i = 1; i <= n; i++) {
        int l = i - 1, r = n;
        while (l < r) {
            int mid = l + r + 1 >> 1;
            int st = i, ed = mid;
            int a = cnt2[i - 1] + cnt2[n] - cnt2[mid];
            int b = cnt5[i - 1] + cnt5[n] - cnt5[mid];
            int c = cnt0[i - 1] + cnt0[n] - cnt0[mid];
            int res = c + min(a, b);
            // cout << "mid = " << mid << " res = " << res << '\n';
            if (res >= k)
                l = mid;
            else
                r = mid - 1;
        }
        // cout << l << '\n';
        ans += 1ll * (l - i + 1);
    }
    cout << ans << '\n';
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
