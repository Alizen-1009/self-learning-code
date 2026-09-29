/*
中文说明：给出 n 个非零整数，统计删除一个非空连续子数组后，剩余元素的乘积
至少有 k 个末尾 0 的方案数。若输入允许数值 0，则乘积及“末尾 0”定义必须单独
明确；当前分解循环不能处理 0。
【方法】每个数的末尾 0 来自一对 (2,5)。代码先剥离完整因子 10，把数量记入
cnt0；再分别计剩余的 2、5。因此某段乘积的末尾 0 数为 cnt0+min(cnt2,cnt5)。
用前缀和可 O(1) 算出删除 [i,r] 后两侧剩余的这三种计数。固定左端 i 后，r 越大，
删掉的因子只会更多，剩余末尾 0 数不会增加；所以二分最大的合格右端，再累加
合法右端的数量。时间 O(n log n+质因子分解次数)，空间 O(n)。

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
            int mid = l + (r - l + 1) / 2;
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
