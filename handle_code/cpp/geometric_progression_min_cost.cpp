/*
中文说明：把排序后的数组调整成整数等比数列 1,r,r^2,...，最小化逐项绝对差之和。
解题方法：排序后枚举整数公比 r，从 1 开始生成各项并累计代价；当当前代价不可能
优于答案或幂发生溢出时剪枝。该枚举依赖原题对 N/数值范围的约束，N 很小时需特别
限制 r；当前代码在 N=1 时不会自然结束，属于使用时必须注意的边界。
复杂度：若枚举到 R，时间 O(N log N+N·R)，空间 O(N)。

English: Fit the sorted array to 1,r,r^2,... for an integer ratio r, minimizing
the sum of absolute differences. Enumerate r and prune by the current answer and
overflow. The usable bound on r depends on the original constraints; notably,
the current loop needs an explicit bound when N=1. O(N log N+N*R) time, O(N) space.
*/
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INF = (1LL << 62);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    cin >> N;

    vector<ll> a(N);
    ll mx = 0;
    for (int i = 0; i < N; i++) {
        cin >> a[i];
        mx = max(mx, a[i]);
    }

    sort(a.begin(), a.end());

    ll ans = INF;

    // r=1 单独也会被包含
    for (ll r = 1;; r++) {

        ll cur = 0;
        ll val = 1;
        bool ok = true;

        for (int i = 0; i < N; i++) {
            cur += llabs(a[i] - val);

            if (cur >= ans) {
                ok = false;
                break;
            }

            if (i == N - 1) break;

            // 防止溢出，同时避免继续枚举无意义的大值
            if (val > (ll)2e18 / max(1LL, r)) {
                ok = false;
                break;
            }

            val *= r;
        }

        if (ok) ans = min(ans, cur);

        // 当 r^(N-1) 已经远超数据范围时即可停止
        __int128 t = 1;
        bool stop = false;
        for (int i = 1; i < N; i++) {
            t *= r;
            if (t > (__int128)2e18) {
                stop = true;
                break;
            }
        }
        if (stop) break;
    }

    cout << ans << "\n";
}
