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