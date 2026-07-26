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