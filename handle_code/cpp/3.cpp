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