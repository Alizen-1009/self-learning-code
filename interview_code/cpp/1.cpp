#include <bits/stdc++.h>
#define ll long long
#define pf(x) cout << "(" << __LINE__ << ")" << #x << "=" << x << endl
using namespace std;
int n;
const int N = 1005;
const ll INF = 1e18;
ll x[N], y[N];
ll dist[N];
bool vis[N];
void solve() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> x[i] >> y[i];
    }
    for (int i = 1; i <= n; i++)
        dist[i] = INF;
    dist[1] = 0;
    ll ans = 0;
    for (int i = 1; i <= n; i++) {
        int v = -1;
        for (int j = 1; j <= n; j++) {
            if (!vis[j] && (v == -1 || dist[j] < dist[v])) {
                v = j;
            }
        }
        vis[v] = 1;
        ans = max(ans, dist[v]);
        // cout << ans << '\n';

        for (int j = 1; j <= n; j++) {
            // cout << dist[j] << '\n';
            ll res = (abs(x[j] - x[v]) + abs(y[j] - y[v]) + 1) / 2;
            // cout << res << '\n';
            dist[j] = min(dist[j], res);
        }
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