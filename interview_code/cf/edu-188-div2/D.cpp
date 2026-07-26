#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;

void solve() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> color(n + 1, -1);
    ll ans = 0;
    for (int i = 1; i <= n; i++) {
        if (color[i] != -1) continue;
        queue<int> q;
        q.push(i);
        color[i] = 0;
        int cnt0 = 1, cnt1 = 0;
        bool ok = true;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v : g[u]) {
                if (color[v] == -1) {
                    color[v] = color[u] ^ 1;
                    if (color[v] == 0) cnt0++;
                    else cnt1++;
                    q.push(v);
                } else if (color[v] == color[u]) {
                    ok = false;
                }
            }
        }
        if (ok) ans += max(cnt0, cnt1);
    }
    cout << ans << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}
