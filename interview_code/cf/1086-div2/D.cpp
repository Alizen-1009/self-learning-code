#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n;
    cin >> n;
    vector<string> s(n);
    for (int i = 0; i < n; i++) cin >> s[i];

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if ((int)s[i].size() != n) ok = false;
        if (s[i][i] != '1') ok = false;
    }

    vector<int> anc_cnt(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (s[i][j] == '1') {
                anc_cnt[j]++;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (s[i][j] == '1' && s[j][i] == '1') ok = false;
        }
    }

    vector<int> topo(n);
    iota(topo.begin(), topo.end(), 0);
    sort(topo.begin(), topo.end(), [&](int a, int b) {
        if (anc_cnt[a] != anc_cnt[b]) return anc_cnt[a] < anc_cnt[b];
        return a < b;
    });

    int W = (n + 63) >> 6;
    vector<vector<unsigned long long>> out_bits(n, vector<unsigned long long>(W, 0ULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (s[i][j] == '1') out_bits[i][j >> 6] |= 1ULL << (j & 63);
        }
    }

    vector<pair<int, int>> edges;
    if (ok) {
        vector<unsigned long long> covered(W);
        for (int u = 0; u < n && ok; u++) {
            fill(covered.begin(), covered.end(), 0ULL);
            for (int v : topo) {
                if (v == u || s[u][v] == '0') continue;
                if ((covered[v >> 6] >> (v & 63)) & 1ULL) continue;

                edges.push_back({u, v});
                for (int w = 0; w < W; w++) covered[w] |= out_bits[v][w];
            }
        }
    }

    if ((int)edges.size() != n - 1) ok = false;

    vector<vector<int>> g(n), und(n);
    if (ok) {
        for (auto [u, v] : edges) {
            g[u].push_back(v);
            und[u].push_back(v);
            und[v].push_back(u);
        }

        vector<int> vis(n, 0);
        queue<int> q;
        q.push(0);
        vis[0] = 1;
        int reached = 1;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v : und[u]) {
                if (!vis[v]) {
                    vis[v] = 1;
                    reached++;
                    q.push(v);
                }
            }
        }
        if (reached != n) ok = false;
    }

    if (ok) {
        vector<int> indeg(n, 0);
        for (int u = 0; u < n; u++) {
            for (int v : g[u]) indeg[v]++;
        }

        queue<int> q;
        for (int i = 0; i < n; i++) {
            if (indeg[i] == 0) q.push(i);
        }

        vector<int> topo2;
        topo2.reserve(n);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            topo2.push_back(u);
            for (int v : g[u]) {
                indeg[v]--;
                if (indeg[v] == 0) q.push(v);
            }
        }
        if ((int)topo2.size() != n) ok = false;

        if (ok) {
            vector<vector<unsigned long long>> reach(n, vector<unsigned long long>(W, 0ULL));
            for (int idx = n - 1; idx >= 0; idx--) {
                int u = topo2[idx];
                reach[u][u >> 6] |= 1ULL << (u & 63);
                for (int v : g[u]) {
                    for (int w = 0; w < W; w++) {
                        reach[u][w] |= reach[v][w];
                    }
                }
            }

            for (int i = 0; i < n && ok; i++) {
                for (int w = 0; w < W; w++) {
                    if (reach[i][w] != out_bits[i][w]) {
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    if (!ok) {
        cout << "No\n";
        return;
    }

    cout << "Yes\n";
    for (auto [u, v] : edges) {
        cout << u + 1 << ' ' << v + 1 << '\n';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
