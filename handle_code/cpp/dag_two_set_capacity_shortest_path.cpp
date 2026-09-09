#include <bits/stdc++.h>
using namespace std;

using ll = long long;

const ll INF = (ll)4e18;

struct Edge {
    int to;
    ll smallCost; // same set
    ll largeCost; // different sets
};

vector<int> topoSort(int n, const vector<vector<Edge>>& g) {
    vector<int> indeg(n + 1, 0);
    for (int u = 1; u <= n; u++) {
        for (const auto& e : g[u]) {
            indeg[e.to]++;
        }
    }

    queue<int> q;
    for (int i = 1; i <= n; i++) {
        if (indeg[i] == 0) q.push(i);
    }

    vector<int> topo;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        topo.push_back(u);

        for (const auto& e : g[u]) {
            if (--indeg[e.to] == 0) q.push(e.to);
        }
    }

    return topo;
}

ll solveDagTwoSetShortestPath(
    int n,
    int s,
    int t,
    int cap0,
    int cap1,
    const vector<int>& pointWeight,
    const vector<vector<Edge>>& g
) {
    vector<int> topo = topoSort(n, g);


    // dp[u][color][used0][used1] = best distance.
    // color is the set of vertex u.
    vector dp(
        n + 1,
        vector(2, vector(cap0 + 1, vector<ll>(cap1 + 1, INF)))
    );

    if (pointWeight[s] <= cap0) {
        dp[s][0][pointWeight[s]][0] = 0;
    }
    if (pointWeight[s] <= cap1) {
        dp[s][1][0][pointWeight[s]] = 0;
    }

    for (int u : topo) {
        for (int colorU = 0; colorU < 2; colorU++) {
            for (int used0 = 0; used0 <= cap0; used0++) {
                for (int used1 = 0; used1 <= cap1; used1++) {
                    ll cur = dp[u][colorU][used0][used1];
                    if (cur == INF) continue;

                    for (const auto& e : g[u]) {
                        for (int colorV = 0; colorV < 2; colorV++) {
                            int next0 = used0 + (colorV == 0 ? pointWeight[e.to] : 0);
                            int next1 = used1 + (colorV == 1 ? pointWeight[e.to] : 0);
                            if (next0 > cap0 || next1 > cap1) continue;

                            ll edgeCost = (colorU == colorV ? e.smallCost : e.largeCost);
                            ll& nxt = dp[e.to][colorV][next0][next1];
                            nxt = min(nxt, cur + edgeCost);
                        }
                    }
                }
            }
        }
    }

    ll ans = INF;
    for (int color = 0; color < 2; color++) {
        for (int used0 = 0; used0 <= cap0; used0++) {
            for (int used1 = 0; used1 <= cap1; used1++) {
                ans = min(ans, dp[t][color][used0][used1]);
            }
        }
    }

    return ans == INF ? -1 : ans;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    /*
        Input format:

        n m s t cap0 cap1
        pointWeight[1] pointWeight[2] ... pointWeight[n]
        u v smallCost largeCost   (m lines, directed edge u -> v)

        Vertices are 1-indexed.
    */
    int n, m, s, t, cap0, cap1;
    cin >> n >> m >> s >> t >> cap0 >> cap1;

    vector<int> pointWeight(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> pointWeight[i];
    }

    vector<vector<Edge>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        ll smallCost, largeCost;
        cin >> u >> v >> smallCost >> largeCost;
        g[u].push_back({ v, smallCost, largeCost });
    }

    cout << solveDagTwoSetShortestPath(n, s, t, cap0, cap1, pointWeight, g) << '\n';
    return 0;
}
