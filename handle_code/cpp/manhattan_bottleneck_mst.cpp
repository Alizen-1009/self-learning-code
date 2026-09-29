/*
题目：曼哈顿距离完全图的最小瓶颈连接
【题意】输入 n 个二维整数坐标。任意两点都可相连，边权为两点曼哈顿距离除以 2
后向上取整。要把所有点连通，并使所用边中最大的权值尽可能小。
【方法】任一最小生成树都是最小瓶颈生成树。无需显式构造全部 O(n²) 条边：
Prim 的 dist[j] 保存点 j 到当前生成树的最小边权，每轮选 dist 最小的未加入点，
用它更新所有其他点，并记录已选边的最大值。时间 O(n²)，空间 O(n)。

English: On the complete graph of points, edge weight is ceil(Manhattan/2).
Compute the minimum possible maximum edge needed to connect all points. A dense
Prim algorithm maintains each point's cheapest connection; the answer is the
largest selected edge. O(n^2) time and O(n) space.
*/
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
