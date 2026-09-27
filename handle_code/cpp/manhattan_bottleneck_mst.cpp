/*
中文说明：把平面点构成完全图，两点边权为 ceil(曼哈顿距离/2)，求连接所有点时
必须使用的最小最大边权，即最小瓶颈生成树的瓶颈值。
解题方法：在隐式完全图上运行朴素 Prim。dist[j] 保存 j 到当前生成树的最小边权，
每次加入 dist 最小的点，并用它更新所有其他点；答案是所有入树边权的最大值。
复杂度：时间 O(n^2)，空间 O(n)，无需显式保存 O(n^2) 条边。

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
