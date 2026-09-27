/*
中文说明：处理无向图中的删边操作和两点连通性询问。
解题方法：并查集不擅长在线删除，因此先记录所有会被删除的边，用从未删除的边建立
最终图；随后倒序处理询问，把“删边”反转成“加边”，连通性查询直接比较两个根。
代码还对原始点编号做了离散化。复杂度约 O((m+q)log(m+q)+(m+q)α(n))，空间 O(m+q)。
当前 set 写法假设边无重数、同一条边的删除语义不重复。

English: Answer edge-deletion and connectivity queries in an undirected graph.
Build the final graph without deleted edges, then process operations backward so
each deletion becomes a DSU union. Vertex IDs are compressed. Complexity is
roughly O((m+q)log(m+q)+(m+q)alpha(n)); the set-based code assumes simple edges.
*/
#include <bits/stdc++.h>
#define ll long long
#define pf(x) cout << "(" << __LINE__ << ")" << #x << "=" << x << endl
using namespace std;
const int N = 2e5 + 7;
int n, m, q;
set<pair<int, int>> edge, rmv;
unordered_map<int, int> mp;
struct query {
    int op, u, v;
}a[N];
int p[N];
vector<string> ans;
int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}
void merge(int a, int b) {
    int pa = find(a);
    int pb = find(b);
    if (pa != pb) p[pa] = pb;
}
void solve() {
    cin >> n >> m >> q;
    int cnt = 0;
    while (m--) {
        int a, b;
        cin >> a >> b;
        if (!mp[a]) mp[a] = ++cnt;
        if (!mp[b]) mp[b] = ++cnt;
        a = mp[a], b = mp[b];
        if (a > b) swap(a, b);
        edge.insert({ a, b });
    }
    for (int i = 1; i <= q; i++) {
        int op, u, v;
        cin >> op >> u >> v;
        if (!mp[u]) mp[u] = ++cnt;
        if (!mp[v]) mp[v] = ++cnt;
        u = mp[u], v = mp[v];
        if (u > v) swap(u, v);
        if (op == 1) rmv.insert({ u, v });
        a[i] = { op, u, v };
    }
    for (int i = 1; i < N; i++) p[i] = i;

    for (auto pr : edge) {
        if (rmv.count(pr)) continue;
        merge(pr.first, pr.second);
    }
    for (int i = q; i >= 1; i--) {
        auto [op, u, v] = a[i];
        if (op == 1) {
            if (edge.count({ u, v })) merge(u, v);
        }
        else {
            if (find(u) == find(v)) ans.push_back("Yes");
            else ans.push_back("No");
        }
    }
    reverse(ans.begin(), ans.end());
    for (auto k : ans) cout << k << '\n';

}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    // cin >> T;
    while (T--) solve();

    return 0;
}
