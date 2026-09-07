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