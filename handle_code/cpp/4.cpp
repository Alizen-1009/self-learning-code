#include <bits/stdc++.h>
#define ll long long
#define pf(x) cout << "(" << __LINE__ << ")" << #x << "=" << x << endl
using namespace std;
void solve() {
    int n, m;
    string s;
    cin >> n >> m >> s;
    int cnt = 0;
    for (auto k : s) {
        if (k == 'M' || k == 'T')
            cnt++;
    }
    cout << min(n, cnt + m) << '\n';
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