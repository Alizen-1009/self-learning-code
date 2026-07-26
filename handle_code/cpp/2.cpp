#include <bits/stdc++.h>
#define ll long long
#define pf(x) cout << "(" << __LINE__ << ")" << #x << "=" << x << endl
using namespace std;
const int N = 202;
int g[N][N];
int sum[N][N];
int n;
void solve() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            scanf("%1d", &g[i][j]);
            sum[i][j] =
                sum[i - 1][j] + sum[i][j - 1] + g[i][j] - sum[i - 1][j - 1];
        }
    }

    int ans = 0;
    for (int len = 1; len <= n; len++) {
        int cnt = 0;
        for (int i = 1; i <= n && len % 2 == 0; i++) {
            for (int j = 1; j <= n; j++) {
                if (i + len - 1 > n || j + len - 1 > n)
                    continue;
                int res = sum[i + len - 1][j + len - 1] + sum[i - 1][j - 1] -
                          sum[i - 1][j + len - 1] - sum[i + len - 1][j - 1];
                if (res == len * len / 2)
                    cnt++;
            }
        }
        printf("%d\n", cnt);
    }
}
int main() {
    // ios::sync_with_stdio(false);
    // cin.tie(nullptr);
    int T = 1;
    // cin >> T;
    while (T--)
        solve();

    return 0;
}