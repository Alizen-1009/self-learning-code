#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 1e5 + 7;
int c[N], p[N];
void solve() {
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> c[i] >> p[i];
    vector<double> dp(n + 2, 0.0);
    dp[n + 1] = 0;
    for (int i = n; i >= 1; i--) {
        dp[i] = max(dp[i + 1], c[i] + (1.0 - p[i] / 100.0) * dp[i + 1]);
    }
    printf("%.6f\n", dp[1]);
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}