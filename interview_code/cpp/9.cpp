#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int MOD = 998244353;

void dfs(int col, int m, int curMask, int nextMask, bool canPutDown,
    int ways, vector<int>& ndp) {
    while (col < m && (curMask >> col & 1)) col++;
    if (col == m) {
        ndp[nextMask] += ways;
        if (ndp[nextMask] >= MOD) ndp[nextMask] -= MOD;
        return;
    }

    dfs(col + 1, m, curMask | (1 << col), nextMask, canPutDown, ways, ndp);

    if (canPutDown && col + 1 < m
        && !(curMask >> (col + 1) & 1)) {
        int bits = (1 << col) | (1 << (col + 1));
        dfs(col + 2, m, curMask | bits, nextMask | bits, canPutDown, ways, ndp);
    }
}

void solve() {
    int n, m;
    while (cin >> n >> m) {
        if (m > n) swap(n, m);

        int states = 1 << m;
        vector<int> dp(states), ndp(states);
        dp[0] = 1;

        for (int row = 0; row < n; row++) {
            fill(ndp.begin(), ndp.end(), 0);
            bool canPutDown = (row + 1 < n);

            for (int mask = 0; mask < states; mask++) {
                if (!dp[mask]) continue;
                dfs(0, m, mask, 0, canPutDown, dp[mask], ndp);
            }

            dp.swap(ndp);
        }

        cout << dp[0] << '\n';
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
