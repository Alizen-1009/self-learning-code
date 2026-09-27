/*
中文说明：计算在 n×m 网格中放置若干互不重叠 2×2 方块的方案数（允许一个也不放）。
解题方法：使用按行轮廓 DP。mask 表示当前行已被上一行方块占用的位置；DFS 从左到右
选择跳过当前格，或在当前行和下一行放置一个 2×2 方块，并生成下一行的 mask。
复杂度：约 O(n·3^m)，空间 O(2^m)，因此先令 m 为较小维度。

English: Count ways to place any number of non-overlapping 2x2 squares in an
n-by-m grid. A row-profile DP stores cells occupied from the previous row, and
a DFS enumerates skipping or placing each square. About O(n*3^m) time and
O(2^m) space; m is chosen as the smaller dimension.
*/
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
