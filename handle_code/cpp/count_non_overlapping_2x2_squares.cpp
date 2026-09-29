/*
题目：互不重叠的 2×2 方块摆放方案数
【题意】输入网格高 n、宽 m，统计放置任意数量 2×2 方块的方案，要求所有方块
都在网格内且两两不共享格子。什么都不放也是一种方案；答案对 998244353 取模。
【方法】令 m 为较小维度。按行推进时，mask 标记当前行哪些格子已被上一行的方块
占据。DFS 从左到右处理未占用格：可以空着，或在当前行及下一行放一个 2×2 方块。
新放方块占据下一行的两格，记入 nextMask。整行处理完，累计到下一行的 DP。
例如 2×2 网格只有“不放”和“放一块”两种方案。轮廓状态 O(2^m)，每行用 DFS
枚举合法动作；适用于较小的 m，空间 O(2^m)。

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
