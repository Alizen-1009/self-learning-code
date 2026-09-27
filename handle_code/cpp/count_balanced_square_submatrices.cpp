/*
中文说明：统计二进制方阵中“0 和 1 数量相等”的正方形子矩阵，并按边长输出数量。
解题方法：先建立二维前缀和；奇数边长不可能平衡，偶数边长枚举左上角，
用 O(1) 的矩形和判断其中 1 的数量是否等于 len * len / 2。
复杂度：时间 O(n^3)，空间 O(n^2)。

English: Count square submatrices containing the same number of zeros and ones,
and print the count for every side length. Build a 2D prefix sum; skip odd side
lengths and test every even square in O(1). Complexity: O(n^3) time, O(n^2) space.
*/
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
