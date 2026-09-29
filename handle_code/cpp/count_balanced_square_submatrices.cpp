/*
题目：按边长统计 0/1 数量相同的正方形
【题意】输入 n 和一个 n×n 的二进制矩阵。对边长 1 到 n，分别输出有多少个
正方形子矩阵的 0 与 1 数量相同；即使答案为 0 也要输出一行。
【方法】奇数边长的面积为奇数，必然无法平分。对每个偶数边长，枚举左上角，
用二维前缀和 O(1) 计算其中 1 的个数，检查是否等于面积的一半。例如 2×2
正方形有两个 1 才满足条件。时间 O(n³)，空间 O(n²)。

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
        if (len % 2 == 0) {
            for (int i = 1; i + len - 1 <= n; i++) {
                for (int j = 1; j + len - 1 <= n; j++) {
                    int res = sum[i + len - 1][j + len - 1] + sum[i - 1][j - 1] -
                              sum[i - 1][j + len - 1] - sum[i + len - 1][j - 1];
                    if (res == len * len / 2)
                        cnt++;
                }
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
