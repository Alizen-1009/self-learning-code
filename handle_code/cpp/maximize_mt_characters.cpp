/*
中文说明：字符串长度为 n，已有字符 M 或 T 的位置算作满足；最多修改 m 个其他字符，
求最终最多能有多少个 M/T 字符。
解题方法：线性统计原有 M/T 数量 cnt，再补上至多 m 个，答案为 min(n,cnt+m)。
复杂度：时间 O(n)，空间 O(1)。

English: Maximize the number of characters equal to M or T after changing at
most m other positions. Count existing M/T characters and return min(n,cnt+m).
O(n) time and O(1) extra space.
*/
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
