/*
题目：最多修改 m 个字符后增加 M/T 的数量
【题意】输入长度 n、操作上限 m 和字符串 s。已有的 'M' 与 'T' 均算作目标
字符；一次操作可把其他一个位置改成 M 或 T。求最终目标字符的最大数量。
【方法】原本已有 cnt 个目标字符；每次操作最多增加 1，且最终数量不超过 n，
答案就是 min(n,cnt+m)。例如 s="MAT"、m=1，原有两个 M/T，可改掉 A，答案 3。
扫描一次即可，时间 O(n)，额外空间 O(1)。

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
