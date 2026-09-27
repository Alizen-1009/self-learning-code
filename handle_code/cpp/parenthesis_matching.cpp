/*
中文说明：判断字符串中的每个 '*' 能否分别替换为 '(' 或 ')'，使整个字符串成为
合法括号序列；'*' 不能被删除。
解题方法：扫描时维护当前可能括号余额的最小值 low 和最大值 high。普通括号同步
增减，'*' 让区间向两边扩展；high<0 表示任何方案都非法。由于每步余额奇偶性固定，
low<0 时修正为 1。最终 low==0 即存在合法替换。复杂度 O(n) 时间、O(1) 空间。

English: Decide whether every '*' can be replaced by '(' or ')' so the string
is balanced; '*' cannot be empty. Track the minimum and maximum reachable open
balance, preserving parity when the minimum drops below zero. A negative maximum
is impossible, and final minimum zero is feasible. O(n) time, O(1) space.
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 1010;
bool dp[N][N];
void solve() {
    int n;
    string s;
    cin >> n >> s;

    // 每个字符都必须变成一个括号，总长度必须为偶数
    if (n % 2 != 0) {
        cout << 0 << '\n';
        return;
    }

    int low = 0, high = 0;

    for (char c : s) {
        if (c == '(') {
            ++low;
            ++high;
        }
        else if (c == ')') {
            --low;
            --high;
        }
        else { // '*'
            --low;
            ++high;
        }

        if (high < 0) {
            cout << 0 << '\n';
            return;
        }

        if (low < 0) low = 1;
    }

    cout << (low == 0) << '\n';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
