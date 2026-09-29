/*
题目：每个星号都必须变成括号时能否匹配
【题意】输入长度 n 和只含 '('、')'、'*' 的字符串。每个 '*' 必须恰好改成
'(' 或 ')'，不能删除。判断能否得到合法括号串，输出 1 或 0。
【方法】括号余额是“已读左括号数减右括号数”；合法前缀余额不能负，最终为 0。
扫描时维护可能余额的最小值 low 和最大值 high：'(' 都加 1，')' 都减 1，
'*' 可加 1 或减 1。若 high<0，所有选法都失败。每读一个字符余额奇偶性固定，
因此 low<0 时把它抬到 1（下一种非负可达余额），而不是抬到 0。总长度必须为
偶数，最后 low=0 才可行。例子 "(*)" 长度为奇数，因星号不可删除而必定失败。
时间 O(n)，空间 O(1)。

English: Decide whether every '*' can be replaced by '(' or ')' so the string
is balanced; '*' cannot be empty. Track the minimum and maximum reachable open
balance, preserving parity when the minimum drops below zero. A negative maximum
is impossible, and final minimum zero is feasible. O(n) time, O(1) space.
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
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
