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