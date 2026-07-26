#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
void solve() {
    int n;
    cin >> n;
    unordered_map<int, int> mp;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int x; cin >> x;
            mp[x]++;
        }
    }
    bool flg = true;
    for (auto [k, v] : mp) {
        if (v > n * (n - 1)) {
            flg = false;
            break;
        }
    }
    if (flg) puts("yes");
    else puts("no");
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}