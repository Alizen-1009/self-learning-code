#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
void solve() {
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            // cout << j << ' ' << (i ^ j) << " ";
            cout << (i ^ j) << " ";
        }
        cout << '\n';
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