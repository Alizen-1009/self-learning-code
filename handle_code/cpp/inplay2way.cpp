#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
vector<int> a, b;

void solve() {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] <= b[0]) continue;

        swap(a[i], b[0]);
        int val = b[0];

        int j = 1;
        while (j < b.size() && b[j] < val) {
            b[j - 1] = b[j];
            j++;
        }
        b[j - 1] = val;
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
