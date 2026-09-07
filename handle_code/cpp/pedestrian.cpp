#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 1e5 + 7;
pair<int, int> p[N];
void solve() {
    int n;
    cin >> n;

    for (int i = 1; i <= n; i++) {
        cin >> p[i].first >> p[i].second;
    }

    sort(p + 1, p + n + 1);

    int ans = 0, cnt = 0;
    for (int i = 1; i <= n; i++) {
        if (p[i].second) cnt++;
        else ans += cnt;
    }
    cout << ans << '\n';

}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}