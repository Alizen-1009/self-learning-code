#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
ll a, b, c, m;
ll gcdll(ll a, ll b) {
    if (b == 0) return a;
    return gcdll(b, a % b);
}
ll __lcm(ll a, ll b) {
    return a / gcdll(a, b) * b;
}
void solve() {
    cin >> a >> b >> c >> m;
    ll ab = __lcm(a, b);
    ll bc = __lcm(b, c);
    ll ac = __lcm(a, c);
    ll abc = __lcm(ab, bc);
    ll ansa = m / a * 6 - 3 * (m / ab + m / ac) + 2 * (m / abc);
    ll ansb = m / b * 6 - 3 * (m / ab + m / bc) + 2 * (m / abc);
    ll ansc = m / c * 6 - 3 * (m / ac + m / bc) + 2 * (m / abc);
    cout << ansa << ' ' << ansb << ' ' << ansc << '\n';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}
