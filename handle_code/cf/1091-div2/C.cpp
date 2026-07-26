#include<bits/stdc++.h>
#define ll long long
using namespace std;

ll gcdll(ll a, ll b) {
    while (b) {
        ll t = a % b;
        a = b;
        b = t;
    }
    return a;
}

void solve() {
    ll n, m, a, b;
    cin >> n >> m >> a >> b;

    if (gcdll(n, a) == 1 && gcdll(m, b) == 1 && gcdll(n, m) <= 2)
        cout << "YES\n";
    else
        cout << "NO\n";

}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}
