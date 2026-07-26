#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 2e5 + 7;
int t[N];
int lowbit(int x) {
    return x & -x;
}
void add(int x, int c) {
    while (x < N) {
        t[x] += c;
        x += lowbit(x);
    }
}
int get(int x) {
    int res = 0;
    while (x) {
        res += t[x];
        x -= lowbit(x);
    }
    return res;
}
int query(int l, int r) {
    return get(r) - get(l - 1);
}
void solve() {
    vector<int> a(10);
    for (int i = 0; i < 10; i++) a[i] = i ^ 10;
    int ans = 0;
    for (int i = 9; i >= 0; i--) {
        int k = a[i];
        int tmp = get(k - 1);
        add(k, 1);
        ans += tmp;
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