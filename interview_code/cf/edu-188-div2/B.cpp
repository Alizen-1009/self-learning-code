#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
#define PII pair<int,int>
using namespace std;
const int N = 2e5 + 7;
int a[N];
int n;

void solve() {
    cin >> n;
    priority_queue<PII> pq;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        pq.push({ a[i], i });
    }
    int mxid = N;
    int ans = 0;
    while (mxid != 1) {
        // pf(mxid);
        auto tmp = pq.top();
        pq.pop();
        if (tmp.second < mxid) {
            ans++;
            mxid = tmp.second;
        }
    }
    cout << ans << '\n';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}