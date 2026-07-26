#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 5050;
int a[N];
struct node {
    int id, val;
};
struct Cmp {
    bool operator()(const node& a, const node& b) const {
        return a.val > b.val; // 大根堆按 val 大的优先
    }
};
void solve() {
    int n, k, p, m;
    cin >> n >> k >> p >> m;
    priority_queue<node, vector<node>, Cmp> pq;
    queue<node> q;
    for (int i = 1; i <= n; i++) cin >> a[i];

    if (k == n) {
        printf("%d\n", m / a[p]);
        return;
    }
    int ans = 0;
    for (int i = 1; i <= n; i++) {
        if (i <= k) {
            if (i != p) pq.push({ i, a[i] });
            else {
                ans++;
                m -= a[p];
            }
        }
        else q.push({ i, a[i] });
    }
    if (ans) {
        node tmp = q.front();
        q.pop();
        pq.push(tmp);
        q.push({ p, a[p] });
    }
    while (m) {
        node mi = pq.top();
        pq.pop();
        if (mi.val <= m) {
            m -= mi.val;
            //pf(mi.id);
           // pf(m);
            q.push(mi);
            node com = q.front();
            q.pop();
            //pf(com.id);
            if (com.id == p) {
                if (com.val <= m) {
                    m -= com.val;
                    //pf(com.id);
                    //pf(m);
                    ans++;
                    q.push(com);
                    node com2 = q.front();
                    q.pop();
                    pq.push(com2);
                }
                else break;
            }
            else {
                pq.push(com);
            }
        }
        else break;
    }
    printf("%d\n", ans);

}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}