#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
const int N = 2e5 + 7;
int a[N];
struct node {
    int l, r;
    int val;
}t[N << 2];
node operator + (const node& A, const node& B) {
    node C;
    C.l = A.l, C.r = B.r;
    C.val = A.val + B.val;
    return C;
}
void build(int l, int r, int x = 1) {
    if (l == r) {
        t[x].l = l, t[x].r = r;
        t[x].val = a[l];
    }
    int mid = l + r >> 1;
    build(l, mid, x << 1);
    build(mid + 1, r, x << 1 | 1);
    t[x] = t[x << 1] + t[x << 1 | 1];
}
void modify(int l, int r, int c, int x = 1) {
    if (l <= t[x].l && t[x].r <= r) {
        t[x].val = max(t[x].val, c);
        return;
    }
    int mid = t[x].l + t[x].r >> 1;
    if (l <= mid) modify(l, r, c, x << 1);
    if (r > mid) modify(l, r, c, x << 1 | 1);
    t[x] = t[x << 1] + t[x << 1 | 1];
}
node query(int l, int r, int x = 1) {
    if (l <= t[x].l && t[x].r <= r) return t[x];
    int mid = t[x].l + t[x].r >> 1;
    if (l > mid) return query(l, r, x << 1 | 1);
    else if (r <= mid) return query(l, r, x << 1);
    return query(l, mid, x << 1) + query(mid + 1, r, x << 1 | 1);
}
void solve() {
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
