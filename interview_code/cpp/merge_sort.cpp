#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;

void merge_sort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int mid = l + r >> 1;
    merge_sort(a, l, mid);
    merge_sort(a, mid + 1, r);
    vector<int> tmp(r - l + 1);
    int i = l, j = mid + 1;
    int cnt = 0;
    while (i <= mid || j <= r) {
        if (j == r + 1) {
            tmp[cnt++] = a[i++];
            continue;
        }
        if (i == mid + 1) {
            tmp[cnt++] = a[j++];
            continue;
        }
        if (a[i] <= a[j]) tmp[cnt++] = a[i++];
        else tmp[cnt++] = a[j++];
    }
    for (int i = l; i <= r; i++) a[i] = tmp[i - l];
}

int merge_sort_inv(vector<int>& a, int l, int r) {
    if (l >= r) return 0;
    int mid = l + r >> 1;
    int lans = merge_sort_inv(a, l, mid);
    int rans = merge_sort_inv(a, mid + 1, r);
    vector<int> tmp(r - l + 1);
    int i = l, j = mid + 1;
    int cnt = 0;
    int ans = lans + rans;
    while (i <= mid || j <= r) {
        if (j == r + 1) {
            tmp[cnt++] = a[i++];
            continue;
        }
        if (i == mid + 1) {
            tmp[cnt++] = a[j++];
            continue;
        }
        if (a[i] <= a[j]) tmp[cnt++] = a[i++];
        else {
            ans += mid - i + 1;
            tmp[cnt++] = a[j++];
        }
    }
    for (int i = l; i <= r; i++) a[i] = tmp[i - l];
    return ans;
}
void solve() {
    vector<int> a(10);
    for (int i = 0; i < 10; i++) a[i] = i ^ 10;
    cout << merge_sort_inv(a, 0, 9) << '\n';

    for (auto k : a) cout << k << ' ';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}