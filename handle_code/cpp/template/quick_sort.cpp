#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;

void quick_sort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int target = a[l + r >> 1];
    int i = l, j = r;
    while (i <= j) {
        while (i < j && a[j] > target) j--;
        while (i < j && a[i] < target) i++;
        if (i <= j) {
            swap(a[i], a[j]);
            i++, j--;
        }
    }
    quick_sort(a, l, j);
    quick_sort(a, i, r);
}
void solve() {
    vector<int> a(10);
    for (int i = 0; i < 10; i++) a[i] = ((int)rand());
    quick_sort(a, 0, 9);
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