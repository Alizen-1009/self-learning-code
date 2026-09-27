/*
中文说明：按第一维坐标排序后，统计所有“前面标记为 1、后面标记为 0”的点对，
可理解为两类行人在一维顺序中的相遇/逆序对数量。
解题方法：从左到右维护已出现的 1 的数量 cnt；遇到 0 时把 cnt 加入答案。
复杂度：排序 O(n log n)，扫描 O(n)，空间 O(n)。若 n 可达 1e5，答案可能超过 int，
当前 ans 最好改为 long long 后再用于大数据。

English: After sorting points by the first coordinate, count pairs where a
label-1 point appears before a label-0 point, interpretable as crossings between
two pedestrian directions. Maintain the number of prior ones. O(n log n) time
and O(n) space. The current int answer can overflow for large n.
*/
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
