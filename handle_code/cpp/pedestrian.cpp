/*
中文说明：这是一个二元标记逆序对练习，不是“最多选多少个永不相遇的人”那道题。
输入 n 个 (位置,标记)；按位置升序后，统计标记为 1 的元素排在标记为 0 的元素
之前的点对数。相同位置按 pair 默认规则把标记 0 排在 1 前；若原题对同位置有特殊
判定，需要依据原题重新定义这个排序。
解题方法：从左向右维护已经出现的 1 的数量 cnt；每遇到一个 0，就把 cnt 加入答案。
答案最大为 O(n²)，必须用 long long。时间 O(n log n)，空间 O(n)。

English: After sorting points by the first coordinate, count pairs where a
label-1 point appears before a label-0 point. Ties use pair's default ordering.
Maintain the number of prior ones. O(n log n) time and O(n) space; use a 64-bit
answer. This is distinct from the maximum non-meeting people problem.
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

    long long ans = 0, cnt = 0;
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
