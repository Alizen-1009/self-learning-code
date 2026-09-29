/*
题目：用 m 个点填补空缺后的最长连续区间

【题意】给出 n 个已经覆盖的闭区间 [l_i,r_i]。允许额外覆盖至多 m 个原本未覆盖的
整数点，求最后能得到的最长连续覆盖区间的长度。相邻区间之间的空缺按整数点数计算：
例如 [1,3] 与 [6,7] 之间缺少 4、5，共 2 个点。

【方法】先把相交或相邻的区间合并，得到互不相邻的区间。若选取连续的一组区间，
必须填满组内的所有空隙；跨过某个区间却不把它选进来不会更优。用双指针维护一组
内部空隙总数不超过 m 的区间，另用 coverage 维护这些区间原本覆盖的点数。此时可
获得的最长长度就是 coverage+m：组内用一部分预算补空隙，剩余预算可向两端延长。
当加入新区间使空隙超预算时，向右移动左端点并移除相应覆盖长度和空隙。

正确性：任意最优连续区间所碰到的原区间必然是合并列表中的连续一段；把它们之间
空缺补齐后，未使用的预算都能在边界处延长。双指针枚举了每个右端点可行的最左
左端点，且覆盖数随窗口扩大只增不减，所以不会漏掉最优解。

复杂度：排序 O(n log n)，合并与双指针 O(n)，空间 O(n)。

English: Given covered closed integer intervals, fill up to m uncovered points
to maximize a continuous covered run. Merge overlapping/adjacent intervals and
use a sliding window whose internal gaps cost at most m. The achievable length
is covered points in the window plus m; unused budget extends its ends.
O(n log n) time and O(n) space.
*/

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
using int64 = long long;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    int64 budget;
    cin >> n >> budget;
    vector<pair<int64, int64>> intervals(n);
    for (auto& [left, right] : intervals) cin >> left >> right;
    sort(intervals.begin(), intervals.end());

    vector<pair<int64, int64>> merged;
    for (const auto& interval : intervals) {
        if (merged.empty() || interval.first > merged.back().second + 1) {
            merged.push_back(interval);
        } else {
            merged.back().second = max(merged.back().second, interval.second);
        }
    }

    int left = 0;
    int64 gaps = 0;
    int64 covered = 0;
    int64 answer = budget;
    for (int right = 0; right < static_cast<int>(merged.size()); ++right) {
        covered += merged[right].second - merged[right].first + 1;
        if (right > 0) {
            gaps += merged[right].first - merged[right - 1].second - 1;
        }
        while (gaps > budget) {
            covered -= merged[left].second - merged[left].first + 1;
            gaps -= merged[left + 1].first - merged[left].second - 1;
            ++left;
        }
        answer = max(answer, covered + budget);
    }

    cout << answer << '\n';
    return 0;
}
