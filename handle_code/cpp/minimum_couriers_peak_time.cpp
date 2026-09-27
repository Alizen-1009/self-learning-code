/*
题目名称：最少配送员与满负荷时长

【中文题意】
配送站有 n 个必须准时执行的任务。第 i 个任务从时刻 s_i 开始，到时刻 e_i 结束，
执行期间必须由同一名配送员负责。一名配送员不能同时执行两个任务，但如果一个任务
恰好在时刻 x 结束、另一个任务恰好在 x 开始，可以由同一人无缝衔接。

需要输出：
1. 完成全部任务至少需要多少名配送员；
2. 在按这个最少人数安排后，所有配送员都同时忙碌的时间总长度。

输入格式：
第一行 n；接下来 n 行，每行两个整数 s_i、e_i，满足 s_i < e_i。
输出格式：两个整数，依次为最少配送员人数和所有人同时忙碌的累计时长。

【解题方法】
把任务看成左闭右开的时间段 [s_i,e_i)。某一时刻需要的配送员数等于覆盖该时刻的
任务数，所以最少配送员数就是最大重叠数 K。把每个开始时刻记为 +1、结束时刻记为
-1，排序并合并相同时刻的变化量。相邻事件时刻 [last,time) 内，正在执行的任务数
保持不变；若它等于全局最大值 K，就把该段长度计入答案。

相同时间的开始和结束统一合并，正好体现“结束后可以立即接下一单”的规则。

正确性要点：任意时刻有 c 个重叠任务时至少需要 c 人，因此至少需要最大重叠数 K；
区间图可以按结束时间复用人员，K 人也一定足够。故 K 就是最少人数，而覆盖数为 K
的时间段恰好是所有 K 名配送员同时工作的时间段。

复杂度：排序 O(n log n)，扫描 O(n)，空间 O(n)。

English: Each task occupies one courier on the half-open interval [s_i,e_i).
Output the minimum number of couriers and the total time during which all of
those couriers are simultaneously busy. Sweep +1 start and -1 end events after
merging equal timestamps. The maximum overlap is the minimum workforce, and the
length of segments having that overlap is the requested peak-load duration.
Complexity: O(n log n) time and O(n) space.
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
    cin >> n;

    vector<pair<int64, int>> events;
    events.reserve(2LL * n);
    for (int i = 0; i < n; ++i) {
        int64 start, finish;
        cin >> start >> finish;
        events.push_back({start, +1});
        events.push_back({finish, -1});
    }

    sort(events.begin(), events.end());

    int active = 0;
    int maximumActive = 0;
    int64 peakDuration = 0;
    int64 previousTime = events.front().first;

    for (int i = 0; i < static_cast<int>(events.size());) {
        const int64 currentTime = events[i].first;
        const int64 segmentLength = currentTime - previousTime;

        if (active > maximumActive) {
            maximumActive = active;
            peakDuration = segmentLength;
        } else if (active == maximumActive) {
            peakDuration += segmentLength;
        }

        int delta = 0;
        while (i < static_cast<int>(events.size()) &&
               events[i].first == currentTime) {
            delta += events[i].second;
            ++i;
        }
        active += delta;
        previousTime = currentTime;
    }

    cout << maximumActive << ' ' << peakDuration << '\n';
    return 0;
}
