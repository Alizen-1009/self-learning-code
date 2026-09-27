/*
中文说明：第 i 个人从初始位置 p_i 出发，以速度 1 直线走向目标位置 q_i，到达后
停止。选择尽可能多的人，使任意两人在包括 t=0 在内的任何时刻都不会处于同一位置。

核心结论：对于初始位置 p_i < p_j 的两个人，他们永不相遇，当且仅当 q_i < q_j。
如果终点次序相反或相同，两条连续轨迹的左右顺序最终会交换或重合，因此必然相遇；
如果起点和终点的严格次序一致，两人在运动过程中也始终保持该次序。

解题方法：
1. 按 p 从小到大排序；p 相同时按 q 从大到小排序。
2. 在排序后的 q 序列上求严格递增最长子序列（LIS）。
3. 同起点的人在 t=0 已经相遇；将同 p 的 q 降序排列，可保证严格 LIS 最多选一个。
   严格 LIS 也会自动排除终点相同的人。

复杂度：排序和 LIS 均为 O(n log n)，空间 O(n)。

English: Person i moves from p_i toward q_i at unit speed and stops on arrival.
Choose the largest subset in which no two people ever occupy the same position,
including at t=0.

For p_i < p_j, the two trajectories never meet iff q_i < q_j. Therefore sort
by p ascending and, for equal p, by q descending, then compute a strictly
increasing LIS of q. The descending tie order prevents selecting two people with
the same start, while a strict LIS prevents equal destinations.

Complexity: O(n log n) time and O(n) space.
*/

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<pair<long long, long long>> people(n);
    for (auto& [start, target] : people) {
        cin >> start >> target;
    }

    sort(people.begin(), people.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) return lhs.first < rhs.first;
        return lhs.second > rhs.second;
    });

    // tails[len - 1]：长度为 len 的严格递增子序列能取得的最小末尾值。
    vector<long long> tails;
    tails.reserve(n);

    for (const auto& [start, target] : people) {
        (void)start;
        auto it = lower_bound(tails.begin(), tails.end(), target);
        if (it == tails.end()) {
            tails.push_back(target);
        } else {
            *it = target;
        }
    }

    cout << tails.size() << '\n';
    return 0;
}
