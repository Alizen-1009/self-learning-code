/*
题目名称：严格递增水晶能量的最小调整次数

【中文题意】
有 n 颗从左到右排列的水晶，初始能量为 a_1...a_n。一次操作可以任选一颗水晶，
让它的能量增加 1 或减少 1。需要把最终能量调整为严格递增的整数序列
b_1<b_2<...<b_n，并最小化总操作次数 sum|a_i-b_i|。

输入格式：第一行 n（n<=2000），第二行 n 个整数 a_i。
输出格式：最少操作次数。

【关键变换】
严格递增整数满足 b_i >= b_{i-1}+1。令

    c_i = b_i-i，x_i = a_i-i，

则约束变成 c_1<=c_2<=...<=c_n，代价保持为

    |a_i-b_i| = |(a_i-i)-(b_i-i)| = |x_i-c_i|。

问题因此变成 L1 意义下的非降序回归。最优 c_i 可以取自某个 x_j，所以把所有 x_i
排序去重得到候选值 v。动态规划 dp[i][j] 表示处理前 i 项且 c_i=v_j 的最小代价：

    dp[i][j] = |x_i-v_j| + min(dp[i-1][0..j])。

扫描 j 时维护上一行的前缀最小值，即可把转移从 O(n) 降为 O(1)。

正确性要点：变换在严格递增整数序列与非降序整数序列之间一一对应；L1 最优解在
每个常量块上可取该块数据的中位数，因此一定存在只使用 x_i 候选值的最优解；DP
枚举了全部非降候选序列。

复杂度：候选值最多 n 个，时间 O(n^2)，滚动数组空间 O(n)。

English: Change integer energies into a strictly increasing sequence with
minimum L1 cost. Transform b_i into c_i=b_i-i and a_i into x_i=a_i-i; then c
only needs to be nondecreasing and the cost is |x_i-c_i|. An L1 isotonic optimum
uses values drawn from x. DP over sorted unique candidates uses prefix minima:
dp[i][j]=|x_i-v_j|+min(dp[i-1][0..j]). O(n^2) time and O(n) space.
*/

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;
using int64 = long long;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<int64> transformed(n);
    for (int i = 0; i < n; ++i) {
        int64 energy;
        cin >> energy;
        transformed[i] = energy - (i + 1LL);
    }

    vector<int64> candidates = transformed;
    sort(candidates.begin(), candidates.end());
    candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

    const int candidateCount = static_cast<int>(candidates.size());
    vector<int64> previous(candidateCount);
    vector<int64> current(candidateCount);

    for (int j = 0; j < candidateCount; ++j) {
        previous[j] = llabs(transformed[0] - candidates[j]);
    }

    for (int i = 1; i < n; ++i) {
        int64 prefixMinimum = numeric_limits<int64>::max();
        for (int j = 0; j < candidateCount; ++j) {
            prefixMinimum = min(prefixMinimum, previous[j]);
            current[j] = prefixMinimum + llabs(transformed[i] - candidates[j]);
        }
        previous.swap(current);
    }

    cout << *min_element(previous.begin(), previous.end()) << '\n';
    return 0;
}
