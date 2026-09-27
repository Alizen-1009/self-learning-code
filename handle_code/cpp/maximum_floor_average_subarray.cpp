/*
题目名称：至少 L 天的最大下取整平均产量

【中文题意】
采摘季共有 n 天，第 i 天的净产量为整数 a_i，允许为负数。选择一段连续日期 [l,r]，
要求长度 r-l+1 至少为 L。该区间的报表平均产量定义为

    floor((a_l+a_{l+1}+...+a_r)/(r-l+1))，

其中 floor 是数学意义的向下取整，负数也向负无穷取整。需要求所有合法区间中这个
整数平均值的最大值。

输入格式：第一行 T；每组先输入 n、L，再输入 n 个整数 a_i。
输出格式：每组输出一行最大可能的下取整平均值。

【解题方法：二分答案 + 前缀和】
二分一个整数答案 x。存在平均值至少为 x 的区间，等价于存在长度至少 L 的区间满足

    sum(a_i-x) >= 0。

令前缀和 prefix[i]=sum_{1..i}(a_i-x)。对于每个右端点 r>=L，只要

    prefix[r] - min(prefix[0..r-L]) >= 0，

就说明某个长度至少 L 的区间平均值不小于 x。该判定为 O(n)，并且关于 x 单调，
所以可在 [-10^9,10^9] 上二分最大的可行整数。直接判定“平均值 >= x”避免了 C++
负数整数除法向 0 截断与题目 floor 定义不同的问题。

正确性要点：对固定区间，floor(avg)>=x 当且仅当 avg>=x；移项后就是变换数组的
区间和非负。维护允许左端点之前的最小前缀和，枚举了所有长度至少 L 的区间。

复杂度：每组 O(n log 2·10^9)，空间 O(n)；所有测试的 n 之和不超过 2×10^5。

English: Find the maximum floored average among subarrays of length at least L.
Binary-search an integer x. Such a subarray has average at least x iff its sum
after replacing every a_i by a_i-x is nonnegative. Prefix sums plus the minimum
prefix up to r-L test this in O(n). Total complexity is O(n log value_range)
time and O(n) space. This avoids incorrect truncation behavior for negatives.
*/

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;
using int64 = long long;

bool canReachAverage(const vector<int64>& values, int minimumLength, int64 target) {
    const int n = static_cast<int>(values.size());
    vector<int64> prefix(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        prefix[i] = prefix[i - 1] + values[i - 1] - target;
    }

    int64 minimumPrefix = 0;
    for (int right = minimumLength; right <= n; ++right) {
        minimumPrefix = min(minimumPrefix, prefix[right - minimumLength]);
        if (prefix[right] >= minimumPrefix) return true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int testCases;
    cin >> testCases;
    while (testCases--) {
        int n, minimumLength;
        cin >> n >> minimumLength;

        vector<int64> values(n);
        for (int64& value : values) cin >> value;

        int64 low = *min_element(values.begin(), values.end());
        int64 high = *max_element(values.begin(), values.end());
        while (low < high) {
            const int64 middle = low + (high - low + 1) / 2;
            if (canReachAverage(values, minimumLength, middle)) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }

        cout << low << '\n';
    }
    return 0;
}
