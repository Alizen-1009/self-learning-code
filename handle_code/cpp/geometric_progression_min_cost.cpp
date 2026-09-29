/*
题目：把数组调整为整数等比数列的最小代价

【题意】可以把每个整数 a_i 增减任意次数，每次改变 1 的代价为 1。重新排列并调整后，
希望数组变成 1,c,c^2,...,c^(n-1)，其中 c 是正整数；求最小总绝对差。

【方法】目标数列单调不降，所以先排序 a。枚举公比 c，并计算 sum|a_i-c^i|。当前
代价已经不小于最好答案时立即停止本轮。对于 n>=3，若 c^(n-1)-a[n-1] 已经不小于
当前答案，那么仅最后一项就不可能带来更优解；更大的 c 只会更差，因此结束枚举。
幂使用 __int128 并在超过界限时截断，避免溢出。

n=1 时目标只能是 [1]；n=2 时第二项单独选 c=max(1,a_2)，最小代价是
|a_1-1|+max(0,1-a_2)。这些分支修复了旧代码 n=1 不结束、
n=2 枚举范围极大的问题。

复杂度：排序 O(n log n)。若实际枚举 R 个公比，计算为 O(nR)，空间 O(n)。

English: Sort the array and enumerate the positive integer ratio c for the
target sequence 1,c,c^2,... . Stop once the last target term alone cannot beat
the current answer, using __int128 to avoid overflow. Handle n=1 and n=2
directly, fixing the former non-terminating implementation.
*/

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;
using int64 = long long;
using int128 = __int128_t;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<int64> values(n);
    for (int64& value : values) cin >> value;
    sort(values.begin(), values.end());

    if (n <= 2) {
        int64 answer = llabs(values[0] - 1);
        if (n == 2) answer += std::max<int64>(0, 1 - values[1]);
        cout << answer << '\n';
        return 0;
    }

    int64 answer = 0;
    for (int64 value : values) answer += llabs(value - 1);

    for (int64 ratio = 2;; ++ratio) {
        const int128 cutoff = static_cast<int128>(values.back()) + answer;
        int128 lastPower = 1;
        for (int exponent = 1; exponent < n && lastPower <= cutoff; ++exponent) {
            lastPower *= ratio;
        }
        if (lastPower - values.back() >= answer) break;

        int64 cost = 0;
        int128 target = 1;
        for (int i = 0; i < n; ++i) {
            const int128 difference = target >= values[i]
                                          ? target - values[i]
                                          : values[i] - target;
            if (difference >= answer - cost) {
                cost = answer;
                break;
            }
            cost += static_cast<int64>(difference);
            target *= ratio;
        }
        answer = min(answer, cost);
    }

    cout << answer << '\n';
    return 0;
}
