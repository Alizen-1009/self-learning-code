/*
中文说明：这是一个“在两个数组间交换并保持 b 有序”的算法片段，不是完整可运行题解。
解题方法：遍历 a；若 a[i] 大于 b 的最小值 b[0]，就交换二者，再用线性移动把换入 b
的较大值插回正确位置。这样 b 始终有序，并把较小元素逐步换入 a。
复杂度：最坏 O(|a|·|b|)，额外空间 O(1)。注意：当前 main 没有读入，a、b 也未初始化，
所以程序直接运行不会产生有意义结果，需要补上具体题目的输入输出。

English: An incomplete fragment that swaps values between arrays while keeping
b sorted. If a[i] exceeds b[0], swap them and linearly reinsert the displaced
value into b. Worst-case O(|a|*|b|) time and O(1) extra space. The current main
has no input and the arrays are empty, so this is not yet a complete solution.
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;
vector<int> a, b;

void solve() {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] <= b[0]) continue;

        swap(a[i], b[0]);
        int val = b[0];

        int j = 1;
        while (j < b.size() && b[j] < val) {
            b[j - 1] = b[j];
            j++;
        }
        b[j - 1] = val;
    }

}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
