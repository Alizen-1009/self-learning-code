/*
题目：两个数组间的有序交换练习

【可从原代码确认的任务】给定数组 a 和已经升序排列的非空数组 b，依次处理 a[i]：
若 a[i]>b[0]，把 a[i] 与 b 的最小值交换，再把换入 b 的值重新插到正确位置。
因此处理过程中 b 始终升序，并且每一步都用 b 中最小的值替换较大的 a[i]。

原文件没有读入或输出，不能确定它来自哪道原题。为使算法可以运行和学习，这里约定
输入为 n m、n 个 a 元素、m 个 b 元素；输出处理后的 a 和 b，各占一行。若你找到
原始题目，需按题目重新核对输入输出及是否允许重排 b。

【方法】交换后，旧的 a[i] 暂时放在 b[0]。顺次左移 b 中小于它的值，把旧 a[i]
插入空出的槽。固定例子：a=[8,1]，b=[2,5,9]。处理 8 后 a=[2,1]，b=[5,8,9]；
处理 1 时不交换。最坏时间 O(nm)，除数组外额外空间 O(1)。

English: Given a and a nonempty sorted b, scan a. If a[i] exceeds b[0], swap
them and reinsert the displaced value into b to preserve sorted order. The
original file omitted all input/output, so this runnable study version uses
the documented n,m,array input and prints the two resulting arrays. O(nm) time.
*/

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<int> a(n), b(m);
    for (int& value : a) cin >> value;
    for (int& value : b) cin >> value;
    if (b.empty()) return 0;

    sort(b.begin(), b.end());
    for (int& value : a) {
        if (value <= b.front()) continue;
        swap(value, b.front());
        const int displaced = b.front();
        int position = 1;
        while (position < m && b[position] < displaced) {
            b[position - 1] = b[position];
            ++position;
        }
        b[position - 1] = displaced;
    }

    for (int i = 0; i < n; ++i) cout << a[i] << (i + 1 == n ? '\n' : ' ');
    for (int i = 0; i < m; ++i) cout << b[i] << (i + 1 == m ? '\n' : ' ');
    return 0;
}
