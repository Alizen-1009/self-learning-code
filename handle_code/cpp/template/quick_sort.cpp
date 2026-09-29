/*
中文说明：原地快速排序模板，使用区间中点元素作为枢轴并进行双指针分区。
解题方法：i 从左找不小于枢轴的元素，j 从右找不大于枢轴的元素，交换后继续，
直到指针交错，再递归处理 [l,j] 和 [i,r]。
复杂度：平均 O(n log n)，最坏 O(n^2)；递归栈平均 O(log n)、最坏 O(n)。

English: In-place quicksort using the middle element as pivot and a two-pointer
partition. Swap misplaced values until pointers cross, then recurse on [l,j]
and [i,r]. Average O(n log n), worst O(n^2); stack depth is average O(log n),
worst O(n).
*/
#include<bits/stdc++.h>
#define ll long long
#define pf(x) cout<<"("<<__LINE__<<")"<<#x<<"="<<x<<endl
using namespace std;

void quick_sort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int target = a[l + (r - l) / 2];
    int i = l, j = r;
    while (i <= j) {
        while (i < j && a[j] > target) j--;
        while (i < j && a[i] < target) i++;
        if (i <= j) {
            swap(a[i], a[j]);
            i++, j--;
        }
    }
    quick_sort(a, l, j);
    quick_sort(a, i, r);
}
void solve() {
    vector<int> a(10);
    for (int i = 0; i < 10; i++) a[i] = ((int)rand());
    quick_sort(a, 0, 9);
    for (auto k : a) cout << k << ' ';
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T = 1;
    //cin >> T;
    while (T--) solve();

    return 0;
}
