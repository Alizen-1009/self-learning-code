/*
 * @lc app=leetcode.cn id=4 lang=cpp
 *
 * [4] 寻找两个正序数组的中位数
 */

// @lc code=start
class Solution {
public:
    int GetKth(vector<int>& nums1, vector<int>& nums2, int k) {
        int cur1 = 0;
        int cur2 = 0;

        while (true) {
            // nums1 已经排除完
            if (cur1 == nums1.size()) {
                return nums2[cur2 + k - 1];
            }

            // nums2 已经排除完
            if (cur2 == nums2.size()) {
                return nums1[cur1 + k - 1];
            }

            // 只需要找剩余元素中的最小值
            if (k == 1) {
                return min(nums1[cur1], nums2[cur2]);
            }

            int half = k / 2;

            int newCur1 = min(
                cur1 + half,
                static_cast<int>(nums1.size())
            ) - 1;

            int newCur2 = min(
                cur2 + half,
                static_cast<int>(nums2.size())
            ) - 1;

            if (nums1[newCur1] <= nums2[newCur2]) {
                int removed = newCur1 - cur1 + 1;
                cur1 = newCur1 + 1;
                k -= removed;
            } else {
                int removed = newCur2 - cur2 + 1;
                cur2 = newCur2 + 1;
                k -= removed;
            }
        }
    }
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int total = nums1.size() + nums2.size();

        int leftK = (total + 1) / 2;
        int rightK = (total + 2) / 2;

        int leftValue = GetKth(nums1, nums2, leftK);
        int rightValue = GetKth(nums1, nums2, rightK);

        return (static_cast<double>(leftValue) + rightValue) / 2.0;
    
    }
};
// @lc code=end

