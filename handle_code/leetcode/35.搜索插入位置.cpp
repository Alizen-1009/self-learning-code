/*
 * @lc app=leetcode.cn id=35 lang=cpp
 *
 * [35] 搜索插入位置
 */

// @lc code=start
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = nums.size();
        int l = 0, r = n-1;
        while(l < r){
            int mid = l + r + 1 >> 1;
            if(nums[mid] > target) r = mid - 1;
            else l = mid;
        }
        return l + (nums[l] < target);
    }
};
// @lc code=end

