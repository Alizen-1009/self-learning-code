/*
 * @lc app=leetcode.cn id=34 lang=cpp
 *
 * [34] 在排序数组中查找元素的第一个和最后一个位置
 */

// @lc code=start
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        
        if(nums.size() == 0) return {-1,-1};
        int l = -1, r = nums.size();
        while(l + 1 < r){
            int mid = l + r  >> 1;
            if(nums[mid] >= target) r = mid;
            else l = mid; 
        }
        int l1, r1;
        r1 = r;

        l = -1, r = nums.size();
        while(l + 1 < r){
            int mid = l + r >> 1;
            if(nums[mid] <= target) l = mid;
            else r = mid;
        }
        if(r1 > l) return {-1, -1};
        return {r1, l};
    }
};
// @lc code=end

