/*
 * @lc app=leetcode.cn id=33 lang=cpp
 *
 * [33] 搜索旋转排序数组
 */

// @lc code=start
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while(l < r - 1){
            int mid = l + r >> 1;
            if(nums[mid] == target) return mid;
            if(nums[mid] > nums[l]){
                if(target >= nums[l] && nums[mid] >= target) r = mid;
                else l = mid + 1;
            }
            else{
                if(target >= nums[mid] && nums[r] >= target) l = mid;
                else r = mid - 1;
            }
        }
        if(nums[l] == target) return l;
        if(nums[r] == target) return r;
        return -1;
    }
};
// @lc code=end

