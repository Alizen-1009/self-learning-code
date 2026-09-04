/*
 * @lc app=leetcode.cn id=41 lang=cpp
 *
 * [41] 缺失的第一个正数
 */

// @lc code=start
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        int cur = 0;
        for(int i = 0; i < n; i ++){
            while(nums[i] >= 1 && nums[i] <= n){
                int target_id = nums[i] - 1;
                if(nums[i] == nums[target_id]) break;
                swap(nums[i], nums[target_id]);
            }
        }

        for(int i = 0; i < n; i ++){
            if(nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }
};
// @lc code=end

