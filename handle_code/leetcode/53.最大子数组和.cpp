/*
 * @lc app=leetcode.cn id=53 lang=cpp
 *
 * [53] 最大子数组和
 */

// @lc code=start
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        vector<int> dp(nums.size(), -1e9);
        dp[0] = nums[0];
        int ans = dp[0];
        for(int i = 1; i < nums.size(); i ++){
            dp[i] = max(dp[i-1]+nums[i], nums[i]);
            ans = max(dp[i], ans);
        }
        return ans;
    }
};
// @lc code=end

