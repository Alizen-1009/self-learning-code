/*
 * @lc app=leetcode.cn id=45 lang=cpp
 *
 * [45] 跳跃游戏 II
 */

// @lc code=start
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        // vector<int> dp(n, 1e9);
        // dp[0] = 0;
        // for(int i = 1; i < n; i ++){
        //     for(int j = 0; j < n; j ++){
        //         if(nums[j] + j >= i) dp[i] = min(dp[j] + 1, dp[i]);
        //     }
        // }
        // return dp[n-1];

        int ans = 0;
        int ed = 0;
        int r = 0;
        for(int i = 0; i < n - 1; i ++){
            r = max(r, nums[i] + i);
            if(i == ed){
                ++ans;
                ed = r;
            }
        }
        return ans;
    }
};
// @lc code=end

