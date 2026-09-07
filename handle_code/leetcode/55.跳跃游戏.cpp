/*
 * @lc app=leetcode.cn id=55 lang=cpp
 *
 * [55] 跳跃游戏
 */

 // @lc code=start
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int ans = 0;
        int mx = -1;
        int nex = 0;
        for (int i = 0; i < nums.size() - 1; i++) {
            mx = max(mx, i + nums[i]);
            if (i == nex) {
                nex = mx;
                ans++;
            }
        }
        return nex >= (nums.size() - 1);
    }
};
// @lc code=end

