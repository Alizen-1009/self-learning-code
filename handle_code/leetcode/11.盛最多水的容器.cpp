/*
 * @lc app=leetcode.cn id=11 lang=cpp
 *
 * [11] 盛最多水的容器
 */

// @lc code=start
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int ans = 0;
        ans = min(height[l], height[r]) * (r - l);
        while(l < r){
            if(height[l] < height[r]) l ++;
            else r --;
            ans = max(ans, min(height[l], height[r]) * (r - l));
        }
        return ans;
    }
};
// @lc code=end

