/*
 * @lc app=leetcode.cn id=42 lang=cpp
 *
 * [42] 接雨水
 */

// @lc code=start
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        vector<int> l(n, 0), r(n, 0);
        
        for(int i = 1; i < n; i ++)  l[i] = max(l[i-1], height[i-1]);
        for(int i = n - 2; ~i; i --) r[i] = max(r[i+1], height[i+1]);

        int ans = 0;
        for(int i = 0; i < n; i ++){
            // cout << l[i] << " " << r[i] << '\n';
            ans += max(0, min(l[i], r[i]) - height[i]);
        }
        return ans;
    }
};
// @lc code=end

