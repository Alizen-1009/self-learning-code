/*
 * @lc app=leetcode.cn id=46 lang=cpp
 *
 * [46] 全排列
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> ans;
    void dfs(vector<int> nums, vector<int> tmp, unordered_map<int,bool> mp){
        if(tmp.size() == nums.size()) {
            ans.push_back(tmp);
            return;
        }
        for(int i = 0; i < nums.size(); i ++){
            if(mp[i]) continue;
            tmp.push_back(nums[i]);
            mp[i] = 1;
            dfs(nums, tmp, mp);
            tmp.pop_back();
            mp[i] = 0;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        dfs(nums, {}, {});
        return ans;
    }
};
// @lc code=end

