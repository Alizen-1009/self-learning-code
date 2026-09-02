/*
 * @lc app=leetcode.cn id=39 lang=cpp
 *
 * [39] 组合总和
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> ans;
    void dfs(vector<int>& cand, int now, int target, vector<int> res, int st){
        if(now == target){
            ans.push_back(res);
            return;
        }
        for(int i = st; i < cand.size(); i ++){
            int k = cand[i];
            if(now + k > target) break;
            res.push_back(k);
            dfs(cand, now + k, target, res, i);
            res.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& cand, int target) {
        // vector<vector<int>> dp[target + 1];
        // for(auto k : cand){
        //     dp[k].push_back({k});
        // }
        // for(int i = 1; i <= target; i ++){
        //     for(auto k : cand){
        //         if(i > k) {
        //             for(vector<int> v : dp[i-k]){
        //                 v.push_back(k);
        //                 dp[i].push_back(v);
        //             }
        //         }
        //         if(i == k){
        //             dp[i].push_back({k});
        //         }
        //     }
        // }
        // return dp[target];
        sort(cand.begin(), cand.end());
        dfs(cand, 0, target, {}, 0);
        return ans;
    }
};
// @lc code=end

