/*
 * @lc app=leetcode.cn id=22 lang=cpp
 *
 * [22] 括号生成
 */

// @lc code=start
class Solution {
public:
    vector<string> ans;
    void dfs(string now, int l, int r, int n){
        if(l + r == 2 * n) ans.push_back(now);    
        if(l < n) dfs(now + '(', l + 1, r, n);
        if(l > r) dfs(now + ')', l, r + 1, n);

    }
    vector<string> generateParenthesis(int n) {
        
        dfs("", 0, 0, n);
        return ans;
    }
};
// @lc code=end

