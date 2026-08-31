/*
 * @lc app=leetcode.cn id=17 lang=cpp
 *
 * [17] 电话号码的字母组合
 */

// @lc code=start
class Solution {
public:
    string s[10]={"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    vector<string> letterCombinations(string digits) {
        vector<string> ans;
        ans.push_back("");
        for(char c : digits){
            int num = c - '0';
            vector<string> tmp;
            for(auto k : ans){
                for(auto c2 : s[num]){
                    tmp.push_back(k + c2);
                }
            }
            ans = tmp;
        }
        return ans;    
    }
};
// @lc code=end

