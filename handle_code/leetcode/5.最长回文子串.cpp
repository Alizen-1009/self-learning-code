/*
 * @lc app=leetcode.cn id=5 lang=cpp
 *
 * [5] 最长回文子串
 */

// @lc code=start
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        for(int i = 0; i < n; i ++){
            dp[i][i] = true;
            for(int j = i + 1; j < n; j ++){
                if(i && j < n - 1 && s[i] == s[j]){
                    dp[i][j] == dp[i+1][j-1];
                }
            }
        }
        for(int i = 0; i < n; i ++){
            for(int len = n; len >= 1; len --){
                if(i + len -1 < n){
                    if(dp[i][i+len-1]){
                        return s.substr(i, i + len);
                    } 
                }
            }
        }
        return "";
    }
};
// @lc code=end

