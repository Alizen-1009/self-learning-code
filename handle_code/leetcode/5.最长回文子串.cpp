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
        for(int len = 1; len <= n; len ++){
            for(int i = 0; i < n; i ++){
                int j = i + len - 1;
                if(j >= n) continue;
                if(i == j) dp[i][j] = true;
                else if(j == i + 1 && s[i] == s[j]) dp[i][j] = true;
                else if(s[i] == s[j] && j-i>=2) dp[i][j] = dp[i+1][j-1];
            }
        }
        for (int len = n; len >= 1; len--) {
            for (int i = 0; i + len <= n; i++) {
                if (dp[i][i + len - 1]) {
                    return s.substr(i, len);
                }
            }
        }
        return "";
    }
};
// @lc code=end

