/*
 * @lc app=leetcode.cn id=72 lang=cpp
 *
 * [72] 编辑距离
 */

 // @lc code=start
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n1 = word1.size(), n2 = word2.size();
        word1 = " " + word1;
        word2 = " " + word2;
        vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 1e9));

        dp[0][0] = 0;
        for (int i = 1; i <= n2; i++) dp[0][i] = i;
        for (int i = 1; i <= n1; i++) dp[i][0] = i;
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (word1[i] == word2[j]) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
            }
        }
        return dp[n1][n2];
    }
};
// @lc code=end

