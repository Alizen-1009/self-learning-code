class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int n = grid.size();
        int m = grid[0].size();
        vector<vector<int>> dp(n, vector<int>(m, 1e9));
        dp[0][0] = 0;
        for(int i = 0; i < n; i ++){
            for(int j = 0; j < m; j ++){
                if(i) dp[i][j] = min(dp[i][j], dp[i-1][j]);
                if(j) dp[i][j] = min(dp[i][j], dp[i][j-1]);
                dp[i][j] += grid[i][j];
            }
        }
        return dp[n-1][m-1];

    }
};