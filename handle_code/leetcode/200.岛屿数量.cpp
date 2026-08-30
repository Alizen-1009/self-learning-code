/*
 * @lc app=leetcode.cn id=200 lang=cpp
 *
 * [200] 岛屿数量
 */

// @lc code=start
class Solution {
    int dx[4] = {0,0,1,-1};
    int dy[4] = {1,-1,0,0};
public:
    int numIslands(vector<vector<char>>& g) {
        int n = g.size(), m = g[0].size();
        vector<vector<bool>> vis(n, vector<bool>(m, false));
        

        int ans = 0;
        for(int i = 0; i < n; i ++){
            for(int j = 0; j < m; j ++){
                if(g[i][j] == '0' || vis[i][j]) continue;
                queue<pair<int, int>> q;
                q.push({i, j});
                vis[i][j] = true;
                ans ++;
                while(q.size()){
                    auto [x, y] = q.front();
                    q.pop();
                    for(int k = 0; k < 4; k ++){
                        int tx = x + dx[k];
                        int ty = y + dy[k];
                        if(tx < 0 || tx >= n || ty < 0 || ty >= m) continue;
                        if(vis[tx][ty] || g[tx][ty] == '0') continue;
                        vis[tx][ty] = true;
                        q.push({tx, ty});
                    }
                }
            }
        }
        return ans;
    }
};
// @lc code=end

