class Solution {
public:
    int trapRainWater(vector<vector<int>>& a) {
        int n = a.size(), m = a[0].size();
        vector<vector<int>> l(n + 2, vector<int>(m + 2, 0)), r(n + 2, vector<int>(m + 2, 0));
        vector<vector<int>> up(n + 2, vector<int>(m + 2, 0)),down(n + 2, vector<int>(m + 2, 0));
        for(int i = 1; i <= n; i ++){
            for(int j = 1; j <= m; j ++){
                if(j == 1) continue;
                l[i-1][j-1] = max(a[i-1][j-2], l[i-1][j-2]);
            }
            for(int j = m; j >= 1; j --){
                if(j == m) continue;
                r[i-1][j-1] = max(a[i-1][j], r[i-1][j]);
            }
        }
        

        for(int i = 1; i <= m; i ++){
            for(int j = 1; j <= n; j ++){
                if(j == 1) continue;
                up[j-1][i-1] = max(a[j-2][i-1], up[j-2][i-1]);
            }
            for(int j = n; j >= 1; j --){
                if(j == n) continue;
                down[j-1][i-1] = max(a[j][i-1], down[j][i-1]);
            }
        }

        int ans = 0;
        for(int i = 0; i < n; i ++){
            for(int j = 0; j < m; j ++){
                int res = min(min(l[i][j], r[i][j]),min(up[i][j], down[i][j]));
                ans += max(0,  res - a[i][j]);
            }
        }
        return ans;

    }
};