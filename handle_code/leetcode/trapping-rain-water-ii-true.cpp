class Solution {
    struct node{
        int x, y;
        int h;
        bool operator < (const node& rhs) const{
            return h > rhs.h;
        }
    };
    int dx[4]={0,0,1,-1};
    int dy[4]={1,-1,0,0};
public:
    int trapRainWater(vector<vector<int>>& a) {
        int n = a.size(), m = a[0].size();
        
        priority_queue<node> q;
        vector<vector<int>> vis(n, vector<int>(m, 0));

        for(int i = 0; i < n; i ++){
            for(int j = 0; j < m; j ++){
                if(i == 0 || j == 0 || i == n-1 || j==m-1){
                    vis[i][j] = 1;
                    q.push({i, j, a[i][j]});
                }
            }
        }

        int ans = 0;
        while(!q.empty()){
            auto [x, y, h] = q.top();
            q.pop();

            for(int k = 0; k < 4; k ++){
                int tx = x + dx[k];
                int ty = y + dy[k];
                if(tx < 0 || tx >= n || ty < 0 || ty >= m) continue;
                if(vis[tx][ty]) continue;
                
                ans += max(0, h - a[tx][ty]);

                vis[tx][ty] = 1;
                q.push({tx, ty, max(h,a[tx][ty])});

            }
        }
        return ans;


    }
};