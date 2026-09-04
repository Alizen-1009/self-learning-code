/*
 * @lc app=leetcode.cn id=48 lang=cpp
 *
 * [48] 旋转图像
 */

// @lc code=start
class Solution {
public:
    void rotate(vector<vector<int>>& m) {
        int n = m.size();
        for(int i = 0; i < n / 2; i ++){
            for(int j = i; j < n - 1 - i; j ++){
                int a = m[i][j], b = m[j][n-1-i], c = m[n-1-i][n-1-j], d = m[n-1-j][i];
                m[j][n-1-i] = a, m[n-1-i][n-1-j] = b, m[n-1-j][i] = c;
                m[i][j] = d;
            }
        }
    }
};
// @lc code=end

