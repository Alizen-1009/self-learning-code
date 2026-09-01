/*
 * @lc app=leetcode.cn id=32 lang=cpp
 *
 * [32] 最长有效括号
 */

// @lc code=start
class Solution {
public:
    int longestValidParentheses(string s) {
        // sol1
        // vector<int> dp(s.size(), 0);
        // stack<int> st;
        

        // for(int i = 0; i < s.size(); i ++){
        //     if(s[i] == '(') st.push(i);
        //     else {
        //         if(st.size()) {
        //             dp[i] = 1;
        //             dp[st.top()] = 1;
        //             st.pop();
        //         }
        //     }
        // }

        // int cnt = 0, ans = 0;
        // for(int i = 0; i < s.size(); i ++){
        //     if(dp[i] == 1) {
        //         cnt ++;
        //         ans = max(ans, cnt);
        //     }
        //     else{
        //         cnt = 0;
        //     }
        // }
        // return ans;

        int n = s.size();
        int ans = 0;
        vector<int> dp(n, 0);
        for(int i = 1; i < n;  i ++){
            if(s[i] == '(') continue;
            if(s[i-1] == '(') dp[i] = i >= 2 ? dp[i-2] + 2 : 2;
            else{
                int left = i - dp[i-1] - 1;
                if (left >= 0 && s[left] == '(') {
                    dp[i] = dp[i - 1] + 2;

                    if (left >= 1) {
                        dp[i] += dp[left - 1];
                    }
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
// @lc code=end

