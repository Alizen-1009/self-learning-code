/*
 * @lc app=leetcode.cn id=3 lang=cpp
 *
 * [3] 无重复字符的最长子串
 */

// @lc code=start
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size() == 0) return 0;
        unordered_set<char> mp;
        mp.clear();
        int n = s.size();

        int l = 0, r = 1;

        int ans = 1;
        mp.insert(s[0]);
        while(l < n && r < n){
            while(mp.count(s[r])){
                mp.erase(s[l]);
                l ++;
                // cout << l << endl;
            }
            mp.insert(s[r]);
            ans = max(ans, r - l + 1);
            r++;
        }
        return ans;
    }
};
// @lc code=end

