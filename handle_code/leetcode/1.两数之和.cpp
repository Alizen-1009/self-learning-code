/*
 * @lc app=leetcode.cn id=1 lang=cpp
 *
 * [1] 两数之和
 */

// @lc code=start
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int, int> mp;
        for(int i = 0; i < nums.size(); i ++) mp[nums[i]] = i;
        for(int i = 0; i < nums.size(); i ++){
            int k = nums[i];
            int num = target - k;
            if(mp.contains(num) && mp[num] != i) return {i, mp[num]};
        }
        return {};

    }
};
// @lc code=end

