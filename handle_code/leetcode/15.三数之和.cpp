/*
 * @lc app=leetcode.cn id=15 lang=cpp
 *
 * [15] 三数之和
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for(int i = 0; i < n; i ++){
            int k = n - 1;
            if(i && nums[i] == nums[i-1]) continue;
            for(int j = i + 1; j < k; j ++){
                if(j != i + 1 && nums[j] == nums[j-1]) continue;
                int target = (nums[i] + nums[j]) * -1;
                while(k > j && nums[k] > target) k --;
                if(nums[k] == target && j != k) {
                    ans.push_back({nums[i],nums[j], target});
                    continue;
                }
            }
        }
        return ans;
    }
};
// @lc code=end

