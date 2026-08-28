class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for(auto k : nums){
            ans ^= k;
        }
        return ans;
    }
};