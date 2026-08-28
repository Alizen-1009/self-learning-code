class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int ans;
        int cnt = 0;
        for(auto k : nums){
            if(cnt == 0) ans = k;
            if(ans == k) cnt ++;
            else cnt --;
        }
        return ans;
    }
};