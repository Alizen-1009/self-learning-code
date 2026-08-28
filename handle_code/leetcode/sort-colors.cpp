class Solution {
public:
    void sortColors(vector<int>& nums) {
        int cnt[3] = {0, 0, 0};
        for(auto k : nums){
            cnt[k] ++;
        }

        int cur = 0;
        for(int i = 0; i < 3; i ++){
            while(cnt[i] > 0){
                nums[cur++] = i;
                cnt[i]--;
            }
        }
    }
};