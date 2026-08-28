class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        int pos = -1;
        for(int i = n - 1; i >= 1; i --){
            if(nums[i]>nums[i-1]){
                pos = i;
                break;
            }
        }
        if(~pos){
            int id = pos;
            for(int i = pos; i < n; i ++){
                if(nums[i] > nums[pos-1] && nums[i] < nums[id]){
                    id = i;
                }
            }
            swap(nums[pos-1], nums[id]);
            sort(nums.begin() + pos, nums.end());
        }
        else{
            reverse(nums.begin(), nums.end());
        }
    }
};
