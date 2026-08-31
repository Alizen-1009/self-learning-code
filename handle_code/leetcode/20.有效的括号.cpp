/*
 * @lc app=leetcode.cn id=20 lang=cpp
 *
 * [20] 有效的括号
 */

// @lc code=start
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        for(auto k : s){
            if(k == '[' || k == '(' || k == '{') {
                st.push(k);
            }
            else{
                if(k == ')'){
                    if(!st.empty() && st.top() == '(') st.pop();
                    else return false;
                } 
                if(k == ']'){
                    if(!st.empty() && st.top() == '[') st.pop();
                    else return false;
                } 
                if(k == '}'){
                    if(!st.empty() && st.top() == '{') st.pop();
                    else return false;
                } 
            }
        }
        if(st.empty()) return true;
        return false;
    }
};
// @lc code=end

