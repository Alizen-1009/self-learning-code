/*
 * @lc app=leetcode->cn id=2 lang=cpp
 *
 * [2] 两数相加
 */

// @lc code=start
/**
 * Definition for singly-linked list->
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int extra = 0;
        ListNode dummy;
        ListNode* head = &dummy;
        while(l1 != nullptr || l2 != nullptr || extra){
            int val = extra;
            if(l1 != nullptr){
                val += l1->val;
                l1 = l1 -> next;
            }
            if(l2 != nullptr){
                val += l2 -> val;
                l2 = l2 -> next;
            }

            head->next = new ListNode(val % 10);
            head = head->next;
            extra = val / 10;
        }
        return dummy.next;
    }
};
// @lc code=end

