/*
 * @lc app=leetcode.cn id=25 lang=cpp
 *
 * [25] K 个一组翻转链表
 */

// @lc code=start
/**
 * Definition for singly-linked list.
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
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummy(0, head);
        ListNode *groupNext, *groupPrev;
        
        groupPrev = &dummy;
        while(head != nullptr){
            ListNode *groupEnd = groupPrev;
            ListNode *groupSt = groupPrev -> next;
            for(int i = 1; i <= k; i ++){
                groupEnd = groupEnd -> next;
                if(groupEnd == nullptr) return dummy.next;
            }
            groupNext = groupEnd -> next;

            ListNode *cur = groupSt;
            ListNode *prev = groupNext;
            while(cur != groupNext){
                ListNode *nex = cur -> next;
                cur -> next = prev;
                prev = cur;
                cur = nex;
            }
            groupPrev ->next = groupEnd;
            groupPrev = groupSt;
        }
        return dummy.next;
    }
};
// @lc code=end

