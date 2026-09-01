/*
 * @lc app=leetcode.cn id=24 lang=cpp
 *
 * [24] 两两交换链表中的节点
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
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy(0, head);
        ListNode *prev = &dummy;
        while(head != nullptr && head -> next != nullptr) {
            ListNode *a = head, *b = head->next, *c = head->next->next;
            a -> next = c;
            b -> next = a;
            prev -> next = b;
            head = c;
            prev = a;
        }
        return dummy.next;
    }
};
// @lc code=end

