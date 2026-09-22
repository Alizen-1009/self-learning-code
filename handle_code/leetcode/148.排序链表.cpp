/*
 * @lc app=leetcode.cn id=148 lang=cpp
 *
 * [148] 排序链表
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
    ListNode* sortList(ListNode* head) {
        ListNode dummy(0);
        ListNode* cur = head;

        while (cur != nullptr) {
            ListNode* next = cur->next;

            ListNode* pos = &dummy;
            while (pos->next != nullptr &&
                pos->next->val <= cur->val) {
                pos = pos->next;
            }

            cur->next = pos->next;
            pos->next = cur;

            cur = next;
        }

        return dummy.next;
    }
};
// @lc code=end

