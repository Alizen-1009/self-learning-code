/*
 * @lc app=leetcode.cn id=23 lang=cpp
 *
 * [23] 合并 K 个升序链表
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
    struct node{
        ListNode * p;
        int val;
        bool operator < (const node &rhs) const{
            return val > rhs.val;
        }
    };
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode dummy(0);
        ListNode *head = &dummy;

        priority_queue<node> q;
        for(auto k : lists){
            if(k != nullptr) q.push({k, k->val});
        }

        while(q.size()){
            auto t = q.top();
            q.pop();
            head -> next = t.p;
            head = head -> next;
            if(head->next != nullptr){
                q.push({head->next, head->next->val});
            }
        }
        return dummy.next;
    }
};
// @lc code=end

