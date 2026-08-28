/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while(fast != NULL && fast ->next != NULL){
            fast = fast -> next -> next;
            slow = slow -> next;
            if(slow == fast){
                fast = head;
                while(fast != slow){
                    fast = fast -> next;
                    slow = slow -> next;
                }
                return fast;
            }
        }
        return NULL;
    }
};