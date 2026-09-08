/*
 * @lc app=leetcode.cn id=146 lang=cpp
 *
 * [146] LRU 缓存
 */

 // @lc code=start
class LRUCache {
public:
    struct ListNode {
        int val;
        ListNode* pre, * next;
        ListNode(int val) {
            val = val;
            pre = nullptr;
            next = nullptr;
        }
    };
    unordered_map<int, ListNode*> mp;
    int num;
    ListNode dumm1 = ListNode(-1);
    ListNode dumm2 = ListNode(-1);
    ListNode* head = &dumm1;
    ListNode* end = &dumm2;
    LRUCache(int capacity) {
        num = capacity;
    }

    int get(int key) {
        if (mp[key]) {
            int val = mp[key]->val;
            movetohead(mp[key]);
            return val;
        }
        return -1;
    }

    void put(int key, int val) {
        ListNode tmp = ListNode(val);
        mp[key] = &tmp;
        movetohead(mp[key]);
        if (mp.size() > num) delete_last();
    }

    void movetohead(ListNode* p) {
        if (p->pre->next) p->pre->next = p->next;
        if (p->next->pre) p->next->pre = p->pre;
        p->next = head;
    }

    void delete_last() {
        // mp[end->pre] = nullptr;
        end->pre = end->pre->pre;

        end->pre->pre->next = end;

    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
 // @lc code=end

