/*
 * @lc app=leetcode.cn id=146 lang=cpp
 *
 * [146] LRU 缓存
 */

 // @lc code=start
class LRUCache {
    struct ListNode {
        int key, val;
        ListNode* pre, * next;
        ListNode(int key, int val) {
            this->key = key;
            this->val = val;
            pre = nullptr;
            next = nullptr;
        }
    };
    int capacity = 0;
    unordered_map<int, ListNode*> mp;
    ListNode* head, * tail;
    void remove(ListNode* p) {
        p->next->pre = p->pre;
        p->pre->next = p->next;
    }

    void addtohead(ListNode* p) {
        head->next->pre = p;
        p->next = head->next;
        p->pre = head;
        head->next = p;
    }

    void movetohead(ListNode* p) {
        remove(p);
        addtohead(p);
    }

    void delete_last() {
        ListNode* p = tail->pre;
        mp.erase(p->key);
        remove(p);
        delete p;
    }

public:
    LRUCache(int capacity) {
        this->capacity = capacity;
        head = new ListNode(0, 0);
        tail = new ListNode(0, 0);
        head->next = tail;
        tail->pre = head;
    }

    int get(int key) {
        if (mp.contains(key)) {
            int val = mp[key]->val;
            movetohead(mp[key]);
            return val;
        }
        return -1;
    }

    void put(int key, int val) {
        if (mp.contains(key)) {
            mp[key]->val = val;
            movetohead(mp[key]);
        }
        else {
            ListNode* tmp = new ListNode(key, val);
            mp[key] = tmp;
            addtohead(mp[key]);
            if (mp.size() > capacity)
                delete_last();
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
 // @lc code=end

