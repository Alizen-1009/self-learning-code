
#include <unordered_map>

class LRUCache {
    struct Node {
        int key;
        int value;
        Node* next, * pre;

        Node(int k, int v) : key(k), value(v), next(nullptr), pre(nullptr) {}
    };
    int capacity;
    int size;
    Node* head;
    Node* tail;
    std::unordered_map<int, Node*> cache;

    void remove(Node* node) {
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }

    void addToHead(Node* node) {
        node->pre = head;
        node->next = head->next;
        head->next->pre = node;
        head->next = node;
    }

    void moveToHead(Node* node) {
        remove(node);
        addToHead(node);
    }

    Node* removeTail() {
        Node* node = tail->pre;
        remove(node);
        return node;
    }

public:
    LRUCache(int capacity) : capacity(capacity), size(0) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->pre = head;
    }

    int get(int key) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            return -1;
        }

        moveToHead(it->second);
        return it->second->value;
    }

    void put(int key, int value) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            it->second->value = value;
            moveToHead(it->second);
            return;
        }

        Node* node = new Node(key, value);
        cache[key] = node;
        addToHead(node);
        ++size;

        if (size > capacity) {
            Node* removed = removeTail();
            cache.erase(removed->key);
            delete removed;
            --size;
        }
    }

    ~LRUCache() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }
};
