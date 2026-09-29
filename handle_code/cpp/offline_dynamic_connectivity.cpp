/*
题目：删边后的无向图连通性查询

【题意】给出一张无向图，以及按时间顺序执行的两类操作：op=1 表示删除边 (u,v)，
其他 op 表示询问 u、v 此时是否连通。这里把重删同一条边视为无操作，删除原本
不存在的边也视为无操作。不同端点编号可以很大，因此需要离散化。

【方法】并查集只擅长加边。先读完所有操作，计算最终仍在图中的边，并用它们建立
并查集。然后把操作倒着走：正向的“删边”变成反向的“加边”；连通性查询比较两个
顶点的并查集根。对重复删边计数，仅当逆向跨过最早那次有效删除时才真正恢复边。
反向得到的回答最后再反转，恢复原查询顺序。

固定例子：初始有 1-2、2-3，随后删除 2-3、询问 1 与 3。最终图只有 1-2，
反向先得到“不连通”，然后再加回 2-3；这正好对应正向查询的状态。

正确性：处理反向时间点 i 前，并查集恰好表示正向完成前 i 个操作后的图。查询
读取当前连通性，恢复边则把图状态退回上一个时间点，所以归纳成立。

复杂度：设不同顶点数为 V，边数 m、操作数 q。排序/集合处理约
O((m+q)log(m+q))，并查集合并近似 O((m+q)α(V))，空间 O(V+m+q)。

English: Answer connectivity queries under edge deletions by processing all
operations backward. Build DSU from edges present at the end; a reversed
deletion restores its edge. Count repeated deletions so an edge is restored only
when reversing the first effective deletion. Vertex IDs are compressed.
*/

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

class DisjointSet {
    vector<int> parent;
    vector<int> size;

public:
    explicit DisjointSet(int n) : parent(n), size(n, 1) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int vertex) {
        while (parent[vertex] != vertex) {
            parent[vertex] = parent[parent[vertex]];
            vertex = parent[vertex];
        }
        return vertex;
    }

    void merge(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
};

struct Operation {
    int type;
    int u;
    int v;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, q;
    cin >> n >> m >> q;
    (void)n;

    unordered_map<int, int> id;
    auto compress = [&](int vertex) {
        auto [it, inserted] = id.emplace(vertex, static_cast<int>(id.size()));
        (void)inserted;
        return it->second;
    };
    auto normalize = [](int u, int v) {
        if (u > v) swap(u, v);
        return make_pair(u, v);
    };

    set<pair<int, int>> initialEdges;
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        initialEdges.insert(normalize(compress(u), compress(v)));
    }

    vector<Operation> operations(q);
    map<pair<int, int>, int> deletionCount;
    for (auto& operation : operations) {
        int u, v;
        cin >> operation.type >> u >> v;
        const auto [a, b] = normalize(compress(u), compress(v));
        operation.u = a;
        operation.v = b;
        if (operation.type == 1 && initialEdges.count({a, b})) {
            ++deletionCount[{a, b}];
        }
    }

    DisjointSet dsu(static_cast<int>(id.size()));
    for (const auto& edge : initialEdges) {
        if (deletionCount[edge] == 0) dsu.merge(edge.first, edge.second);
    }

    vector<string> answers;
    for (int i = q - 1; i >= 0; --i) {
        const auto& operation = operations[i];
        const auto edge = make_pair(operation.u, operation.v);
        if (operation.type == 1) {
            if (initialEdges.count(edge) && --deletionCount[edge] == 0) {
                dsu.merge(edge.first, edge.second);
            }
        } else {
            answers.push_back(dsu.find(operation.u) == dsu.find(operation.v)
                                  ? "Yes" : "No");
        }
    }

    reverse(answers.begin(), answers.end());
    for (const string& answer : answers) cout << answer << '\n';
    return 0;
}
