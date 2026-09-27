/*
题目名称：避开禁用标签串的最短路线及方案数

【中文题意】
物流网络有 n 个仓库和 m 条带正整数费用的有向道路。每条道路还有一个 A/B/C 标签。
从 s 到 t 行驶时，按经过顺序连接道路标签，得到路线标签串。给定非空禁用串 P，
只有标签串不包含 P 作为连续子串的路线才合法。允许重复经过仓库和道路，平行边、
自环也存在；道路按输入编号，不同道路编号序列视为不同路线。

输出合法路线的最小总费用，以及达到该费用的不同路线数（模 1,000,000,007）。
若 s=t，空路线合法，费用为 0、标签串为空。若不存在合法路线，本实现输出 -1 0。

输入格式：
第一行 n、m、s、t；第二行禁用串 P；接下来 m 行为 u、v、w、ch，表示一条编号
由输入顺序确定的 u->v 道路，费用 w，标签 ch。

【解题方法：KMP 自动机 × Dijkstra】
仅知道当前仓库不足以判断继续走一条边后是否出现 P，还需要记录当前标签串后缀与 P
前缀匹配了多长。令状态 k（0<=k<|P|）表示最长匹配长度。用 KMP 预处理状态 k 读入
A/B/C 后的新状态；若新状态等于 |P|，说明刚形成禁用串，这条转移直接丢弃。

这样得到乘积图状态 (u,k)：人在仓库 u，当前匹配长度为 k。乘积图边仍为正权，故从
(s,0) 运行 Dijkstra。松弛到更短距离时覆盖方案数；距离相等时累加方案数。原图中的
每条平行边都单独遍历，所以自然按照道路编号序列计数。最后在所有 (t,k) 中取最短
距离并汇总具有该距离的方案数。

正确性要点：KMP 状态完整刻画了未来是否会首次出现 P；合法原路线与乘积图中不进入
禁用状态的路径一一对应，且费用、道路编号序列不变。Dijkstra 因边权均为正数可正确
求最短距离，并按标准等距松弛统计全部最短路径。

复杂度：乘积图最多 n|P| 个状态、m|P| 条转移；时间
O((n|P|+m|P|)log(n|P|))，空间 O(n|P|+m)。

English: Find the cheapest s-to-t walk whose A/B/C edge-label string does not
contain forbidden pattern P, and count cheapest edge-ID sequences modulo 1e9+7.
Build a KMP automaton state recording the longest suffix matching a prefix of P;
discard transitions reaching |P|. Run Dijkstra with path counting on product
states (vertex, automaton_state), then combine minimum-distance states at t.
Complexity: O((n|P|+m|P|) log(n|P|)) time and O(n|P|+m) space.
*/

#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

using namespace std;
using int64 = long long;

constexpr int MOD = 1'000'000'007;
constexpr int64 INF = numeric_limits<int64>::max() / 4;

struct Edge {
    int to;
    int64 cost;
    int label;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, source, target;
    cin >> n >> m >> source >> target;

    string forbidden;
    cin >> forbidden;
    const int patternLength = static_cast<int>(forbidden.size());

    vector<vector<Edge>> graph(n + 1);
    for (int edgeId = 0; edgeId < m; ++edgeId) {
        int from, to;
        int64 cost;
        char label;
        cin >> from >> to >> cost >> label;
        graph[from].push_back({to, cost, label - 'A'});
    }

    vector<int> prefixFunction(patternLength, 0);
    for (int i = 1; i < patternLength; ++i) {
        int matched = prefixFunction[i - 1];
        while (matched > 0 && forbidden[i] != forbidden[matched]) {
            matched = prefixFunction[matched - 1];
        }
        if (forbidden[i] == forbidden[matched]) ++matched;
        prefixFunction[i] = matched;
    }

    vector<array<int, 3>> transition(patternLength);
    for (int state = 0; state < patternLength; ++state) {
        for (int label = 0; label < 3; ++label) {
            const char character = static_cast<char>('A' + label);
            int matched = state;
            while (matched > 0 && forbidden[matched] != character) {
                matched = prefixFunction[matched - 1];
            }
            if (forbidden[matched] == character) ++matched;
            transition[state][label] = matched;
        }
    }

    const int stateCount = (n + 1) * patternLength;
    auto encode = [patternLength](int vertex, int state) {
        return vertex * patternLength + state;
    };

    vector<int64> distance(stateCount, INF);
    vector<int> ways(stateCount, 0);
    using QueueNode = pair<int64, int>;
    priority_queue<QueueNode, vector<QueueNode>, greater<QueueNode>> queue;

    const int startId = encode(source, 0);
    distance[startId] = 0;
    ways[startId] = 1;
    queue.push({0, startId});

    while (!queue.empty()) {
        const auto [currentDistance, id] = queue.top();
        queue.pop();
        if (currentDistance != distance[id]) continue;

        const int vertex = id / patternLength;
        const int automatonState = id % patternLength;

        for (const Edge& edge : graph[vertex]) {
            const int nextState = transition[automatonState][edge.label];
            if (nextState == patternLength) continue;

            const int nextId = encode(edge.to, nextState);
            const int64 nextDistance = currentDistance + edge.cost;
            if (nextDistance < distance[nextId]) {
                distance[nextId] = nextDistance;
                ways[nextId] = ways[id];
                queue.push({nextDistance, nextId});
            } else if (nextDistance == distance[nextId]) {
                ways[nextId] += ways[id];
                if (ways[nextId] >= MOD) ways[nextId] -= MOD;
            }
        }
    }

    int64 bestDistance = INF;
    int answerWays = 0;
    for (int state = 0; state < patternLength; ++state) {
        const int id = encode(target, state);
        if (distance[id] < bestDistance) {
            bestDistance = distance[id];
            answerWays = ways[id];
        } else if (distance[id] == bestDistance) {
            answerWays += ways[id];
            if (answerWays >= MOD) answerWays -= MOD;
        }
    }

    if (bestDistance == INF) {
        cout << -1 << ' ' << 0 << '\n';
    } else {
        cout << bestDistance << ' ' << answerWays << '\n';
    }
    return 0;
}
