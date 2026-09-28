/*
题目：Jensen's GPU Factory（预算内雇人并安排流水线）

【题意重新表述】
每块 GPU 板必须依次经过三个工序：parts 取回一箱零件；mount 消耗一箱零件并产出
一块半成品；box 消耗一块半成品并产出一块装箱成品。

开始前先永久雇佣 P、M、B 名工人，费用不能超过预算 D，三类工人都至少一名。每个
整数分钟最多发出一条 parts/mount/box 命令；没有输出的分钟自动视为 idle。同一工位
只有空闲工人时才能开工，mount/box 开工时必须立刻消耗对应库存。

工位 s 的一次任务至少执行 L_s 分钟。另外，该工位相邻两次“完成”的时间差至少为
T_s；若某任务本来应该更早完成，它会占着工人继续等待，直到冷却间隔满足才完成。
需要输出合法的工人数和命令序列，让第 N 块成品尽早完成。这是评分题：完成时间越短，
分数越高。

【为什么原来的工人分配不够好】
原实现按 max(1,T_s,L_s/workers) 衡量每个工位，等价于希望三个工位都达到“每分钟
完成一件”。但每块板必须发三条命令，而全系统每分钟最多只发一条命令，所以长期
吞吐量最多约为“每 3 分钟一块”。继续把某个工位从每 3 分钟提升到每 1 分钟，经常
只会浪费预算。

【改进方案】
1. 生成多组合法工人数候选：1/1/1；以全局瓶颈 3 分钟为目标按单位费用的吞吐改善
   贪心加人；以 1 分钟为目标生成激进候选；再加入把预算集中给单站或依次买满各站的
   极端配置。
2. 对每组工人数实际模拟六种工位优先级。最小堆维护完成事件，严格执行工人占用、
   完成冷却、库存消耗和“一分钟一条命令”。
3. 选择实际模拟完成时间最短的候选，而不是只凭近似吞吐公式决定最终输出。

固定例子：样例 N=3、D=10、三类成本都是 2，L=(5,4,3)、T=(3,2,2)。每块板需要
三条命令，所以先以周期 3 估算：parts 需 ceil(5/3)=2 人，mount 需 ceil(4/3)=2
人，box 需 ceil(3/3)=1 人，得到样例中的 2 2 1。

合法性：所有候选从 1/1/1 开始且只在预算允许时加人；模拟只在有空闲工人和足够
库存时发命令，并用 finish=max(start+L,lastFinish+T) 计算真实完成时刻。因此输出
一定合法。它比单一贪心覆盖更多方案，但评分题仍无法据此证明全局最优。

复杂度：候选数和优先级数均为常数。每次模拟启动 3N 个任务，时间 O(N log N)，
空间 O(N)，最终输出 3N 条非 idle 命令。

English: Hire workers for a three-stage parts/mount/box pipeline and emit a
legal one-command-per-minute schedule. This version accounts for the global
three-commands-per-board bottleneck, generates several affordable allocations,
simulates all six station priorities under the exact worker/inventory/cooldown
rules, and prints the candidate with the earliest measured completion. Each
simulation is O(N log N). The result is legal and stronger than one heuristic,
but the scoring optimization is not claimed globally optimal.
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

using int64 = long long;
using Workers = std::array<int64, 3>;

struct Event {
    int64 time;
    int station;
    bool operator>(const Event& other) const { return time > other.time; }
};

struct Schedule {
    int64 finishTime;
    Workers workers;
    std::vector<std::pair<int64, int>> commands;
};

int64 hiringCost(const Workers& workers, const std::array<int64, 3>& cost) {
    return workers[0] * cost[0] + workers[1] * cost[1] +
           workers[2] * cost[2];
}

Workers greedyWorkers(int n, int64 budget,
                      const std::array<int64, 3>& cost,
                      const std::array<int64, 3>& length,
                      const std::array<int64, 3>& cooldown,
                      long double globalInterval) {
    Workers workers{1, 1, 1};
    int64 moneyLeft = budget - hiringCost(workers, cost);

    while (true) {
        int bestStation = -1;
        long double bestGainPerDollar = 0;
        for (int station = 0; station < 3; ++station) {
            if (workers[station] >= n || cost[station] > moneyLeft) continue;
            const long double before = std::max<long double>(
                {globalInterval, static_cast<long double>(cooldown[station]),
                 static_cast<long double>(length[station]) / workers[station]});
            const long double after = std::max<long double>(
                {globalInterval, static_cast<long double>(cooldown[station]),
                 static_cast<long double>(length[station]) /
                     (workers[station] + 1)});
            const long double gain =
                (before - after) / static_cast<long double>(cost[station]);
            if (gain > bestGainPerDollar) {
                bestGainPerDollar = gain;
                bestStation = station;
            }
        }
        if (bestStation == -1) break;
        ++workers[bestStation];
        moneyLeft -= cost[bestStation];
    }
    return workers;
}

Schedule buildSchedule(int n, const Workers& workers,
                       const std::array<int64, 3>& length,
                       const std::array<int64, 3>& cooldown,
                       const std::array<int, 3>& priority) {
    Workers freeWorkers = workers;
    std::array<int64, 3> lastFinish{-1, -1, -1};
    std::array<int, 3> started{0, 0, 0};
    int partsStorage = 0;
    int mountedStorage = 0;
    int boxed = 0;

    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;
    std::vector<std::pair<int64, int>> commands;
    commands.reserve(3LL * n);

    auto finishEvents = [&](int64 now) {
        while (!events.empty() && events.top().time <= now) {
            const int station = events.top().station;
            events.pop();
            ++freeWorkers[station];
            if (station == 0) ++partsStorage;
            else if (station == 1) ++mountedStorage;
            else ++boxed;
        }
    };

    auto canStart = [&](int station) {
        if (started[station] >= n || freeWorkers[station] == 0) return false;
        if (station == 1 && partsStorage == 0) return false;
        if (station == 2 && mountedStorage == 0) return false;
        return true;
    };

    auto startJob = [&](int station, int64 now) {
        if (station == 1) --partsStorage;
        if (station == 2) --mountedStorage;
        --freeWorkers[station];
        ++started[station];

        int64 finish = now + length[station];
        if (lastFinish[station] >= 0) {
            finish = std::max(finish,
                              lastFinish[station] + cooldown[station]);
        }
        lastFinish[station] = finish;
        events.push({finish, station});
        commands.push_back({now, station});
    };

    int64 now = 0;
    while (boxed < n) {
        finishEvents(now);
        if (boxed == n) break;

        int chosen = -1;
        for (int station : priority) {
            if (canStart(station)) {
                chosen = station;
                break;
            }
        }

        if (chosen != -1) {
            startJob(chosen, now);
            ++now;
        } else {
            now = events.top().time;
        }
    }

    return {now, workers, std::move(commands)};
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    int64 budget;
    std::array<int64, 3> cost{}, length{}, cooldown{};
    if (!(std::cin >> n)) return 0;
    std::cin >> budget >> cost[0] >> cost[1] >> cost[2];
    std::cin >> length[0] >> length[1] >> length[2];
    std::cin >> cooldown[0] >> cooldown[1] >> cooldown[2];

    std::vector<Workers> candidates;
    auto addCandidate = [&](const Workers& workers) {
        if (hiringCost(workers, cost) > budget) return;
        if (std::find(candidates.begin(), candidates.end(), workers) ==
            candidates.end()) {
            candidates.push_back(workers);
        }
    };

    const Workers baseline{1, 1, 1};
    addCandidate(baseline);
    addCandidate(greedyWorkers(n, budget, cost, length, cooldown, 3.0L));
    addCandidate(greedyWorkers(n, budget, cost, length, cooldown, 1.0L));

    for (int station = 0; station < 3; ++station) {
        Workers workers = baseline;
        const int64 left = budget - hiringCost(workers, cost);
        workers[station] +=
            std::min<int64>(n - 1, left / cost[station]);
        addCandidate(workers);
    }

    std::array<int, 3> order{0, 1, 2};
    do {
        Workers workers = baseline;
        int64 left = budget - hiringCost(workers, cost);
        for (int station : order) {
            const int64 extra =
                std::min<int64>(n - workers[station], left / cost[station]);
            workers[station] += extra;
            left -= extra * cost[station];
        }
        addCandidate(workers);
    } while (std::next_permutation(order.begin(), order.end()));

    const std::array<std::array<int, 3>, 6> priorities{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
    }};

    Schedule best{std::numeric_limits<int64>::max(), baseline, {}};
    for (const Workers& workers : candidates) {
        for (const auto& priority : priorities) {
            Schedule current =
                buildSchedule(n, workers, length, cooldown, priority);
            if (current.finishTime < best.finishTime) {
                best = std::move(current);
            }
        }
    }

    static const std::array<const char*, 3> commandName{
        "parts", "mount", "box"};
    std::cout << best.workers[0] << ' ' << best.workers[1] << ' '
              << best.workers[2] << '\n';
    for (const auto& [time, station] : best.commands) {
        std::cout << time << ' ' << commandName[station] << '\n';
    }
    return 0;
}
