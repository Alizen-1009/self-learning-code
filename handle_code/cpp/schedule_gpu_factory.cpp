/*
中文说明：GPU 工厂评分题。先在预算内雇佣 parts/mount/box 三类工人，再输出严格
递增时间的命令，使恰好 N 块板依次完成三个工序。
解题方法：先保证每站一人，再按“增加一名工人带来的估计吞吐提升/成本”贪心使用
剩余预算。调度阶段用优先队列维护完成事件，每分钟优先启动下游 box、再 mount、
最后 parts；无命令可发时直接跳到下一事件。复杂度 O(N log N)，输出恰好 3N 条命令。
策略保证预算和流程合法，但启发式工人数/调度不保证特殊评测的最优完成时间。

English: GPU Factory scoring solution. Hire legal worker counts, then issue
strictly timed parts/mount/box commands for exactly N boards. Extra budget is
assigned by estimated throughput gain per dollar; an event queue drives a
downstream-first schedule and skips idle time. O(N log N), exactly 3N commands.
The schedule is valid but the scoring heuristic is not guaranteed optimal.
*/
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

using int64 = long long;

struct Event {
    int64 time;
    int station;
    bool operator>(const Event& other) const { return time > other.time; }
};

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

    // The statement guarantees that one worker at every station is affordable.
    // Spend the remaining budget where one extra worker gives the largest
    // estimated throughput improvement per dollar.  A station's output interval
    // is lower-bounded by command rate 1, exit cooldown T, and L / workers.
    std::array<int64, 3> workers{1, 1, 1};
    int64 moneyLeft = budget - cost[0] - cost[1] - cost[2];
    while (true) {
        int best = -1;
        long double bestGainPerDollar = 0;
        for (int s = 0; s < 3; ++s) {
            if (workers[s] >= n || cost[s] > moneyLeft) continue;
            const long double before = std::max<long double>(
                {1.0L, static_cast<long double>(cooldown[s]),
                 static_cast<long double>(length[s]) / workers[s]});
            const long double after = std::max<long double>(
                {1.0L, static_cast<long double>(cooldown[s]),
                 static_cast<long double>(length[s]) / (workers[s] + 1)});
            const long double gainPerDollar = (before - after) / cost[s];
            if (gainPerDollar > bestGainPerDollar) {
                bestGainPerDollar = gainPerDollar;
                best = s;
            }
        }
        if (best == -1) break;
        ++workers[best];
        moneyLeft -= cost[best];
    }

    std::array<int64, 3> freeWorkers = workers;
    std::array<int64, 3> lastFinish{-1, -1, -1};
    std::array<int, 3> started{0, 0, 0};
    int partsStorage = 0;
    int mountedStorage = 0;
    int boxed = 0;

    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;
    std::vector<std::pair<int64, std::string>> commands;

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

    auto startJob = [&](int station, int64 now, const char* name) {
        --freeWorkers[station];
        ++started[station];
        int64 finish = now + length[station];
        if (lastFinish[station] >= 0)
            finish = std::max(finish,
                              lastFinish[station] + cooldown[station]);
        lastFinish[station] = finish;
        events.push({finish, station});
        commands.push_back({now, name});
    };

    int64 now = 0;
    while (boxed < n) {
        finishEvents(now);

        // Downstream-first keeps intermediate storage small and never starts
        // more than the N jobs needed at any station.
        if (mountedStorage > 0 && freeWorkers[2] > 0 && started[2] < n) {
            --mountedStorage;
            startJob(2, now, "box");
            ++now;
        } else if (partsStorage > 0 && freeWorkers[1] > 0 && started[1] < n) {
            --partsStorage;
            startJob(1, now, "mount");
            ++now;
        } else if (freeWorkers[0] > 0 && started[0] < n) {
            startJob(0, now, "parts");
            ++now;
        } else {
            // Omitted minutes are idle, so jump instead of iterating through a
            // possibly 1e6-minute processing interval.
            now = events.top().time;
        }
    }

    std::cout << workers[0] << ' ' << workers[1] << ' ' << workers[2] << '\n';
    for (const auto& [time, command] : commands)
        std::cout << time << ' ' << command << '\n';
}
