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
