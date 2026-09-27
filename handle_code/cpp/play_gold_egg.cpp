/*
中文说明：6×6 Gold Egg 交互评分题。奖品格返回 PRIZE，相邻格返回 HINT，其余格
返回 EMPTY；目标是在有限猜测次数内反复找到奖品。
解题方法：用 36 位掩码保存候选奖品位置。每次选择能让 HINT/EMPTY 两个最坏分支
尽量小的格子，根据反馈取交集或删去闭邻域。程序先离线模拟 36 个奖品位置，算出
该策略每轮最坏需要 12 次，只在剩余次数足够保证完成一轮时开始新一轮。
复杂度：棋盘固定，单轮为常数开销；这是保证完成的启发式策略，不保证评分最优。

English: Interactive 6x6 Gold Egg scoring strategy. A 36-bit mask tracks prize
candidates. Each guess greedily minimizes the larger HINT/EMPTY branch, then
intersects or removes the queried closed neighborhood. Exhaustive simulation
shows a 12-guess worst case, so a new round starts only when it can be completed.
Constant work per fixed-size round; valid but not claimed score-optimal.
*/
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

// These names/functions are supplied by the special-judge skeleton.
extern const int EMPTY_EGG;
extern const int HINT_EGG;
extern const int PRIZE;
extern const int ILLEGAL_INPUT;
int guess(int row, int col);

namespace {
using Mask = std::uint64_t;
constexpr int SIDE = 6;
constexpr int CELLS = SIDE * SIDE;

Mask closedNeighborhood(int cell) {
    const int r = cell / SIDE;
    const int c = cell % SIDE;
    Mask mask = Mask{1} << cell;
    if (r > 0) mask |= Mask{1} << (cell - SIDE);
    if (r + 1 < SIDE) mask |= Mask{1} << (cell + SIDE);
    if (c > 0) mask |= Mask{1} << (cell - 1);
    if (c + 1 < SIDE) mask |= Mask{1} << (cell + 1);
    return mask;
}

int chooseCell(Mask candidates, Mask opened) {
    if ((candidates & (candidates - 1)) == 0)
        return __builtin_ctzll(candidates);

    int bestCell = -1;
    int bestWorst = std::numeric_limits<int>::max();
    int bestUseful = -1;
    for (int cell = 0; cell < CELLS; ++cell) {
        if (opened >> cell & 1ULL) continue;
        const Mask prizeBit = Mask{1} << cell;
        const Mask adjacent = closedNeighborhood(cell) & ~prizeBit;
        const int hintCount = __builtin_popcountll(candidates & adjacent);
        const int emptyCount =
            __builtin_popcountll(candidates & ~closedNeighborhood(cell));
        const int worst = std::max(hintCount, emptyCount);
        const int useful = hintCount + ((candidates & prizeBit) != 0);
        if (worst < bestWorst || (worst == bestWorst && useful > bestUseful)) {
            bestWorst = worst;
            bestUseful = useful;
            bestCell = cell;
        }
    }
    return bestCell;
}

// Number of guesses this deterministic strategy needs for a fixed prize.
int simulateRound(int prizeCell) {
    Mask candidates = (Mask{1} << CELLS) - 1;
    Mask opened = 0;
    for (int used = 1; used <= CELLS; ++used) {
        const int cell = chooseCell(candidates, opened);
        if (cell == prizeCell) return used;
        opened |= Mask{1} << cell;
        if (closedNeighborhood(cell) >> prizeCell & 1ULL)
            candidates &= closedNeighborhood(cell) & ~(Mask{1} << cell);
        else
            candidates &= ~closedNeighborhood(cell);
    }
    return CELLS;
}
}  // namespace

void luckyDraw(int n) {
    int worstCase = 0;
    for (int prize = 0; prize < CELLS; ++prize)
        worstCase = std::max(worstCase, simulateRound(prize));

    int remaining = n;
    while (remaining >= worstCase) {
        Mask candidates = (Mask{1} << CELLS) - 1;
        Mask opened = 0;

        while (true) {
            const int cell = chooseCell(candidates, opened);
            const int result = guess(cell / SIDE, cell % SIDE);
            --remaining;

            if (result == PRIZE) break;
            opened |= Mask{1} << cell;
            if (result == HINT_EGG) {
                candidates &= closedNeighborhood(cell) & ~(Mask{1} << cell);
            } else if (result == EMPTY_EGG) {
                candidates &= ~closedNeighborhood(cell);
            } else {
                return;  // Defensive: never continue after an illegal response.
            }
        }
    }
}
