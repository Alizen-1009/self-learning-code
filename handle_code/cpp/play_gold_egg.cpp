/*
题目：Jensen's Gold Egg（6×6 交互寻宝）

【题意重新表述】
棋盘有 6×6=36 个蛋，每轮恰好一个蛋里有奖品。奖品上下左右相邻且仍在棋盘内的蛋
都是提示蛋，其他蛋为空。调用 guess(row,col) 会砸开一个尚未打开的蛋，并返回：

    PRIZE     ：这个格子就是奖品；本轮立刻结束并刷新整张棋盘
    HINT_EGG  ：奖品在这个格子的上、下、左、右之一
    EMPTY_EGG ：这个格子及其四邻域都不是奖品

总共只有 n 次 guess。每找到一次奖品获得 Happiness，但一轮最坏可能使用的猜测次数
越多，隐藏的 Unhappy 惩罚可能越大。因此策略既要多完成轮次，也要控制最坏猜测数。

【固定例子】
若猜 (2,2) 得到 HINT，只剩 (1,2)、(3,2)、(2,1)、(2,3) 四个候选；若得到 EMPTY，
则这五个格子可以一次全部排除；若得到 PRIZE，下一次 guess 已属于全新的棋盘。

【状态表示】
36 位整数 candidates 表示仍可能藏奖品的格子，opened 表示本轮已经砸过、不能再次
调用的格子。收到 HINT 时令 candidates 与四邻域取交集；收到 EMPTY 时删掉“自己+
四邻域”；PRIZE 时重置两个集合。

【比原来一层贪心更好的选点】
候选较多时，选择让 HINT/EMPTY 两个分支中较大者尽量小的格子。候选缩小到 10 个
以内后，不再只看下一步，而是做 4 层 minimax 前瞻：对每个可猜格子同时递归评估
HINT 与 EMPTY 的后续最坏分支，选择最坏代价最小的格子。搜索结果用
(candidates,opened,depth) 记忆化。

程序会在开始时把奖品分别放在 36 个位置进行完整模拟，从而得到这套确定性策略的
真实最坏次数。原来的一层贪心最坏需要 12 次；改进策略为 11 次。只有剩余预算足够
保证完成一轮时才开新局，避免把猜测浪费在无法保证收尾的最后一轮。

复杂度：棋盘大小固定。记忆化前瞻只在候选不超过 10 时启用，状态数有固定上界；
每轮实际 guess 最多 11 次。它是更强且有穷举验证的启发式策略，但不声称 11 是该题
所有可能策略的理论最优值。

English: A 36-bit candidate mask represents possible prize cells. HINT keeps
the four-neighbor set; EMPTY removes the queried cell and those neighbors.
Large states use balanced one-step splitting. Once at most ten candidates
remain, a memoized four-ply minimax evaluates future HINT/EMPTY branches. An
exhaustive simulation of all 36 prize positions verifies an 11-guess worst case,
improving the former greedy policy's 12. New rounds start only with enough
remaining guesses to guarantee completion.
*/
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <unordered_map>

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
constexpr int LOOKAHEAD_THRESHOLD = 10;
constexpr int LOOKAHEAD_DEPTH = 4;

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

struct SearchKey {
    Mask candidates;
    Mask opened;
    int depth;

    bool operator==(const SearchKey& other) const {
        return candidates == other.candidates && opened == other.opened &&
               depth == other.depth;
    }
};

struct SearchKeyHash {
    std::size_t operator()(const SearchKey& key) const {
        std::uint64_t value = key.candidates;
        value ^= key.opened + 0x9e3779b97f4a7c15ULL + (value << 6) +
                 (value >> 2);
        value ^= static_cast<std::uint64_t>(key.depth) *
                 0xbf58476d1ce4e5b9ULL;
        return static_cast<std::size_t>(value);
    }
};

std::unordered_map<SearchKey, int, SearchKeyHash> searchMemo;

int greedyCell(Mask candidates, Mask opened) {
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

int estimateWorst(Mask candidates, Mask opened, int depth) {
    const int count = __builtin_popcountll(candidates);
    if (count <= 1) return count;
    if (depth == 0) return count;  // Sequential search is a safe leaf estimate.

    const SearchKey key{candidates, opened, depth};
    const auto memoIt = searchMemo.find(key);
    if (memoIt != searchMemo.end()) return memoIt->second;

    int best = CELLS;
    for (int cell = 0; cell < CELLS; ++cell) {
        if (opened >> cell & 1ULL) continue;

        const Mask hint = candidates &
                          (closedNeighborhood(cell) & ~(Mask{1} << cell));
        const Mask empty = candidates & ~closedNeighborhood(cell);
        if (hint == candidates || empty == candidates) continue;

        const Mask nextOpened = opened | (Mask{1} << cell);
        int worst = 1;  // PRIZE branch ends immediately.
        if (hint) {
            worst = std::max(
                worst, 1 + estimateWorst(hint, nextOpened, depth - 1));
        }
        if (empty) {
            worst = std::max(
                worst, 1 + estimateWorst(empty, nextOpened, depth - 1));
        }
        best = std::min(best, worst);
    }

    searchMemo.emplace(key, best);
    return best;
}

int chooseCell(Mask candidates, Mask opened) {
    if ((candidates & (candidates - 1)) == 0)
        return __builtin_ctzll(candidates);
    if (__builtin_popcountll(candidates) > LOOKAHEAD_THRESHOLD)
        return greedyCell(candidates, opened);

    int bestCell = -1;
    int bestWorst = std::numeric_limits<int>::max();
    int bestLargestBranch = std::numeric_limits<int>::max();
    for (int cell = 0; cell < CELLS; ++cell) {
        if (opened >> cell & 1ULL) continue;

        const Mask hint = candidates &
                          (closedNeighborhood(cell) & ~(Mask{1} << cell));
        const Mask empty = candidates & ~closedNeighborhood(cell);
        if (hint == candidates || empty == candidates) continue;

        const Mask nextOpened = opened | (Mask{1} << cell);
        int worst = 1;
        if (hint) {
            worst = std::max(
                worst,
                1 + estimateWorst(hint, nextOpened, LOOKAHEAD_DEPTH - 1));
        }
        if (empty) {
            worst = std::max(
                worst,
                1 + estimateWorst(empty, nextOpened, LOOKAHEAD_DEPTH - 1));
        }
        const int largestBranch = std::max(__builtin_popcountll(hint),
                                           __builtin_popcountll(empty));
        if (worst < bestWorst ||
            (worst == bestWorst && largestBranch < bestLargestBranch)) {
            bestWorst = worst;
            bestLargestBranch = largestBranch;
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
