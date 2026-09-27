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
