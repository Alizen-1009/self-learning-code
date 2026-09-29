/*
题目：4×4 棋盘上骑士随机游走回到角落的期望步数

【题意】骑士从左上角出发，每一步都在当前所有合法的骑士走法中等概率选择一个。
至少走一步后，第一次到达任意角落时停止，求所需步数的数学期望。初始角落不算到达。

【方法】原代码用一百万次随机模拟，只能得到带随机误差的近似值。这里对每个非角落
格子建立期望 E(x,y)：

    E(x,y) = 1 + 所有下一格 E 的平均值。

角落是吸收状态，E=0。4×4 棋盘只有 12 个非角落状态，因此建立 12 元一次方程组并
用高斯消元求解。起点必须先走一步，所以答案是 1 加上起点所有下一格期望的平均值。

正确性来自期望的全期望公式；线性方程直接描述了每个状态的一步转移，因此没有蒙特
卡洛误差。时间复杂度 O(S^3)，S=12；空间 O(S^2)，在本题中都是常数。

English: Compute the exact expected number of uniformly random legal knight
moves needed to reach any corner again on a 4x4 board. Corners are absorbing
after the first move. Set E(s)=1+average(E(next)) for the 12 non-corner states
and solve the linear system by Gaussian elimination. This replaces the previous
Monte Carlo estimate with a deterministic answer.
*/

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

constexpr int SIDE = 4;
constexpr array<int, 8> DX{-1, -1, 1, 1, 2, 2, -2, -2};
constexpr array<int, 8> DY{2, -2, 2, -2, 1, -1, 1, -1};

bool isCorner(int x, int y) {
    return (x == 0 || x == SIDE - 1) && (y == 0 || y == SIDE - 1);
}

vector<pair<int, int>> legalMoves(int x, int y) {
    vector<pair<int, int>> result;
    for (int direction = 0; direction < 8; ++direction) {
        const int nextX = x + DX[direction];
        const int nextY = y + DY[direction];
        if (0 <= nextX && nextX < SIDE && 0 <= nextY && nextY < SIDE) {
            result.push_back({nextX, nextY});
        }
    }
    return result;
}

int main() {
    array<array<int, SIDE>, SIDE> id{};
    for (auto& row : id) row.fill(-1);

    vector<pair<int, int>> states;
    for (int x = 0; x < SIDE; ++x) {
        for (int y = 0; y < SIDE; ++y) {
            if (!isCorner(x, y)) {
                id[x][y] = static_cast<int>(states.size());
                states.push_back({x, y});
            }
        }
    }

    const int stateCount = static_cast<int>(states.size());
    vector<vector<long double>> matrix(
        stateCount, vector<long double>(stateCount + 1, 0));

    for (int state = 0; state < stateCount; ++state) {
        const auto [x, y] = states[state];
        const auto moves = legalMoves(x, y);
        matrix[state][state] = 1;
        matrix[state][stateCount] = 1;
        for (const auto& [nextX, nextY] : moves) {
            if (!isCorner(nextX, nextY)) {
                matrix[state][id[nextX][nextY]] -=
                    1.0L / static_cast<long double>(moves.size());
            }
        }
    }

    for (int column = 0; column < stateCount; ++column) {
        int pivot = column;
        for (int row = column + 1; row < stateCount; ++row) {
            if (fabsl(matrix[row][column]) > fabsl(matrix[pivot][column])) {
                pivot = row;
            }
        }
        swap(matrix[column], matrix[pivot]);

        const long double divisor = matrix[column][column];
        for (int j = column; j <= stateCount; ++j) {
            matrix[column][j] /= divisor;
        }
        for (int row = 0; row < stateCount; ++row) {
            if (row == column) continue;
            const long double factor = matrix[row][column];
            for (int j = column; j <= stateCount; ++j) {
                matrix[row][j] -= factor * matrix[column][j];
            }
        }
    }

    vector<long double> expectation(stateCount);
    for (int state = 0; state < stateCount; ++state) {
        expectation[state] = matrix[state][stateCount];
    }

    const auto firstMoves = legalMoves(0, 0);
    long double answer = 1;
    for (const auto& [x, y] : firstMoves) {
        answer += expectation[id[x][y]] /
                  static_cast<long double>(firstMoves.size());
    }

    cout << fixed << setprecision(10) << answer << '\n';
    return 0;
}
