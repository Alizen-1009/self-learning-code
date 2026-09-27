/*
中文说明：用蒙特卡洛模拟估计骑士在 4×4 棋盘上从角落出发、随机选择合法走法，
再次到达任意角落所需的平均步数；初始角落不计作结束。
解题方法：独立模拟 1,000,000 次。每一步枚举八种骑士走法，收集合法位置后等概率
随机选择，直到落在角落，最后输出总步数/模拟次数。
复杂度：时间与“模拟次数×每次实际步数”成正比，额外空间 O(1)。结果带随机误差。

English: Monte Carlo estimate of the expected number of random legal knight
moves on a 4x4 board, starting at a corner and stopping upon reaching a corner
again. Run one million independent trials. Runtime is proportional to the total
simulated moves, with O(1) extra space; the output has sampling error.
*/
#include <iostream>
#include <random>
#include <vector>
using namespace std;

int dx[] = { -1, -1, 1, 1, 2, 2, -2, -2 };
int dy[] = { 2, -2, 2, -2, 1, -1, 1, -1 };

bool is_corner(int x, int y) {
    return (x == 1 || x == 4) && (y == 1 || y == 4);
}

int main() {
    mt19937 rng(random_device{}());  // 随机数生成器
    int times = 1000000;            // 模拟次数
    long long total = 0;            // 所有模拟的总步数

    for (int t = 0; t < times; t++) {
        int x = 1, y = 1;
        // 先走一步再判断是否到角落，因此初始位置不算结束。
        do {
            vector<int> path;
            for (int i = 0; i < 8; i++) {
                int tx = x + dx[i];
                int ty = y + dy[i];
                if (tx >= 1 && tx <= 4 && ty >= 1 && ty <= 4) {
                    path.push_back(i);
                }
            }

            // 从合法走法中等概率选一个，不限制走回头路。
            uniform_int_distribution<int> choose(0, (int)path.size() - 1);
            int k = path[choose(rng)];
            x += dx[k];
            y += dy[k];
            total++;
        } while (!is_corner(x, y));
    }

    cout << (double)total / times << '\n';
    return 0;
}
