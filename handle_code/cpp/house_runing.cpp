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
