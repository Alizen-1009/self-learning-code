/*
题目：Robot Dance（用一辆手推车重排机器人）

【题意重新表述】
一排位置最初放着：

    A1,A2,...,Am, B1,B2,...,Bn, C1,C2,...,Ck

要求最终变成：

    C1,C2,...,Ck, B1,B2,...,Bn, A1,A2,...,Am

A、B、C 每组内部的相对顺序不能改变。手推车最多装一个机器人，初始为空；所有位置
初始都有机器人。moveRobot(pos) 的含义取决于当前状态：车为空时，把 pos 的机器人
搬上车并留下空位；车非空且 pos 有机器人时，把 pos 的机器人搬到当前空位，pos
成为新空位；车非空且 pos 正好为空时，把车上的机器人放入该空位并清空手推车。
目标是合法完成重排，并尽量减少 moveRobot 调用次数。

【固定例子】
m=2,n=1,k=2 时，初始 [A1,A2,B1,C1,C2]，目标 [C1,C2,B1,A1,A2]。
目标位置 0 需要原位置 3 的 C1，目标位置 3 又需要原位置 0 的 A1，所以 (0,3)
构成一个长度 2 的置换环；同理 (1,4) 是另一个环。每个二元环用 3 次移动完成。

【解题方法：置换环】
对目标位置 p，应该从哪个原位置取机器人：

    p < k       ：m+n+p       （C 组）
    k <= p<k+n ：m+p-k       （B 组）
    p >= k+n   ：p-k-n       （A 组）

这个映射是一张置换。长度为 L 的非平凡环必须先拿走一个机器人制造空位，再移动环中
其余 L-1 个机器人，最后把车上的机器人放回，因此至少 L+1 次；代码也恰好使用 L+1
次，所以移动数对每个环都是最优的。

特殊边界：总位置数可达 1e9，不能无条件申请同样大的 visited。当 m==k 时，整个 B
区间已经在原位，A_i 与 C_i 只是 m 个两两交换，直接各用 3 次完成，不扫描巨大的 n。
当 m!=k 时没有固定位置；题目保证最少移动数不超过 2e7，因此总位置数也不超过该值，
此时才安全地使用 visited。

复杂度：O(实际涉及重排的位置数) 时间；一般情况使用同阶 visited，m==k 时为 O(m)
时间和 O(1) 额外空间。

English: Reorder [A,B,C] into [C,B,A] with one cart while preserving order
inside each group. Map every target position to its original source position and
decompose that permutation into cycles. A nontrivial length-L cycle needs and
uses exactly L+1 API calls. When m==k, B is already fixed, so directly swap each
A_i/C_i pair without scanning a potentially billion-element B segment.
*/
#include <cstdint>
#include <vector>

// The judge provides this function.  Do not print to stdout in robotDance().
void moveRobot(int pos);

// Initial order: A[0..m), B[0..n), C[0..k)
// Target order : C, B, A
//
// Every non-trivial permutation cycle of length L needs exactly L + 1 calls:
// take one robot onto the cart, fill the hole around the cycle, then put the
// cart's robot into the last hole.
void robotDance(int m, int n, int k) {
    const std::int64_t total =
        static_cast<std::int64_t>(m) + n + k;

    // When m == k, every B robot is already at its target position.  Avoid
    // scanning/allocating O(n), because total may be as large as 1e9.
    if (m == k) {
        for (int i = 0; i < m; ++i) {
            const int cPos = m + n + i;
            moveRobot(i);
            moveRobot(cPos);
            moveRobot(cPos);
        }
        return;
    }

    // The statement guarantees that the minimum number of moves is at most
    // 20,000,000.  If m != k there are no fixed positions, hence total itself
    // is at most that limit and this visited array is safe.
    std::vector<unsigned char> visited(static_cast<std::size_t>(total), 0);

    // Which original position contains the robot needed at target position p?
    auto sourceForTarget = [=](int p) -> int {
        if (p < k) return m + n + p;          // C
        if (p < k + n) return m + p - k;      // B
        return p - k - n;                     // A
    };

    for (int start = 0; start < total; ++start) {
        if (visited[start]) continue;

        int pos = start;
        const int firstNext = sourceForTarget(pos);
        if (firstNext == pos) {
            visited[pos] = 1;
            continue;
        }

        moveRobot(start);  // cart <- robot at start, hole = start
        while (true) {
            visited[pos] = 1;
            const int next = sourceForTarget(pos);
            if (next == start) {
                moveRobot(pos);  // cart -> the final hole
                break;
            }
            moveRobot(next);     // next robot -> current hole
            pos = next;
        }
    }
}
