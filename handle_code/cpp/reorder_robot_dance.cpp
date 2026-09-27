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
