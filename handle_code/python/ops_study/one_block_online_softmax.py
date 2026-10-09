"""用纯 Python 理解单个 CUDA block 的 KV compress。

目标（score、kv 均为 [128, 512]，softmax 沿 128 行）：
    out[i] = sum_j softmax(score[:, i])[j] * kv[j][i]

单 block 映射：启动 <<<1, 512>>>，threadIdx.x = i。
每个线程独占一列 i，顺序读取该列的 128 行；不同列互不依赖。
所以不需要把 [128, 512] 放进 shared memory，也不需要线程间归约。

下面 Python 的外层 for i 只是在模拟 512 个 CUDA 线程。真正的 CUDA
实现中，这 512 个 i 会由同一个 block 的线程并行执行。
"""

import math


def compress_one_block(
    kv: list[list[float]],
    score: list[list[float]],
    *,
    trace_col: int | None = None,
) -> list[float]:
    """在线计算每列的 softmax 加权和，不生成完整的 softmax 矩阵。

    处理完第 j 行后，当前列的三个变量满足：
        m   = max(score[0:j+1, i])
        den = sum_k exp(score[k, i] - m)
        num = sum_k exp(score[k, i] - m) * kv[k, i]
    最终 out[i] = num / den。
    """
    rows = len(kv)
    if rows == 0 or len(score) != rows:
        raise ValueError("kv 和 score 必须有相同的非零行数")
    cols = len(kv[0])
    if cols == 0 or any(len(row) != cols for row in kv + score):
        raise ValueError("kv 和 score 必须是形状相同的非空二维列表")

    out = [0.0] * cols
    if trace_col is not None:
        print("列 i 的递推：row | score | old_m | alpha | weight | new_m | den | num")

    for i in range(cols):  # CUDA 中：i = threadIdx.x，外层循环变成线程并行
        m = -math.inf
        den = 0.0
        num = 0.0

        for j in range(rows):  # 每个线程沿自己的列顺序处理 128 行
            s = score[j][i]
            if s == -math.inf:  # 可选的 mask；这一行权重为 0
                continue

            old_m = m
            new_m = max(m, s)
            alpha = math.exp(m - new_m)  # 旧统计量换到新最大值的尺度
            weight = math.exp(s - new_m)  # 当前元素在新尺度下的权重
            den = alpha * den + weight
            num = alpha * num + weight * kv[j][i]
            m = new_m

            if i == trace_col:
                print(
                    f"{j:5d} | {s:5.1f} | {old_m:5.1f} | {alpha:5.3f} | "
                    f"{weight:6.3f} | {m:5.1f} | {den:5.3f} | {num:6.3f}"
                )

        out[i] = num / den if den else math.nan

    return out


if __name__ == "__main__":
    # 小例子：[3, 2]。第 0 列的 score=[1, 3, 2]，kv=[10, 20, 30]。
    # 把形状换成 [128, 512]，算法和单 block 的线程映射不变。
    kv = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]
    score = [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]]

    result = compress_one_block(kv, score, trace_col=0)
    reference = []
    for i in range(2):
        maximum = max(score[j][i] for j in range(3))
        weights = [math.exp(score[j][i] - maximum) for j in range(3)]
        reference.append(
            sum(weights[j] * kv[j][i] for j in range(3)) / sum(weights)
        )

    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(result, reference))
    print("online 结果:", result)
    print("普通 softmax 结果:", reference)
