"""Linear Attention 的 prefix cache（前缀缓存），纯 Python，无第三方依赖。

运行：python3 handle_code/python/LLM/LA_prefix_cache.py

用单层、单头、二维的无归一化 gated linear attention 演示：
    S_t = 0.9 * S_{t-1} + k_t outer v_t     # S: [d_k, d_v] = [2, 2]
    y_t = q_t @ S_t                        # y: [d_v]
q/k/v 由 token 人工生成，只用于演示，不是训练过的模型或完整 GDN。

缓存项：完整前缀 token tuple -> 该位置的独立 S 快照。
每 block_size 个 token 保存一次；每个快照包含整个前缀的信息，
不是这个 block 自己的增量。即使较短前缀被驱逐，较长快照也能独立使用。
只有 S_4 就不能直接截取出 S_3；要复用长度 3，须有更早快照并重算。

简化：单模型、串行请求；容量只统计缓存快照，活动请求另持有一份状态。
真实多层模型需保存所有递推层所需状态（可能还有卷积状态等），
缓存 key 也应区分模型/adapter。这里没有 CPU/GPU prefetch 或显存池。
"""

from collections import OrderedDict


def zero_state():
    return [[0.0, 0.0], [0.0, 0.0]]


def copy_state(state):
    return [row[:] for row in state]


def step(state, token):
    """原地推进活动状态；故缓存写入和取出都必须复制。"""
    q = [1.0, token / 10.0]
    k = [token / 10.0, 1.0]
    v = [token / 10.0, -token / 10.0]
    for i in range(2):
        for j in range(2):
            state[i][j] = 0.9 * state[i][j] + k[i] * v[j]
    return [sum(q[i] * state[i][j] for i in range(2)) for j in range(2)]


class LinearPrefixCache:
    def __init__(self, capacity=3, block_size=2):
        if capacity <= 0 or block_size <= 0:
            raise ValueError("capacity 和 block_size 必须大于 0")
        self.capacity = capacity
        self.block_size = block_size
        self.states = OrderedDict()  # 左边最久未使用，右边最近使用

    def lookup(self, tokens):
        """找最长的已缓存前缀；返回命中长度和可独立修改的状态。"""
        end = len(tokens) // self.block_size * self.block_size
        for n in range(end, 0, -self.block_size):
            key = tuple(tokens[:n])
            if key in self.states:
                self.states.move_to_end(key)
                print(f"  命中 {key}，跳过前 {n} 个 token")
                return n, copy_state(self.states[key])
        print("  未命中，从零状态开始")
        return 0, zero_state()

    def put(self, prefix, state):
        """准入策略：每个完整 block 的末尾都缓存；淘汰策略：LRU。"""
        key = tuple(prefix)
        if not key or len(key) % self.block_size:
            raise ValueError("只保存非空、对齐 block 边界的前缀")
        if key not in self.states and len(self.states) == self.capacity:
            old_key, _ = self.states.popitem(last=False)
            print(f"  驱逐 {old_key} 的整个状态快照")
        self.states[key] = copy_state(state)
        self.states.move_to_end(key)
        print(f"  写入 {key} -> S_{len(key)}，固定 2×2 状态")

    def prefill(self, tokens):
        print(f"\n请求 {tokens}")
        matched, state = self.lookup(tokens)
        for i in range(matched, len(tokens)):
            step(state, tokens[i])
            if (i + 1) % self.block_size == 0:
                self.put(tokens[:i + 1], state)
        print(f"  新计算 {len(tokens) - matched} 个 token")
        print(f"  LRU → MRU: {list(self.states)}")
        # 只返回最终递推状态，不生成文本，也不返回已跳过位置的输出。
        return state, matched


def demo():
    cache = LinearPrefixCache(capacity=3, block_size=2)
    requests = [
        ([1, 2, 3, 4], 0),      # 写入 S_2、S_4
        ([1, 2, 5, 6], 2),      # 从共享 S_2 分叉，不污染原快照
        ([1, 2, 3, 4, 7, 8], 4),  # 命中 S_4；写入 S_6 时驱逐 S_2
        ([1, 2, 3, 4, 7, 8], 6),  # 完整命中，不计算新 token
        ([1, 2, 9], 0),         # S_2 已被驱逐；S_4/S_6 不能截短使用
    ]
    for tokens, expected_match in requests:
        state, matched = cache.prefill(tokens)
        # 与完全不使用缓存的逐 token 计算核对，验证分叉与驱逐不影响结果。
        reference = zero_state()
        for token in tokens:
            step(reference, token)
        assert matched == expected_match
        assert state == reference
    print("\n验证通过：所有请求的状态都与从头计算完全一致。")


if __name__ == "__main__":
    demo()
