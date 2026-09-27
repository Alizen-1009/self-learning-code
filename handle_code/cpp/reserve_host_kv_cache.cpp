/*
中文说明：为 LLM 服务估算不会发生 Host OOM 的 KV Cache 预留量；这是特殊评测
评分题，预留越紧通常分数越高。
解题方法：任何时刻至少有一个未完成请求仍在 GPU Active，因为最后一个 Active
请求的下一 token 总能满足 current+1<=M<=G；因此最多 N-1 个请求被 Offload。
每个 Offloaded 请求尚未完成，长度最多 M-1，安全上界为 (N-1)(M-1)。
复杂度 O(1)。该上界保证安全，但未声称是利用 G 后的最紧上界或最高分答案。

English: Return a Host KV capacity that cannot OOM in the scoring problem.
At least one unfinished request remains Active, so at most N-1 are Offloaded;
each has length at most M-1. Thus (N-1)(M-1) is a safe O(1) bound. It is not
claimed to be the tightest G-aware allocation or a maximum-score solution.
*/
#include <cstdint>

// Safe scoring solution for the special judge.
//
// At least one unfinished request is Active: the last Active request can
// always take its next token because its current length is < M and M <= G.
// Therefore at most N - 1 requests can be Offloaded.  An Offloaded request is
// unfinished, so its current KV length is at most M - 1.
//
// This bound is deliberately trace-independent and cannot Host-OOM.  The task
// is score-based, so a tighter characterization of the hidden traces may score
// higher, but is not required for correctness.
unsigned long long reserveHostKV(int N,
                                 unsigned long long M,
                                 unsigned long long G) {
    (void)G;
    return static_cast<unsigned long long>(N - 1) * (M - 1);
}
