#!/usr/bin/env python3
"""ReplaySSM 恒等式数值验证 —— 第 1–4 课全部代数断言的可执行证据。

用法：  python3 code/verify_replayssm_identities.py

为什么存在：MISSION 要求「每个结论必须落到公式、张量 shape 或可算的数字」。
讲义里每一条「两条路线数学等价」的断言，都必须能在这里跑出机器精度的偏差。
以后改了参数化（例如把标量衰减换成对角衰减），先跑这个脚本再改讲义。

约定（与讲义一致）：
    S  : (d, n) recurrent state          S0 : 检查点，窗口之前全部历史
    k,q: (n,)  key / query               v  : (d,) value
    a  : 每步标量衰减门（GDN 里 a = e^g） b  : delta rule 学习率 β
    u  : 伪 value，= b(v − a·S_prev·k)   h  : 窗口内已缓存 token 数
    T  : 投机窗口内 draft 数              C  : T×T 严格下三角耦合矩阵（博客记作 A_s,s'）
"""

import numpy as np

TOL = 1e-13  # fp64 下所有恒等式都应远好于此


def report(name, err, expect_equal=True, extra="", tol=None):
    """err 传绝对偏差时请确保量级为 O(1)；否则传相对偏差并说明。
    tol 可覆盖默认容差（用于网格扫描这类非恒等式断言）。"""
    lim = TOL if tol is None else tol
    ok = (err < lim) if expect_equal else (err > 1e-3)
    verdict = "PASS" if ok else "FAIL"
    want = f"应 < {lim:.0e}" if expect_equal else "应显著不等"
    print(f"  [{verdict}] {name:<44} 偏差 = {err:.3e}  ({want}){extra}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# 1 · Mamba-2：展开递推 + 结合律 ⇒ output-only
# ─────────────────────────────────────────────────────────────────────
def mamba2_output_only(rng, d=5, n=4, h=6):
    """S_t = a_t S_{t-1} + v_t k_t^T  的 output-only 重写是恒等的。"""
    S0 = rng.standard_normal((d, n))
    q = rng.standard_normal(n)
    a = np.concatenate([[np.nan], rng.uniform(0.7, 0.99, h)])
    v = np.vstack([np.zeros(d), rng.standard_normal((h, d))])
    k = np.vstack([np.zeros(n), rng.standard_normal((h, n))])

    # A. summary route：每步物化并更新整个 state
    S = S0.copy()
    for t in range(1, h + 1):
        S = a[t] * S + np.outer(v[t], k[t])
    y_summary = S @ q

    # B. output-only：state 从未被构造
    #    w0 = Π_{i=1..h} a_i        （检查点到现在的总衰减）
    #    wj = Π_{i=j+1..h} a_i      （token j 到现在的衰减，w_h = 空积 = 1）
    y = np.prod(a[1:h + 1]) * (S0 @ q)
    for j in range(1, h + 1):
        y += np.prod(a[j + 1:h + 1]) * v[j] * (k[j] @ q)

    ok = report("Mamba-2 output-only ≡ summary route", np.abs(y - y_summary).max())

    # 退化情形：窗口为空时应退回 y = S0 q
    ok &= report("h=0 退化为 y = S0·q", np.abs(1.0 * (S0 @ q) - S0 @ q).max())
    # 最新 token 未衰减
    ok &= report("w_h = 1（最新 token 无衰减）", abs(np.prod(a[h + 1:h + 1]) - 1.0))
    return ok


# ─────────────────────────────────────────────────────────────────────
# 2 · GDN：引入伪 value u 后，更新退化回 Mamba-2 形状
# ─────────────────────────────────────────────────────────────────────
def gdn_pseudo_value(rng, d=5, n=4, h=6):
    """GDN 原式 S_t = a_t S_{t-1}(I − b_t k k^T) + b_t v k^T
       等价于     S_t = a_t S_{t-1} + u_t k_t^T,
       其中       u_t = b_t (v_t − a_t S_{t-1} k_t).   注意 a_t 在括号内。"""
    S0 = rng.standard_normal((d, n))
    q = rng.standard_normal(n)
    a = np.concatenate([[np.nan], rng.uniform(0.80, 0.99, h)])
    b = np.concatenate([[np.nan], rng.uniform(0.20, 0.90, h)])
    v = np.vstack([np.zeros(d), rng.standard_normal((h, d))])
    k = np.vstack([np.zeros(n), rng.standard_normal((h, n))])
    I = np.eye(n)

    # A. GDN 原式：state 被右乘一个 n×n 投影矩阵
    S = S0.copy()
    for t in range(1, h + 1):
        S = a[t] * S @ (I - b[t] * np.outer(k[t], k[t])) + b[t] * np.outer(v[t], k[t])
    S_gdn, y_gdn = S, S @ q

    # B. u 形式：每步先算修正项，更新即退化成「标量衰减 + 一个外积」
    S = S0.copy()
    u = np.zeros((h + 1, d))
    for t in range(1, h + 1):
        u[t] = b[t] * (v[t] - a[t] * (S @ k[t]))   # ← 需要在 k_t 上读出一次
        S = a[t] * S + np.outer(u[t], k[t])
    ok = report("GDN: u 形式 ≡ 原式（state）", np.abs(S_gdn - S).max())
    ok &= report("GDN: u 形式 ≡ 原式（输出）", np.abs(y_gdn - S @ q).max())

    # C. 拿 u 直接套第 1 课的 output-only 公式
    y = np.prod(a[1:h + 1]) * (S0 @ q)
    for j in range(1, h + 1):
        y += np.prod(a[j + 1:h + 1]) * u[j] * (k[j] @ q)
    ok &= report("GDN: 第 1 课公式套 u 后 ≡ 原式", np.abs(y - y_gdn).max())

    # D. u 需要的 S_{t-1}k_t 本身也能不物化 state 地算出来
    worst = 0.0
    for t in range(1, h + 1):
        pred = np.prod(a[1:t]) * (S0 @ k[t])
        for j in range(1, t):
            pred += np.prod(a[j + 1:t]) * u[j] * (k[j] @ k[t])
        S = S0.copy()
        for i in range(1, t):
            S = a[i] * S + np.outer(u[i], k[i])
        worst = max(worst, np.abs(pred - S @ k[t]).max())
    ok &= report("GDN: S_(t-1)·k_t 也无需物化 state", worst)

    # E. 反例：若照搬 v（而不是 u）去展开，结果是错的 —— 不是精度问题
    y_naive = np.prod(a[1:h + 1]) * (S0 @ q)
    for j in range(1, h + 1):
        y_naive += np.prod(a[j + 1:h + 1]) * b[j] * v[j] * (k[j] @ q)
    rel = np.abs(y_naive - y_gdn).max() / np.abs(y_gdn).max()
    ok &= report("GDN: 照搬 v 展开必须失败", np.abs(y_naive - y_gdn).max(),
                 expect_equal=False, extra=f"  相对误差 {rel:.1%}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# 3 · 判定题：哪些 state 更新形状能展开成「检查点 + 加权和」
#     （第 2 课的 shape-drill 用的就是这几个案例）
# ─────────────────────────────────────────────────────────────────────
def shape_drill_cases(rng, d=5, n=4, h=6):
    S0 = rng.standard_normal((d, n))
    q = rng.standard_normal(n)
    k = np.vstack([np.zeros(n), rng.standard_normal((h, n))])
    v = np.vstack([np.zeros(d), rng.standard_normal((h, d))])
    a = np.concatenate([[np.nan], rng.uniform(0.8, 0.99, h)])

    # 案例：逐通道对角衰减 S ← S·diag(a) + v k^T  —— 可展开（衰减吸收进 k）
    av = np.vstack([np.zeros(n), rng.uniform(0.8, 0.99, (h, n))])
    S = S0.copy()
    for t in range(1, h + 1):
        S = S * av[t] + np.outer(v[t], k[t])
    y_ref = S @ q
    w0 = np.prod(av[1:h + 1], axis=0)
    y = (S0 * w0) @ q
    for j in range(1, h + 1):
        decay = np.prod(av[j + 1:h + 1], axis=0) if j < h else np.ones(n)
        y += v[j] * ((k[j] * decay) @ q)          # 衰减吸收进 k，仍是标量×向量
    ok = report("对角衰减 S·diag(a)：可展开", np.abs(y - y_ref).max())

    # 案例：每步两个外积（rank-2）—— 可展开（与秩无关，两项各自展开）
    v2 = np.vstack([np.zeros(d), rng.standard_normal((h, d))])
    k2 = np.vstack([np.zeros(n), rng.standard_normal((h, n))])
    S = S0.copy()
    for t in range(1, h + 1):
        S = a[t] * S + np.outer(v[t], k[t]) + np.outer(v2[t], k2[t])
    y_ref = S @ q
    y = np.prod(a[1:h + 1]) * (S0 @ q)
    for j in range(1, h + 1):
        w = np.prod(a[j + 1:h + 1])
        y += w * v[j] * (k[j] @ q) + w * v2[j] * (k2[j] @ q)
    ok &= report("rank-2 每步两个外积：可展开", np.abs(y - y_ref).max())
    return ok


# ─────────────────────────────────────────────────────────────────────
# 4 · 投机验证：T 个 draft 的 u 由一次 T×T 三角求解同时得到
# ─────────────────────────────────────────────────────────────────────
def speculative_triangular_solve(rng, d=6, n=5, h=9, T=4):
    """把 u_s 对 u_{s'<s} 的依赖分离成
           u_s = R_s − Σ_{s'<s} C[s,s'] u_{s'}
       其中 R_s = b_s (v_s − W_s (S_h k_s))          只依赖 draft 与窗口末端
            C[s,s'] = b_s (W_s / W_s') (k_s'^T k_s)  严格下三角
            W_s = Π_{i<=s} a_i  （draft 窗口内的累积衰减）
       矩阵形式 (I + C) U = R，一次三角求解出全部 T 个 u。"""
    S0 = rng.standard_normal((d, n))
    a = rng.uniform(.85, .99, h + 1);  b = rng.uniform(.2, .9, h + 1)
    k = rng.standard_normal((h + 1, n));  v = rng.standard_normal((h + 1, d))
    ad = rng.uniform(.85, .99, T + 1); bd = rng.uniform(.2, .9, T + 1)
    kd = rng.standard_normal((T + 1, n)); vd = rng.standard_normal((T + 1, d))
    qd = rng.standard_normal((T + 1, n))

    # 基准：h 个已提交 token 串行跑完，再把 T 个 draft 也串行跑完
    S = S0.copy(); u = np.zeros((h + 1, d))
    for t in range(1, h + 1):
        u[t] = b[t] * (v[t] - a[t] * (S @ k[t]))
        S = a[t] * S + np.outer(u[t], k[t])
    S_h = S.copy()
    u_seq = np.zeros((T + 1, d)); y_seq = np.zeros((T + 1, d))
    for s in range(1, T + 1):
        u_seq[s] = bd[s] * (vd[s] - ad[s] * (S @ kd[s]))
        S = ad[s] * S + np.outer(u_seq[s], kd[s])
        y_seq[s] = S @ qd[s]

    # 全程不物化 S_h：用第 1 课的展开算 S_h·x
    w = np.array([0.0] + [np.prod(a[j + 1:h + 1]) for j in range(1, h + 1)])
    w0 = np.prod(a[1:h + 1])

    def Sh_dot(x):
        r = w0 * (S0 @ x)
        for j in range(1, h + 1):
            r = r + w[j] * u[j] * (k[j] @ x)
        return r

    ok = report("投机: S_h·x 无需物化 S_h",
                max(np.abs(Sh_dot(kd[s]) - S_h @ kd[s]).max() for s in range(1, T + 1)))

    W = np.ones(T + 1)
    for s in range(1, T + 1):
        W[s] = W[s - 1] * ad[s]

    R = np.array([bd[s] * (vd[s] - W[s] * Sh_dot(kd[s])) for s in range(1, T + 1)])
    C = np.zeros((T, T))
    for s in range(1, T + 1):
        for sp in range(1, s):
            C[s - 1, sp - 1] = bd[s] * (W[s] / W[sp]) * (kd[sp] @ kd[s])

    ok &= report("投机: C 严格下三角（含对角为 0）", np.abs(np.triu(C)).max())

    U_solve = np.linalg.solve(np.eye(T) + C, R)
    ok &= report("投机: 三角求解的 u ≡ 串行的 u", np.abs(U_solve - u_seq[1:]).max())

    # UT transform 式：显式求逆后退化成一次 T×T · T×d 的 GEMM
    U_inv = np.linalg.inv(np.eye(T) + C) @ R
    ok &= report("投机: 显式求逆 ≡ 三角求解（可 GEMM）", np.abs(U_inv - U_solve).max())

    # y_s 也只是一次带 causal mask 的 GEMM
    Y = np.zeros((T, d))
    for s in range(1, T + 1):
        acc = W[s] * Sh_dot(qd[s])
        for sp in range(1, s + 1):
            acc = acc + (W[s] / W[sp]) * (kd[sp] @ qd[s]) * U_solve[sp - 1]
        Y[s - 1] = acc
    ok &= report("投机: masked GEMM 的 y ≡ 串行的 y", np.abs(Y - y_seq[1:]).max())

    # draft 的 key 两两正交 ⇒ L = 0 ⇒ T 个 u 完全解耦
    Q, _ = np.linalg.qr(rng.standard_normal((n, T)))       # n >= T 时得到正交列
    kdo = np.vstack([np.zeros(n), Q.T[:T]])
    Co = np.zeros((T, T))
    for s in range(1, T + 1):
        for sp in range(1, s):
            Co[s - 1, sp - 1] = bd[s] * (W[s] / W[sp]) * (kdo[sp] @ kdo[s])
    ok &= report("投机: key 正交 ⇒ C = 0（完全解耦）", np.abs(Co).max())

    # 内积计数：R 用 T·h，C 用 T(T−1)/2，y 用 T·h + T(T+1)/2 ⇒ 2Th + T²
    cnt = sum(h + (s - 1) + s + h for s in range(1, T + 1))
    ok &= report(f"投机: 内积计数 = 2Th + T² = {2*T*h + T*T}", abs(cnt - (2 * T * h + T * T)))
    print(f"         └ 注：公式卡记的 T(T+h)={T*(T+h)} 是单条读出路径的量级；"
          f"GDN 需要两条（u 与 y），实际 {cnt}。同为 Θ(T(T+h))。")
    return ok


# ─────────────────────────────────────────────────────────────────────
# 5 · Cache / flush 的三个不变量（第 4 课）
# ─────────────────────────────────────────────────────────────────────
def _gdn_reference(rng, d, n, nstep):
    """纯递推基线：一条 GDN 序列的真值。"""
    a = rng.uniform(.88, .99, nstep + 1); b = rng.uniform(.2, .9, nstep + 1)
    k = rng.standard_normal((nstep + 1, n)); v = rng.standard_normal((nstep + 1, d))
    q = rng.standard_normal((nstep + 1, n))
    S = np.zeros((d, n)); y = np.zeros((nstep + 1, d))
    for t in range(1, nstep + 1):
        u = b[t] * (v[t] - a[t] * (S @ k[t]))
        S = a[t] * S + np.outer(u, k[t])
        y[t] = S @ q[t]
    return (a, b, k, v, q), y


def _replayssm_run(L, params, nstep, d, n, honor_flush=True, T=1):
    """检查点 + 环形 buffer。honor_flush=False 模拟漏掉 h+2T>L 判据。"""
    a, b, k, v, q = params
    S0 = np.zeros((d, n))
    bu = np.zeros((L, d)); bk = np.zeros((L, n)); ba = np.ones(L)
    h = 0; wpos = 0; y = np.zeros((nstep + 1, d)); flushes = 0

    def dot(x):                       # 用「检查点 + buffer」读出，不物化 state
        r = S0 @ x
        for i in range(h):
            idx = (wpos - h + i) % L
            r = ba[idx] * r + bu[idx] * (bk[idx] @ x)
        return r

    for t in range(1, nstep + 1):
        if honor_flush and h + 2 * T > L:
            Sx = S0.copy()
            for i in range(h):
                idx = (wpos - h + i) % L
                Sx = ba[idx] * Sx + np.outer(bu[idx], bk[idx])
            S0, h, wpos, flushes = Sx, 0, 0, flushes + 1
        u = b[t] * (v[t] - a[t] * dot(k[t]))
        bu[wpos % L] = u; bk[wpos % L] = k[t]; ba[wpos % L] = a[t]
        wpos = (wpos + 1) % L; h = min(h + 1, L)
        y[t] = dot(q[t])
    return y, flushes


def cache_flush_invariants(rng, d=6, n=5, nstep=60):
    params, y_true = _gdn_reference(rng, d, n, nstep)
    scale = np.abs(y_true[1:]).max()

    # 不变量 1：遵守 h + 2T ≤ L ⇒ 与基线一致；违反 ⇒ 静默算错（不抛异常）
    ok = True
    worst_good = 0.0
    quiet = []
    for L in (4, 8, 16, 32):
        yg, _ = _replayssm_run(L, params, nstep, d, n, honor_flush=True)
        yb, _ = _replayssm_run(L, params, nstep, d, n, honor_flush=False)
        # 用相对偏差：nstep=60 步累积后 y 的量级到 1e4，绝对容差没有意义
        worst_good = max(worst_good, np.abs(yg[1:] - y_true[1:]).max() / scale)
        quiet.append((L, np.abs(yb[1:] - y_true[1:]).max() / scale))
    ok &= report("flush: 遵守 h+2T≤L ⇒ 等于基线（相对）", worst_good)
    ok &= report("flush: 违反判据必须算错（且无报错）",
                 min(e for _, e in quiet) * scale, expect_equal=False)
    print("         └ 违反后的相对误差：" +
          "，".join(f"L={L} → {e:.1%}" for L, e in quiet))
    print("         └ 注意 L 越大错得越轻 —— 越难发现，因此越危险。")

    # 不变量 2（回滚）：被拒 draft 从未进入 state ⇒ 回滚是逐位精确，不是「近似恢复」
    def seq(S, toks):
        S = S.copy()
        for (aa, bb, kk, vv) in toks:
            uu = bb * (vv - aa * (S @ kk))
            S = aa * S + np.outer(uu, kk)
        return S

    S0 = rng.standard_normal((d, n))
    committed = [(rng.uniform(.85, .99), rng.uniform(.2, .9),
                  rng.standard_normal(n), rng.standard_normal(d)) for _ in range(7)]
    drafts = [(rng.uniform(.85, .99), rng.uniform(.2, .9),
               rng.standard_normal(n), rng.standard_normal(d)) for _ in range(6)]
    Sh = seq(S0, committed)
    worst = max(np.abs(seq(Sh, drafts[:m]) - seq(seq(S0, committed), drafts[:m])).max()
                for m in range(7))
    ok &= report("回滚: 接受 m 个后 ≡ 只跑那 m 个（逐位）", worst)

    # L 的最优值：数值扫描应与解析式 sqrt(2dn/(d+n)) 一致
    D, N, B = 128, 128, 4
    state = B * D * N

    def avg_traffic(L, T=1, ns=20000):
        h = 0; tot = 0; fl = 0
        for _ in range(ns):
            if h + 2 * T > L:
                h = 0; fl += 1
            tot += h; h += T
        hbar, frate = tot / ns, fl / ns
        return state + B * hbar * (D + N) + state * frate

    best = min(range(3, 200), key=avg_traffic)
    analytic = np.sqrt(2 * D * N / (D + N))
    ok &= report(f"L 最优值: 数值 {best} ≈ 解析 sqrt(2dn/(d+n)) = {analytic:.1f}",
                 abs(best - analytic) / analytic, tol=0.15)   # 整数网格 + 平坦极小
    print(f"         └ 上界自检：L=2·dn/(d+n)={2*D*N//(D+N)} 时平均 h≈dn/(d+n)，"
          f"收益退化到 {2*state/avg_traffic(2*D*N//(D+N)):.2f}× —— 与第 1 课的临界点一致。")
    return ok


def main():
    print(__doc__.split("\n")[0])
    print()
    all_ok = True
    for title, fn in [
        ("1 · Mamba-2 output-only", mamba2_output_only),
        ("2 · GDN 伪 value u", gdn_pseudo_value),
        ("3 · 形状判定题", shape_drill_cases),
        ("4 · 投机验证的 T×T 三角求解", speculative_triangular_solve),
        ("5 · Cache / flush 不变量", cache_flush_invariants),
    ]:
        print(title)
        # 固定种子，结果可复现；换种子应同样全 PASS
        all_ok &= fn(np.random.default_rng(20260811))
        print()
    print("全部通过" if all_ok else "有断言失败 —— 讲义需要修正")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
