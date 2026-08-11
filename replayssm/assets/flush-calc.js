/* flush-calc.js — cache 长度 L 的两侧约束算盘（第 4 课）。
 *
 * 用法（复用 ledger.css 的 .calc / .budget-row / .bar / .ledger）：
 *   <div class="calc" data-flush-calc>
 *     <input type="range" data-flush-var="d|n|L|T"> <output data-flush-echo="...">
 *     <span data-flush-out="<key>">  <span class="bar" data-flush-bar="<key>"><i></i></span>
 *   </div>
 *
 * ⚠️ 分工（刻意如此，别合并）：
 *   d、n、L  → 搬运账。按 **非投机稳态（每 forward 一个 token）** 计算，
 *              这样才能和第 1 课的 8dn 基线直接对比。
 *   T        → **只**驱动约束下界，绝不进入搬运账。
 * 理由：投机解码把一次 forward 摊给 T 个 token，那是「投机」的收益（第 3 课），
 * 不是「L」的收益。混进同一个比值会得到 8×–20× 这种把 L 的取舍完全埋掉的假数字。
 *
 * 搬运账（fp32，单 head，每 decode step 的平均值）：
 *   读检查点   4dn                 每步都要
 *   读 buffer  4·h̄·(d+n)          h̄ = flush 周期内 h 的平均值（精确，非估算）
 *   flush 写回 4dn / 周期 token 数  唯一的 state 写回
 *   基线       8dn（summary route 读+写）
 *
 * 周期：每步开始前若 h + 2 > L 则 flush（h→0），否则 h += 1（T=1 的判据）。
 */
(function () {
  "use strict";

  var B = 4; // fp32

  // 非投机稳态：h 从 0 涨到触发 flush，返回周期长度与 h 的平均值
  function cycleT1(L) {
    var steps = 0, sumH = 0, h = 0;
    while (h + 2 <= L && steps < 100000) {
      sumH += h;      // 本步读 buffer 时窗口里有 h 个 token
      h += 1;
      steps += 1;
    }
    return { steps: steps, hbar: steps ? sumH / steps : 0, hmax: steps ? steps - 1 : 0 };
  }

  function compute(d, n, L, T) {
    var state = B * d * n;
    var c = cycleT1(L);
    var viable = c.steps > 0;
    var readState = state;
    var readBuf = B * c.hbar * (d + n);
    var flushWrite = viable ? state / c.steps : Infinity;
    var perStep = viable ? readState + readBuf + flushWrite : Infinity;
    var baseline = 2 * state;
    return {
      viable: viable,
      steps: c.steps,
      hbar: c.hbar,
      hmax: c.hmax,
      readState: readState,
      readBuf: readBuf,
      flushWrite: flushWrite,
      perStep: perStep,
      baseline: baseline,
      ratio: baseline / perStep,
      // 约束（只由 T 决定，与上面的搬运账互不相干）
      lFloor: 2 * T,        // L ≤ 2T ⇒ h 恒为 0，buffer 一次都用不上
      lUseful: 3 * T,       // L ≥ 3T 才可能连做两轮，buffer 才真正非空
      lStar: Math.sqrt(2 * d * n / (d + n)),   // 带宽最优
      lHard: 2 * d * n / (d + n)               // 收益归零处（平均 h ≈ dn/(d+n)）
    };
  }

  function fmtBytes(b) {
    if (!isFinite(b)) return "∞";
    if (b === 0) return "0";
    if (b < 1024) return Math.round(b) + " B";
    var k = b / 1024;
    return (k < 10 ? k.toFixed(1) : Math.round(k)) + " KiB";
  }

  function verdict(m, L, T) {
    var star = Math.round(m.lStar);
    // 先判约束，再判带宽 —— 约束是正确性/可用性问题，优先级更高
    if (L <= m.lFloor) {
      return ["bad", "L = " + L + " ≤ 2T = " + m.lFloor +
        "：h 还是 0 就触发 flush，buffer 一次都用不上，退化成每步写回 state。"];
    }
    if (L < m.lUseful) {
      return ["warn", "L = " + L + " 不到 3T = " + m.lUseful +
        "：一个周期只够一轮投机，窗口里几乎总是空的。带宽最优在 L ≈ " + star + "。"];
    }
    if (m.ratio >= 1.6) {
      return ["ok", "收益 " + m.ratio.toFixed(2) + "×，平均 h = " + m.hbar.toFixed(1) +
        "。L 同时满足下界 3T = " + m.lUseful + " 与带宽最优 ≈ " + star + "。"];
    }
    if (m.ratio > 1.05) {
      return ["warn", "收益只剩 " + m.ratio.toFixed(2) + "×：窗口偏长，读 buffer 正在吃掉省下的写回（平均 h = " +
        m.hbar.toFixed(1) + "）。带宽最优在 L ≈ " + star + "。"];
    }
    return ["bad", "收益 " + m.ratio.toFixed(2) + "×，基本归零：平均 h = " + m.hbar.toFixed(1) +
      " 已到硬上界 dn/(d+n) = " + Math.round(m.lHard / 2) + " 附近。"];
  }

  function wire(root) {
    var vars = ["d", "n", "L", "T"];
    var inputs = {};
    vars.forEach(function (v) {
      inputs[v] = root.querySelector('[data-flush-var="' + v + '"]');
    });

    function set(key, value) {
      var nodes = root.querySelectorAll('[data-flush-out="' + key + '"]');
      for (var i = 0; i < nodes.length; i++) nodes[i].textContent = value;
    }
    function bar(key, frac) {
      var nodes = root.querySelectorAll('[data-flush-bar="' + key + '"]');
      var pct = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + "%";
      for (var i = 0; i < nodes.length; i++) nodes[i].style.setProperty("--w", pct);
    }

    function render() {
      var d = +inputs.d.value, n = +inputs.n.value;
      var L = +inputs.L.value, T = +inputs.T.value;
      var m = compute(d, n, L, T);

      vars.forEach(function (v) {
        var e = root.querySelector('[data-flush-echo="' + v + '"]');
        if (e && inputs[v]) e.value = inputs[v].value;
      });

      set("read-state", fmtBytes(m.readState));
      set("read-buf", fmtBytes(m.readBuf));
      set("flush-write", fmtBytes(m.flushWrite));
      set("per-step", fmtBytes(m.perStep));
      set("baseline", fmtBytes(m.baseline));
      set("ratio", isFinite(m.ratio) ? m.ratio.toFixed(2) + "×" : "—");
      set("hbar", m.viable ? m.hbar.toFixed(1) : "—");
      set("hmax", m.viable ? m.hmax : "—");
      set("cycle", m.viable ? m.steps : 0);
      set("l-floor", m.lFloor);
      set("l-useful", m.lUseful);
      set("l-star", Math.round(m.lStar));
      set("l-hard", Math.round(m.lHard));

      var scale = m.baseline || 1;
      bar("read-state", m.readState / scale);
      bar("read-buf", m.readBuf / scale);
      bar("flush-write", isFinite(m.flushWrite) ? m.flushWrite / scale : 1);

      var v = verdict(m, L, T);
      var box = root.querySelector("[data-flush-verdict]");
      if (box) {
        box.className = "calc-verdict " + v[0];
        box.textContent = v[1];
      }
    }

    vars.forEach(function (v) {
      if (inputs[v]) inputs[v].addEventListener("input", render);
    });
    render();
  }

  function init() {
    var roots = document.querySelectorAll("[data-flush-calc]");
    for (var i = 0; i < roots.length; i++) wire(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
