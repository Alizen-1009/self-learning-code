/* spec-calc.js — 投机验证的开销算盘。
 *
 * 用法（复用 ledger.css 里的 .calc / .calc-slider / .calc-verdict / .ledger）：
 *   <div class="calc" data-spec-calc>
 *     <input type="range" data-spec-var="T|h|L"> <output data-spec-echo="T|h|L">
 *     <span data-spec-out="<key>">
 *   </div>
 *
 * 只有 DOM 里实际出现的 key 会被写入，所以同一个组件可以在不同课里
 * 露出不同的字段（第 3 课只用 T、h；第 4 课可以加上 L 与 flush 判据）。
 *
 * 账本约定（GDN，每个投机轮次，单 head）：
 *   R 的窗口内积      T·h          —— 每个 draft 都要在窗口的 h 个 key 上读出
 *   C 的 draft 间内积  T(T−1)/2     —— 严格下三角
 *   y 的窗口内积      T·h
 *   y 的 draft 间内积  T(T+1)/2     —— 含对角（causal mask）
 *   合计              2Th + T²
 * 公式卡记的 T(T+h) 是「单条读出路径」的量级；GDN 需要两条（u 与 y），
 * 故实际是它的约两倍。两者同为 Θ(T(T+h))。
 */
(function () {
  "use strict";

  function compute(T, h, L) {
    var rWindow = T * h;
    var cDraft = (T * (T - 1)) / 2;
    var yWindow = T * h;
    var yDraft = (T * (T + 1)) / 2;
    var total = rWindow + cDraft + yWindow + yDraft;   // = 2Th + T²
    var perToken = T > 0 ? total / T : 0;              // = 2h + T
    var stepBaseline = 2 * h;                          // 单步 decode 的内积数
    return {
      rWindow: rWindow,
      cDraft: cDraft,
      yWindow: yWindow,
      yDraft: yDraft,
      total: total,
      quadratic: T * T,
      perToken: perToken,
      stepBaseline: stepBaseline,
      overhead: stepBaseline > 0 ? perToken / stepBaseline : Infinity,
      // 顺序执行 T 步需要 T 次「完整读出」；并行后只剩一个 T×T 的耦合
      serialReadouts: T,
      solveOps: T * T,
      needFlush: L != null ? h + 2 * T > L : null
    };
  }

  function fmtInt(x) {
    return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function verdict(m, T, h) {
    // 保留一位小数：T/(2h) 常落在 x.5%，四舍五入会和讲义正文对不上
    var pct = (100 * (m.overhead - 1)).toFixed(1);
    if (m.overhead <= 1.15) {
      return ["ok", "一轮验证 " + T + " 个 token，每 token 只多花 " + pct +
        "% 的内积。二次项 T² = " + fmtInt(m.quadratic) + "，还淹没在 2Th = " + fmtInt(2 * T * h) + " 里。"];
    }
    if (m.overhead <= 1.4) {
      return ["warn", "每 token 多花 " + pct + "% 了。二次项 T² = " + fmtInt(m.quadratic) +
        " 开始咬人 —— 它占总内积的 " + Math.round(100 * m.quadratic / m.total) + "%。"];
    }
    return ["bad", "每 token 多花 " + pct + "%。T² 已占总内积的 " +
      Math.round(100 * m.quadratic / m.total) + "% —— 投机窗口相对缓存窗口太长了，" +
      "接受率必须非常高才划得来。"];
  }

  function wire(root) {
    var vars = ["T", "h", "L"];
    var inputs = {};
    vars.forEach(function (v) {
      inputs[v] = root.querySelector('[data-spec-var="' + v + '"]');
    });

    function set(key, value) {
      var nodes = root.querySelectorAll('[data-spec-out="' + key + '"]');
      for (var i = 0; i < nodes.length; i++) nodes[i].textContent = value;
    }

    function render() {
      var T = +inputs.T.value;
      var h = +inputs.h.value;
      var L = inputs.L ? +inputs.L.value : null;
      var m = compute(T, h, L);

      vars.forEach(function (v) {
        var echo = root.querySelector('[data-spec-echo="' + v + '"]');
        if (echo && inputs[v]) echo.value = inputs[v].value;
      });

      set("r-window", fmtInt(m.rWindow));
      set("c-draft", fmtInt(m.cDraft));
      set("y-window", fmtInt(m.yWindow));
      set("y-draft", fmtInt(m.yDraft));
      set("total", fmtInt(m.total));
      set("quadratic", fmtInt(m.quadratic));
      set("quadratic-share", Math.round(100 * m.quadratic / m.total) + "%");
      set("per-token", m.perToken.toFixed(1));
      set("step-baseline", fmtInt(m.stepBaseline));
      set("overhead", m.overhead.toFixed(2) + "×");
      set("serial-readouts", fmtInt(m.serialReadouts));
      set("solve-ops", fmtInt(m.solveOps));
      if (m.needFlush != null) set("need-flush", m.needFlush ? "需要 flush" : "无需 flush");

      var v = verdict(m, T, h);
      var box = root.querySelector("[data-spec-verdict]");
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
    var roots = document.querySelectorAll("[data-spec-calc]");
    for (var i = 0; i < roots.length; i++) wire(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
