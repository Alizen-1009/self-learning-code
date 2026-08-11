/* amdahl-calc.js — kernel 加速 ⇒ 端到端加速的算盘（第 5 课）。
 *
 * 用法（复用 ledger.css 的 .calc / .calc-slider / .budget-row / .calc-verdict）：
 *   <div class="calc" data-amdahl-calc>
 *     <input type="range" data-amdahl-var="f|x"> <output data-amdahl-echo="f|x">
 *     <span class="bar" data-amdahl-bar="other|gdn-before|gdn-after"><i></i></span>
 *     <span data-amdahl-out="<key>">
 *   </div>
 *
 * 账本约定：
 *   f  — GDN decode 占端到端 decode 时间的比例（0–1，滑块用百分数）
 *   x  — 对这一段的局部加速（kernel speedup）
 *   E  = 1 / ((1−f) + f/x)   端到端加速
 *   上限 = 1/(1−f)           x → ∞ 时的极限
 * 数值锚点与 code/verify_replayssm_identities.py 第 6 节一致。
 */
(function () {
  "use strict";

  function compute(f, x) {
    var after = (1 - f) + f / x;
    return {
      other: 1 - f,
      gdnBefore: f,
      gdnAfter: f / x,
      totalAfter: after,
      E: 1 / after,
      ceiling: 1 / (1 - f),
      savedPct: 100 * f * (1 - 1 / x)
    };
  }

  function verdict(m, f, x) {
    var e = m.E.toFixed(3);
    var c = m.ceiling.toFixed(2);
    if (m.ceiling <= 1.15) {
      return ["bad", "f 只有 " + Math.round(100 * f) + "%：kernel 快 " + x.toFixed(2) +
        "× 只换来端到端 " + e + "×。就算 kernel 无限快，上限也只有 " + c +
        "× —— 这就是 MoE 的处境。"];
    }
    if (m.E >= 1.2) {
      return ["ok", "f = " + Math.round(100 * f) + "% 够大：端到端 " + e +
        "×，上限 " + c + "×。这一段值得写 kernel。"];
    }
    return ["warn", "端到端 " + e + "×，上限 " + c +
      "×。收益存在但不厚 —— 先确认 f 的测量，再决定投入。"];
  }

  function wire(root) {
    var vars = ["f", "x"];
    var inputs = {};
    vars.forEach(function (v) {
      inputs[v] = root.querySelector('[data-amdahl-var="' + v + '"]');
    });

    function set(key, value) {
      var nodes = root.querySelectorAll('[data-amdahl-out="' + key + '"]');
      for (var i = 0; i < nodes.length; i++) nodes[i].textContent = value;
    }

    function bar(key, frac) {
      var node = root.querySelector('[data-amdahl-bar="' + key + '"] i');
      if (node) node.style.width = Math.max(0, Math.min(100, 100 * frac)) + "%";
    }

    function render() {
      var f = +inputs.f.value / 100;
      var x = +inputs.x.value;
      var m = compute(f, x);

      var echoF = root.querySelector('[data-amdahl-echo="f"]');
      if (echoF) echoF.value = inputs.f.value + "%";
      var echoX = root.querySelector('[data-amdahl-echo="x"]');
      if (echoX) echoX.value = x.toFixed(2) + "\u00d7";

      bar("other", m.other);
      bar("gdn-before", m.gdnBefore);
      bar("gdn-after", m.gdnAfter);

      set("other", Math.round(100 * m.other) + "%");
      set("gdn-before", Math.round(100 * m.gdnBefore) + "%");
      set("gdn-after", (100 * m.gdnAfter).toFixed(1) + "%");
      set("total-after", (100 * m.totalAfter).toFixed(1) + "%");
      set("e2e", m.E.toFixed(3) + "\u00d7");
      set("ceiling", m.ceiling.toFixed(3) + "\u00d7");
      set("saved", m.savedPct.toFixed(1) + "%");

      var v = verdict(m, f, x);
      var box = root.querySelector("[data-amdahl-verdict]");
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
    var roots = document.querySelectorAll("[data-amdahl-calc]");
    for (var i = 0; i < roots.length; i++) wire(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
