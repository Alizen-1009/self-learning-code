/* ledger-calc.js — decode 单步搬运账的交互算盘。
 *
 * 用法：容器加 data-ledger-calc，内部放
 *   <input type="range" data-calc-var="d|n|h">
 *   <output data-calc-echo="d|n|h">
 *   <span data-calc-out="<key>">
 * key 见下方 compute() 返回的字段名。
 *
 * 账本约定（fp32，单 head，每个 decode step）：
 *   基线 summary route：读 S 4dn + 写 S 4dn        = 8dn 字节
 *   ReplaySSM         ：读 S0 4dn + 读 buffer 4h(d+n)
 * 「读 buffer」这一项是本课程补的，博客的 8dn→4dn 只说 state traffic。
 * flush 的摊薄写回（每步 4dn/L）不计入，见课文旁注。
 */
(function () {
  "use strict";

  var BYTES = 4; // fp32

  function compute(d, n, h) {
    var stateBytes = BYTES * d * n;
    var bufBytes = BYTES * h * (d + n);
    var baseTotal = 2 * stateBytes;
    var newTotal = stateBytes + bufBytes;
    return {
      baseState: baseTotal,
      baseBuffer: 0,
      baseTotal: baseTotal,
      newState: stateBytes,
      newBuffer: bufBytes,
      newTotal: newTotal,
      trafficRatio: newTotal > 0 ? baseTotal / newTotal : Infinity,
      outerOps: d * n,
      innerOps: d + n,
      flopRatio: (d * n) / (d + n),
      breakEven: Math.floor((d * n) / (d + n))
    };
  }

  function fmtBytes(b) {
    if (b === 0) return "0";
    if (b < 1024) return b + " B";
    var k = b / 1024;
    return (k < 10 ? k.toFixed(1) : Math.round(k)) + " KiB";
  }

  function fmtInt(x) {
    return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function fmtRatio(x) {
    if (!isFinite(x)) return "∞";
    return x.toFixed(2) + "×";
  }

  function verdict(m, h) {
    if (m.trafficRatio >= 1.25) {
      return ["ok", "省下 " + fmtBytes(m.baseTotal - m.newTotal) + "/步。缓存窗口还很便宜 —— buffer 只占读取量的 " +
        Math.round(100 * m.newBuffer / m.newTotal) + "%。"];
    }
    if (m.trafficRatio > 1.02) {
      return ["warn", "还在省，但优势在缩小。buffer 已经占到读取量的 " +
        Math.round(100 * m.newBuffer / m.newTotal) + "% —— 窗口再长就要吃掉省下来的那一半。"];
    }
    if (m.trafficRatio > 0.98) {
      return ["warn", "临界点：h = " + h + " 时读 buffer 的字节数正好等于省下的那次写回，净收益归零。"];
    }
    return ["bad", "已经亏了。读 " + h + " 个 token 的 k 与 v 比读写整个 state 还贵 —— 窗口必须短于 " +
      m.breakEven + " 才有意义。"];
  }

  function setText(root, key, value) {
    var nodes = root.querySelectorAll('[data-calc-out="' + key + '"]');
    for (var i = 0; i < nodes.length; i++) nodes[i].textContent = value;
  }

  function setBar(root, key, frac) {
    var nodes = root.querySelectorAll('[data-calc-bar="' + key + '"]');
    var pct = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + "%";
    for (var i = 0; i < nodes.length; i++) nodes[i].style.setProperty("--w", pct);
  }

  function wire(root) {
    var inputs = {};
    var vars = ["d", "n", "h"];
    vars.forEach(function (v) {
      inputs[v] = root.querySelector('[data-calc-var="' + v + '"]');
    });

    function render() {
      var d = +inputs.d.value, n = +inputs.n.value, h = +inputs.h.value;
      var m = compute(d, n, h);

      vars.forEach(function (v) {
        var echo = root.querySelector('[data-calc-echo="' + v + '"]');
        if (echo) echo.value = inputs[v].value;
      });

      setText(root, "base-state", fmtBytes(m.baseState));
      setText(root, "base-buffer", "—");
      setText(root, "base-total", fmtBytes(m.baseTotal));
      setText(root, "new-state", fmtBytes(m.newState));
      setText(root, "new-buffer", fmtBytes(m.newBuffer));
      setText(root, "new-total", fmtBytes(m.newTotal));
      setText(root, "traffic-ratio", fmtRatio(m.trafficRatio));
      setText(root, "outer-ops", fmtInt(m.outerOps));
      setText(root, "inner-ops", fmtInt(m.innerOps));
      setText(root, "flop-ratio", fmtRatio(m.flopRatio));
      setText(root, "break-even", fmtInt(m.breakEven));

      // 全部条以基线总量为满刻度，肉眼可比。
      var scale = m.baseTotal || 1;
      setBar(root, "base-state", m.baseState / scale);
      setBar(root, "new-state", m.newState / scale);
      setBar(root, "new-buffer", m.newBuffer / scale);

      var v = verdict(m, h);
      var box = root.querySelector("[data-calc-verdict]");
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
    var roots = document.querySelectorAll("[data-ledger-calc]");
    for (var i = 0; i < roots.length; i++) wire(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
