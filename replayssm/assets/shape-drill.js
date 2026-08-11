/* shape-drill.js — 形状判定练习：给一条 state 更新式，判断它能不能展开成
 * 「衰减检查点 + 最近输入的加权和」。点击即刻出对错与理由（最紧的反馈环）。
 *
 * 用法：
 *   <div class="drill" data-shape-drill>
 *     <div class="drill-item" data-ok="true" data-why="理由文本">
 *       <code class="drill-form">S ← a·S + u k⊤</code>
 *       <span class="drill-tag">Mamba-2</span>
 *     </div>
 *     ...
 *     <p class="drill-score"></p>
 *   </div>
 *
 * data-ok="true"  → 可以展开
 * data-ok="false" → 不能展开
 * 两个判定按钮由本脚本注入，讲义里不用手写。
 */
(function () {
  "use strict";

  var LABELS = { yes: "可以展开", no: "不能展开" };

  function wire(root) {
    var items = [].slice.call(root.querySelectorAll(".drill-item"));
    var score = root.querySelector(".drill-score");
    var answered = {};

    function updateScore() {
      if (!score) return;
      var done = Object.keys(answered).length;
      var right = 0;
      for (var key in answered) if (answered[key]) right++;
      if (done === 0) {
        score.textContent = "";
        score.className = "drill-score";
        return;
      }
      if (done < items.length) {
        score.textContent = "已答 " + done + " / " + items.length + "，答对 " + right + "。";
        score.className = "drill-score";
        return;
      }
      score.textContent = right === items.length
        ? items.length + " / " + items.length + "：全对。障碍不在复杂度，在「state 被什么东西右乘」—— 你已经抓住了。"
        : right + " / " + items.length + "：把答错那几条的理由再读一遍，重点看它是不是把 state 和 key 搅在了一起。";
      score.className = "drill-score " + (right === items.length ? "ok" : "bad");
    }

    items.forEach(function (item, idx) {
      var truth = item.dataset.ok === "true";
      var why = item.dataset.why || "";

      var actions = document.createElement("div");
      actions.className = "drill-actions";

      var reveal = document.createElement("p");
      reveal.className = "drill-why";

      ["yes", "no"].forEach(function (choice) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "drill-btn";
        btn.textContent = LABELS[choice];
        btn.addEventListener("click", function () {
          var picked = choice === "yes";
          var correct = picked === truth;
          answered[idx] = correct;

          item.classList.remove("is-right", "is-wrong");
          item.classList.add(correct ? "is-right" : "is-wrong");

          [].slice.call(actions.children).forEach(function (b) {
            b.classList.remove("is-picked", "is-truth");
          });
          btn.classList.add("is-picked");
          // 答错时把正确那一侧也标出来
          if (!correct) {
            var truthBtn = actions.children[truth ? 0 : 1];
            if (truthBtn) truthBtn.classList.add("is-truth");
          }

          reveal.textContent = (correct ? "对。" : "不对 —— 正确答案是「" + LABELS[truth ? "yes" : "no"] + "」。") + why;
          reveal.classList.add("visible");
          updateScore();
        });
        actions.appendChild(btn);
      });

      item.appendChild(actions);
      item.appendChild(reveal);
    });

    updateScore();
  }

  function init() {
    var roots = document.querySelectorAll("[data-shape-drill]");
    for (var i = 0; i < roots.length; i++) wire(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
