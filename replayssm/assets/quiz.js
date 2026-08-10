"use strict";

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".quiz-form").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const questions = [...form.querySelectorAll(".quiz-question")];
      let correct = 0;
      let complete = true;

      questions.forEach((question) => {
        const selected = question.querySelector("input:checked");
        const answer = question.dataset.answer;
        question.querySelectorAll(".option").forEach((option) => {
          option.classList.remove("correct", "incorrect");
        });
        if (!selected) {
          complete = false;
          return;
        }
        const label = selected.closest(".option");
        if (selected.value === answer) {
          correct += 1;
          label.classList.add("correct");
        } else {
          label.classList.add("incorrect");
        }
      });

      const feedback = form.querySelector(".feedback");
      feedback.classList.remove("ok", "bad");
      if (!complete) {
        feedback.textContent = "请先回答全部问题，再检查答案。";
        feedback.classList.add("bad");
        return;
      }
      const explanations = questions
        .filter((question) => question.querySelector("input:checked")?.value !== question.dataset.answer)
        .map((question) => question.dataset.explanation);
      if (correct === questions.length) {
        feedback.textContent = `${correct}/${questions.length}：全部正确。现在关掉公式，完成最后的检索练习。`;
        feedback.classList.add("ok");
      } else {
        feedback.textContent = `${correct}/${questions.length}：${explanations.join(" ")}`;
        feedback.classList.add("bad");
      }
    });
  });

  document.querySelectorAll("[data-reveal-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.revealTarget);
      const visible = target.classList.toggle("visible");
      button.textContent = visible ? "隐藏参考答案" : "显示参考答案";
    });
  });
});
