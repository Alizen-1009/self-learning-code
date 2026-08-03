"use strict";

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-decay-demo]").forEach((demo) => {
    const slider = demo.querySelector("[data-decay-slider]");
    const value = demo.querySelector("[data-decay-value]");
    const matrix = demo.querySelector("[data-decay-matrix]");
    const size = Number(demo.dataset.size || 4);

    const render = () => {
      const alpha = Number(slider.value);
      value.textContent = alpha.toFixed(2);
      const table = document.createElement("table");
      table.className = "matrix-table";
      table.setAttribute("aria-label", `uniform decay ${alpha.toFixed(2)} 的下三角衰减矩阵`);
      for (let i = 0; i < size; i += 1) {
        const row = document.createElement("tr");
        for (let j = 0; j < size; j += 1) {
          const cell = document.createElement("td");
          if (j <= i) {
            cell.textContent = (alpha ** (i - j)).toFixed(3);
            cell.className = i === j ? "diagonal" : "history";
          } else {
            cell.textContent = "0";
            cell.className = "future";
          }
          row.appendChild(cell);
        }
        table.appendChild(row);
      }
      matrix.replaceChildren(table);
    };

    slider.addEventListener("input", render);
    render();
  });
});
