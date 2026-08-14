(function () {
  const clamp = (value, minimum, maximum) =>
    Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : minimum));

  document.querySelectorAll("[data-layout-explorer]").forEach((explorer) => {
    const fields = {
      m: explorer.querySelector('[data-layout-field="m"]'),
      n: explorer.querySelector('[data-layout-field="n"]'),
      sm: explorer.querySelector('[data-layout-field="sm"]'),
      sn: explorer.querySelector('[data-layout-field="sn"]'),
    };
    const equation = explorer.querySelector("[data-layout-equation]");
    const grid = explorer.querySelector("[data-layout-grid]");
    const analysis = explorer.querySelector("[data-layout-analysis]");
    let selected = [0, 0];

    function readValues() {
      return {
        m: clamp(Number(fields.m.value), 1, 12),
        n: clamp(Number(fields.n.value), 1, 12),
        sm: clamp(Number(fields.sm.value), 0, 128),
        sn: clamp(Number(fields.sn.value), 0, 128),
      };
    }

    function render() {
      const values = readValues();
      Object.entries(values).forEach(([name, value]) => {
        fields[name].value = value;
      });
      selected = [
        Math.min(selected[0], values.m - 1),
        Math.min(selected[1], values.n - 1),
      ];

      const offsets = [];
      const frequency = new Map();
      for (let i = 0; i < values.m; i += 1) {
        for (let j = 0; j < values.n; j += 1) {
          const offset = i * values.sm + j * values.sn;
          offsets.push(offset);
          frequency.set(offset, (frequency.get(offset) || 0) + 1);
        }
      }

      const selectedOffset = selected[0] * values.sm + selected[1] * values.sn;
      equation.textContent =
        `L = (${values.m},${values.n}):(${values.sm},${values.sn})    ` +
        `L(${selected[0]},${selected[1]}) = ${selected[0]}×${values.sm} + ` +
        `${selected[1]}×${values.sn} = ${selectedOffset}`;

      grid.replaceChildren();
      grid.style.gridTemplateColumns = `repeat(${values.n}, minmax(0, 1fr))`;
      for (let i = 0; i < values.m; i += 1) {
        for (let j = 0; j < values.n; j += 1) {
          const offset = i * values.sm + j * values.sn;
          const cell = document.createElement("button");
          cell.type = "button";
          cell.className = "layout-cell";
          if (frequency.get(offset) > 1) cell.classList.add("collision");
          if (selected[0] === i && selected[1] === j) cell.classList.add("selected");
          cell.innerHTML = `<span>(${i},${j})</span><strong>@${offset}</strong>`;
          cell.addEventListener("click", () => {
            selected = [i, j];
            render();
          });
          grid.appendChild(cell);
        }
      }

      const uniqueOffsets = frequency.size;
      const maximumOffset = Math.max(...offsets);
      const cosize = maximumOffset + 1;
      const collisions = values.m * values.n - uniqueOffsets;
      const holes = cosize - uniqueOffsets;
      analysis.innerHTML =
        `<strong>size=${values.m * values.n}</strong> 个逻辑坐标；` +
        `<strong>cosize=${cosize}</strong> 个地址跨度；` +
        (collisions
          ? `<strong>${collisions} 次地址别名</strong>（红线标记）`
          : `<strong>无地址别名</strong>`) +
        `；跨度内 ${holes} 个空洞。`;
    }

    Object.values(fields).forEach((field) => field.addEventListener("input", render));
    explorer.querySelectorAll("[data-layout-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const [m, n] = button.dataset.shape.split(",").map(Number);
        const [sm, sn] = button.dataset.stride.split(",").map(Number);
        Object.assign(fields.m, { value: m });
        Object.assign(fields.n, { value: n });
        Object.assign(fields.sm, { value: sm });
        Object.assign(fields.sn, { value: sn });
        selected = [0, 0];
        render();
      });
    });

    render();
  });
})();
