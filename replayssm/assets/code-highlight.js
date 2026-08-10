"use strict";

// Small dependency-free Python highlighter for static course snippets.
// It builds DOM nodes from textContent, so source snippets remain safe and copyable.
document.addEventListener("DOMContentLoaded", () => {
  const keywords = new Set([
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
    "or", "pass", "raise", "return", "try", "while", "with", "yield",
  ]);
  const literals = new Set(["True", "False", "None"]);
  const builtins = new Set([
    "range", "len", "min", "max", "sum", "zip", "enumerate", "print",
    "float", "int", "bool", "list", "tuple", "dict", "set", "super",
  ]);
  const modules = new Set(["torch", "triton", "tl", "cute", "cutlass"]);

  const append = (fragment, text, className = "") => {
    const node = className ? document.createElement("span") : document.createTextNode(text);
    if (className) {
      node.className = className;
      node.textContent = text;
    }
    fragment.appendChild(node);
  };

  document.querySelectorAll("pre code.language-python").forEach((code) => {
    const source = code.textContent;
    const fragment = document.createDocumentFragment();
    let i = 0;
    let expectFunctionName = false;

    while (i < source.length) {
      const char = source[i];

      if (char === "#") {
        const end = source.indexOf("\n", i);
        const stop = end === -1 ? source.length : end;
        append(fragment, source.slice(i, stop), "tok-comment");
        i = stop;
        continue;
      }

      if (char === "'" || char === '"') {
        const quote = char;
        const triple = source.slice(i, i + 3) === quote.repeat(3);
        let end = i + (triple ? 3 : 1);
        while (end < source.length) {
          if (source[end] === "\\") {
            end += 2;
            continue;
          }
          if (triple && source.slice(end, end + 3) === quote.repeat(3)) {
            end += 3;
            break;
          }
          if (!triple && source[end] === quote) {
            end += 1;
            break;
          }
          end += 1;
        }
        append(fragment, source.slice(i, end), "tok-string");
        i = end;
        continue;
      }

      if (/\d/.test(char) || (char === "." && /\d/.test(source[i + 1] || ""))) {
        const match = source.slice(i).match(/^(?:0[xob][\da-fA-F_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:e[+-]?\d[\d_]*)?)/i);
        append(fragment, match[0], "tok-number");
        i += match[0].length;
        continue;
      }

      if (/[A-Za-z_]/.test(char)) {
        const match = source.slice(i).match(/^[A-Za-z_]\w*/);
        const word = match[0];
        let className = "";
        if (expectFunctionName) {
          className = "tok-function";
          expectFunctionName = false;
        } else if (keywords.has(word)) {
          className = "tok-keyword";
          expectFunctionName = word === "def";
        } else if (literals.has(word)) {
          className = "tok-literal";
        } else if (builtins.has(word)) {
          className = "tok-builtin";
        } else if (modules.has(word)) {
          className = "tok-module";
        }
        append(fragment, word, className);
        i += word.length;
        continue;
      }

      if ("+-*/%=<>!@|&^~".includes(char)) {
        const match = source.slice(i).match(/^(?:\*\*|\/\/|:=|==|!=|<=|>=|->|<<|>>|[-+*/%=<>!@|&^~])/);
        append(fragment, match[0], "tok-operator");
        i += match[0].length;
        continue;
      }

      append(fragment, char);
      i += 1;
    }

    code.replaceChildren(fragment);
    code.classList.add("is-highlighted");
  });
});
