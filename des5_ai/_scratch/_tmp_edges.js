// 验证「雷达边数 x 可配置」的逻辑，含边界情况
const fs = require("fs");
const path = require("path");
const dir = process.argv[2];
const html = fs.readFileSync(path.join(dir, "static", "index.html"), "utf8");

function pick(name) {
  const start = html.indexOf("function " + name);
  if (start < 0) throw new Error("找不到 " + name);
  let i = html.indexOf("{", start), depth = 0, end = i;
  for (; end < html.length; end++) {
    if (html[end] === "{") depth++;
    else if (html[end] === "}") { depth--; if (depth === 0) break; }
  }
  return html.slice(start, end + 1);
}
const src = [
  pick("esc"),
  'let RSCALE="sqrt";',
  'const C_LINE="rgba(255,182,39,.30)",C_DIM="#C9AC83",C_FG="#FFF3E0",GOLD="#FFD65C";',
  pick("rScale"), pick("radarSVG"),
  'let CHART="bar",REDGES=0;',
  pick("candChart"), pick("barTable"),
  "module.exports={candChart,barTable,radarSVG,setR:n=>{REDGES=n},setC:c=>{CHART=c}};",
].join("\n");
const modPath = path.join(dir, "_edge_test.js");
fs.writeFileSync(modPath, src);
const M = require(modPath);

const pad = (s, n) => String(s).padEnd(n);
const mk = n => Array.from({ length: n }, (_, i) => ({
  des: "候选" + (i + 1), p: Math.max(0.0004, 0.9 / (i + 1)),
}));

console.log("=== 边数 x 配置矩阵（候选数 × 配置边数）===");
console.log(pad("候选数", 8) + pad("配置x", 8) + pad("实际轴数", 10) + "SVG?" + pad("", 6) + "回退条形?" + pad("", 8) + "说明");
const cands = mk(10);
for (const total of [3, 4, 6, 8, 10]) {
  for (const x of [0, 3, 5, 8, 10, 12]) {
    M.setC("radar"); M.setR(x);
    const data = mk(total);
    const htmlOut = M.candChart(data, { gold: true });
    const isSvg = htmlOut.includes("<svg");
    const fallback = htmlOut.includes("无法构成雷达图");
    // 数 SVG 里的数据圆点 = 轴数
    const circles = (htmlOut.match(/<circle /g) || []).length;
    const axes = isSvg ? circles : 0;
    let note = "";
    const m = htmlOut.match(/雷达图为 <b>(\d+) 条轴<\/b>[^<]*/);
    if (m) note = m[0].replace(/<[^>]+>/g, "");
    console.log(pad(total, 8) + pad(x === 0 ? "跟随" : x, 8) +
      pad(isSvg ? axes : "-", 10) + pad(isSvg ? "是" : "否", 6) +
      pad(fallback ? "是" : "否", 10) + note);
  }
}

console.log("\n=== 轴数是否严格等于 min(配置x, 候选数)，且 >=3 ===");
let allOk = true;
for (const total of [3, 4, 6, 8, 10, 12]) {
  for (const x of [0, 3, 4, 5, 6, 8, 10, 12]) {
    M.setC("radar"); M.setR(x);
    const data = mk(total);
    const out = M.candChart(data, { gold: true });
    const isSvg = out.includes("<svg");
    const circles = (out.match(/<circle /g) || []).length;
    if (isSvg) {
      const expect = Math.min(x === 0 ? total : x, total);
      if (circles !== expect || expect < 3) { allOk = false; console.log("  ✘ 候选" + total + " x=" + x + " 实得" + circles + " 期望" + expect); }
    } else {
      const expect = Math.min(x === 0 ? total : x, total);
      if (expect >= 3) { allOk = false; console.log("  ✘ 候选" + total + " x=" + x + " 应画 SVG 却回退了"); }
    }
  }
}
console.log("  " + (allOk ? "✔ 全部 48 种组合都符合预期" : "✘ 存在不符"));

console.log("\n=== 条形模式完全不受边数影响（底层数据不变）===");
const base = mk(6);
M.setC("bar");
const r0 = M.candChart(base, { gold: true });
const hashes = [];
for (const x of [0, 3, 6, 10, 12]) {
  M.setR(x);
  hashes.push(M.candChart(base, { gold: true }) === r0);
}
console.log("  切换边数 5 次，条形输出是否恒等: " + (hashes.every(Boolean) ? "✔ 是（边数仅作用于雷达图）" : "✘ 否"));

console.log("\n=== 雷达图数据来源未被裁剪（数值表仍列全部候选）===");
M.setC("radar"); M.setR(4);
const out = M.candChart(mk(10), { gold: true });
const rows = (out.match(/<tr class=/g) || []).length;
const svgCircles = (out.match(/<circle /g) || []).length;
console.log("  配置 4 边 → SVG 轴数 " + svgCircles + "，数值表行数 " + rows + (svgCircles === 4 && rows === 10 ? "  ✔ 轴少但表全" : "  ✘"));
