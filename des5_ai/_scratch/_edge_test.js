function esc(s){ return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
let RSCALE="sqrt";
const C_LINE="rgba(255,182,39,.30)",C_DIM="#C9AC83",C_FG="#FFF3E0",GOLD="#FFD65C";
function rScale(p, mode){
  if(mode === "linear") return p;
  if(mode === "log")    return Math.log10(1 + 99 * p) / 2;
  return Math.sqrt(p);
}
function radarSVG(cands, opt){
  opt = opt || {};
  const N = cands.length;
  if(N < 3) return "";                       // 少于 3 边画不成雷达图
  const W = 380, H = 320, cx = W/2, cy = H/2 + 4, R = 82;
  const gid = "rg" + Math.random().toString(36).slice(2, 8);
  const c0 = opt.c0 || "#FFC24A", c1 = opt.c1 || "#FF8A2B";
  const mode = opt.mode || RSCALE;
  const ang = i => (-90 + i * 360 / N) * Math.PI / 180;
  const pt  = (i, r) => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const poly = (rs) => rs.map((r,i) => pt(i,r).map(v => v.toFixed(1)).join(",")).join(" ");

  const RINGS = 4;
  let g = "";
  for(let k = RINGS; k >= 1; k--){
    const r = R * k / RINGS;
    g += `<polygon points="${poly(new Array(N).fill(r))}" fill="none"
           stroke="${C_LINE}" stroke-width="1" ${k===RINGS?'opacity="1"':'opacity=".55"'}/>`;
  }
  // 轴线
  for(let i = 0; i < N; i++){
    const [x,y] = pt(i, R);
    g += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"
           stroke="${C_LINE}" stroke-width="1" opacity=".7"/>`;
  }
  // 刻度标签（只标最外圈，够用且不糊）
  for(let k = 1; k <= RINGS; k++){
    const r = R * k / RINGS, v = Math.pow(k / RINGS, mode === "linear" ? 1 : 2);
    g += `<text x="${(cx+4).toFixed(1)}" y="${(cy-r+3).toFixed(1)}"
           font-size="9" fill="${C_DIM}" opacity=".8">${(v*100).toFixed(0)}%</text>`;
  }
  // 数据多边形
  const rs = cands.map(c => Math.max(1.5, R * rScale(c.p, mode)));
  g += `<polygon points="${poly(rs)}" fill="url(#${gid})" fill-opacity=".30"
         stroke="${c0}" stroke-width="2" stroke-linejoin="round"/>`;
  cands.forEach((c, i) => {
    const [x,y] = pt(i, rs[i]);
    const best = (opt.goldIdx === i);
    g += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${best?4.5:3}"
           fill="${best?GOLD:c0}" stroke="#241705" stroke-width="1"/>`;
  });
  // 轴标签
  cands.forEach((c, i) => {
    const a = ang(i);
    const lr = R + 20;
    let x = cx + lr * Math.cos(a), y = cy + lr * Math.sin(a);
    const cosv = Math.cos(a);
    let anchor = "middle";
    if(cosv > 0.25) anchor = "start";
    else if(cosv < -0.25) anchor = "end";
    y = Math.max(11, Math.min(H - 5, y));
    const name = c.des.length > 9 ? c.des.slice(0,8) + "…" : c.des;
    g += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}"
           font-size="10.5" fill="${opt.goldIdx===i?GOLD:C_FG}"
           font-family="ui-monospace,Consolas,monospace">${esc(name)}</text>`;
    g += `<text x="${x.toFixed(1)}" y="${(y+11).toFixed(1)}" text-anchor="${anchor}"
           font-size="9.5" fill="${C_DIM}"
           font-family="ui-monospace,Consolas,monospace">${(c.p*100).toFixed(1)}%</text>`;
  });

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block;margin:6px auto 2px"
            role="img" aria-label="候选谱师概率雷达图">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${c0}"/><stop offset="100%" stop-color="${c1}"/>
    </linearGradient></defs>${g}</svg>`;
}
let CHART="bar",REDGES=0;
function candChart(cands, opt){
  opt = opt || {};
  if(!cands || !cands.length) return "";

  if(CHART === "radar"){
    /* 轴数 x 可独立配置：
         REDGES = 0  → 跟随候选数（每名候选一条轴）
         REDGES = x  → 只画前 x 名；x 超过候选数时退化为候选数，并如实告知 */
    const want = REDGES > 0 ? Math.min(REDGES, cands.length) : cands.length;
    if(want < 3){
      // 不足 3 条轴构不成雷达图 → 如实回落条形表
      return barTable(cands, opt) +
        `<div class="hint">候选不足 ${REDGES > 0 ? REDGES : 3} 条轴，无法构成雷达图，已显示条形表</div>`;
    }
    const axes = cands.slice(0, want);
    const scaleTxt = {sqrt:"√ 压缩", linear:"线性", log:"对数"}[RSCALE];
    let note;
    if(REDGES === 0){
      note = `雷达图为 <b>${want} 条轴</b>（跟随候选数）`;
    } else if(REDGES < cands.length){
      note = `雷达图为 <b>${want} 条轴</b>（配置 ${REDGES} 边，共 ${cands.length} 个候选，显示前 ${want} 名；下方数值表仍列出全部 ${cands.length} 名）`;
    } else {
      note = `雷达图为 <b>${want} 条轴</b>（配置 ${REDGES} 边，但本次只有 ${cands.length} 个候选，已退化为 ${want} 边）`;
    }
    if(want === 3){
      note += `　<span style="color:var(--warn)">仅 3 边，几何形状信息有限，读数以下方数值表为准</span>`;
    }

    /* 雷达图对极端长尾不敏感：实测 98.5% vs 0.04% 的半径比即使开 √ 仍有 50:1，
       小候选会全挤在中心。所以雷达图下方必附「精确数值表」，
       并把条形按「次高候选」归一化，让 Top2~N 的相对差异也看得见。 */
    const ref = cands[1] ? cands[1].p : (cands[0].p || 1);
    const mini = `<table class="mini">
      <thead><tr><th></th><th>候选谱师</th><th style="width:38%">相对次高候选</th><th class="pct">概率</th></tr></thead>
      <tbody>${cands.map((c,i) => `
        <tr class="${i===0 && opt.gold ? "gold" : ""}">
          <td style="width:24px;color:var(--dim)">${i+1}</td>
          <td class="name" title="${esc(c.des)}">${esc(c.des)}</td>
          <td><div class="bar" style="width:${Math.max(1.5, Math.min(100, c.p/ref*100))}%${opt.miniGrad||""}"></div></td>
          <td class="pct">${(c.p*100).toFixed(2)}%</td>
        </tr>`).join("")}</tbody></table>`;
    return radarSVG(axes, Object.assign({}, opt, {goldIdx: opt.gold ? 0 : -1}))
      + `<div class="hint" style="text-align:center;margin-top:2px">${note}<br>
           半径采用 ${scaleTxt} 刻度（雷达图对长尾分布不敏感，故下方附精确数值表）</div>`
      + mini;
  }
  return barTable(cands, opt);
}
function barTable(cands, opt){
  const maxP = cands[0].p || 1;
  const grad = opt.grad || "";
  return `<table>
    <thead><tr><th></th><th>候选谱师</th><th>概率</th><th class="pct">置信度</th></tr></thead>
    <tbody>${cands.map((c,i) => `
      <tr class="${i===0 && opt.gold ? "gold" : ""}">
        <td style="width:28px;color:var(--dim)">${i+1}</td>
        <td class="name" title="${esc(c.des)}">${esc(c.des)}</td>
        <td style="width:45%"><div class="bar" style="width:${Math.max(2, c.p/maxP*100)}%${grad}"></div></td>
        <td class="pct">${(c.p*100).toFixed(1)}%</td>
      </tr>`).join("")}</tbody></table>`;
}
module.exports={candChart,barTable,radarSVG,setR:n=>{REDGES=n},setC:c=>{CHART=c}};