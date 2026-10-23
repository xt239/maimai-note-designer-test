function esc(s){ return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function rScale(p, mode){
  if(mode === "linear") return p;
  if(mode === "log")    return Math.log10(1 + 99 * p) / 2;
  return Math.sqrt(p);
}
let RSCALE="sqrt";
function radarSVG(cands, opt){
  opt = opt || {};
  const N = cands.length;
  if(N < 3) return "";                       // 少于 3 边画不成雷达图
  const W = 380, H = 320, cx = W/2, cy = H/2 + 4, R = 82;
  const gid = "rg" + Math.random().toString(36).slice(2, 8);
  const c0 = opt.c0 || "#4f9cf9", c1 = opt.c1 || "#79c0ff";
  const mode = opt.mode || RSCALE;
  const ang = i => (-90 + i * 360 / N) * Math.PI / 180;
  const pt  = (i, r) => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const poly = (rs) => rs.map((r,i) => pt(i,r).map(v => v.toFixed(1)).join(",")).join(" ");

  const RINGS = 4;
  let g = "";
  for(let k = RINGS; k >= 1; k--){
    const r = R * k / RINGS;
    g += `<polygon points="${poly(new Array(N).fill(r))}" fill="none"
           stroke="var(--line)" stroke-width="1" ${k===RINGS?'opacity="1"':'opacity=".55"'}/>`;
  }
  // 轴线
  for(let i = 0; i < N; i++){
    const [x,y] = pt(i, R);
    g += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"
           stroke="var(--line)" stroke-width="1" opacity=".7"/>`;
  }
  // 刻度标签（只标最外圈，够用且不糊）
  for(let k = 1; k <= RINGS; k++){
    const r = R * k / RINGS, v = Math.pow(k / RINGS, mode === "linear" ? 1 : 2);
    g += `<text x="${(cx+4).toFixed(1)}" y="${(cy-r+3).toFixed(1)}"
           font-size="9" fill="var(--dim)" opacity=".8">${(v*100).toFixed(0)}%</text>`;
  }
  // 数据多边形
  const rs = cands.map(c => Math.max(1.5, R * rScale(c.p, mode)));
  g += `<polygon points="${poly(rs)}" fill="url(#${gid})" fill-opacity=".30"
         stroke="${c0}" stroke-width="2" stroke-linejoin="round"/>`;
  cands.forEach((c, i) => {
    const [x,y] = pt(i, rs[i]);
    const best = (opt.goldIdx === i);
    g += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${best?4.5:3}"
           fill="${best?"#ffd33d":c0}" stroke="#0d1117" stroke-width="1"/>`;
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
           font-size="10.5" fill="${opt.goldIdx===i?"#ffd33d":"var(--fg)"}"
           font-family="ui-monospace,Consolas,monospace">${esc(name)}</text>`;
    g += `<text x="${x.toFixed(1)}" y="${(y+11).toFixed(1)}" text-anchor="${anchor}"
           font-size="9.5" fill="var(--dim)"
           font-family="ui-monospace,Consolas,monospace">${(c.p*100).toFixed(1)}%</text>`;
  });

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block;margin:6px auto 2px"
            role="img" aria-label="候选谱师概率雷达图">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${c0}"/><stop offset="100%" stop-color="${c1}"/>
    </linearGradient></defs>${g}</svg>`;
}
module.exports={radarSVG,rScale};