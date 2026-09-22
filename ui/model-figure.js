import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';
import { animate, createTimeline } from 'https://cdn.jsdelivr.net/npm/animejs/+esm';

/* ------------------------------------------------------------------ *
 * The centerpiece: one tall, accurate transformer figure that runs down
 * the middle of the page. It follows the conventions of the original
 * "Attention Is All You Need" architecture figure (embeddings + position,
 * masked multi-head attention, Add & Norm with residual bypasses, a
 * position-wise feed-forward MLP, a stack of N blocks, unembedding and
 * softmax), drawn with D3.
 *
 * Scroll is the clock: a "read head" tracks the viewport and everything
 * above it lights up, so activations visibly trickle down the model as
 * you scroll. Below the first block, agent sprites edit the compact
 * layer stack with their bodies.
 *
 * Motion is driven by anime.js animating plain state objects; a single
 * render loop writes SVG attributes from that state. Nothing animates a
 * CSS transform directly — anime only composes transform components it
 * tracks itself, which silently dropped positions/tilts earlier in this
 * page's history.
 * ------------------------------------------------------------------ */

const svgNode = document.getElementById('model-svg');
const wrap = document.getElementById('figure-wrap');
if (svgNode && wrap) buildFigure();

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildFigure() {
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const VB_W = 1200, VB_H = 2900;
  const TOKENS = ['Top-K', 'makes', 'your', 'training', 'run'];
  const LANES = TOKENS.map((_, i) => 396 + i * 102);
  const LANE_TOP = 342, LANE_END = 2170, RAMP = 70;
  const P = {
    panel: '#111e17', panelLit: '#13271e',
    edge: '#243a2d', edgeLit: '#3d8f71',
    wire: '#22362a', teal: '#5fd9b8', mint: '#b6fbe4', amber: '#f2b84b',
    cellOff: '#17251c', label: '#5f786a', labelLit: '#cfe3d6', text: '#e6efe8',
  };
  const heat = d3.interpolateRgbBasis(['#15241b', '#1d5a46', '#5fd9b8', '#effff8']);
  const mix = (a, b) => d3.interpolateRgb(a, b);
  const rand = mulberry32(11);

  const svg = d3.select(svgNode);
  let vbW = VB_W;
  function fitViewBox() {
    const w = wrap.clientWidth;
    const mode = w >= 1100 ? 'wide' : w >= 700 ? 'medium' : 'narrow';
    // Narrow still keeps the agents' resting spots (x ≈ 260–930) inside the crop.
    const [x, width] = mode === 'wide' ? [0, 1200] : mode === 'medium' ? [150, 900] : [250, 700];
    vbW = width;
    svg.attr('viewBox', `${x} 0 ${width} ${VB_H}`);
    wrap.dataset.mode = mode;
  }
  fitViewBox();
  addEventListener('resize', fitViewBox);

  /* ---------- defs ---------- */
  const defs = svg.append('defs');
  const glow = defs.append('filter').attr('id', 'mf-glow')
    .attr('x', '-60%').attr('y', '-60%').attr('width', '220%').attr('height', '220%');
  glow.append('feGaussianBlur').attr('stdDeviation', 3.5).attr('result', 'b');
  const gm = glow.append('feMerge');
  gm.append('feMergeNode').attr('in', 'b');
  gm.append('feMergeNode').attr('in', 'SourceGraphic');
  const hatch = defs.append('pattern').attr('id', 'mf-hatch').attr('width', 6).attr('height', 6)
    .attr('patternUnits', 'userSpaceOnUse').attr('patternTransform', 'rotate(45)');
  hatch.append('rect').attr('width', 6).attr('height', 6).attr('fill', '#101b14');
  hatch.append('line').attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 6).attr('stroke', '#1b2b21').attr('stroke-width', 2.5);
  const beam = defs.append('linearGradient').attr('id', 'mf-beam');
  [[0, 0], [0.5, 0.8], [1, 0]].forEach(([o, a]) => beam.append('stop').attr('offset', o).attr('stop-color', P.teal).attr('stop-opacity', a));
  const exit = defs.append('linearGradient').attr('id', 'mf-exit').attr('x1', 0).attr('x2', 0).attr('y1', 0).attr('y2', 1);
  [[0, 0.9], [1, 0]].forEach(([o, a]) => exit.append('stop').attr('offset', o).attr('stop-color', P.teal).attr('stop-opacity', a));
  defs.append('marker').attr('id', 'mf-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
    .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,1 L9,5 L0,9 z').attr('fill', '#3f6b57');

  /* ---------- layers (paint order) ---------- */
  const gFrame = svg.append('g');
  const gLanes = svg.append('g');
  const gLanesLit = svg.append('g').attr('filter', 'url(#mf-glow)');
  const gParticles = svg.append('g').attr('filter', 'url(#mf-glow)');
  const gWires = svg.append('g');
  const gPanels = svg.append('g');
  const gWiresIn = svg.append('g');
  const gBody = svg.append('g');
  const gNeurons = svg.append('g');
  const gSignals = svg.append('g').attr('filter', 'url(#mf-glow)');
  const gFx = svg.append('g');
  const gAgents = svg.append('g');

  /* Everything that lights up when the read head passes registers here. */
  const lit = [];
  const on = (y, fn) => lit.push({ y, fn, a: -1 });

  const vlink = (x1, y1, x2, y2) => {
    const m = (y1 + y2) / 2;
    return `M${x1},${y1} C${x1},${m} ${x2},${m} ${x2},${y2}`;
  };
  function caps(x, y, text, anchor = 'start') {
    const t = gBody.append('text').attr('class', 'mf-caps').attr('x', x).attr('y', y)
      .attr('text-anchor', anchor).attr('font-size', 11).attr('fill', P.label).text(text);
    const c = mix(P.label, P.labelLit);
    on(y - 10, (a) => t.attr('fill', c(a)));
    return t;
  }
  function wire(d, litY, parent = gWires) {
    const p = parent.append('path').attr('d', d).attr('fill', 'none').attr('stroke', P.wire).attr('stroke-width', 1.2);
    const c = mix(P.wire, P.teal);
    on(litY, (a) => p.attr('stroke', c(a)).attr('stroke-opacity', 0.5 + 0.5 * a));
    return p;
  }
  function box(x, y, w, h, r, litY, parent = gBody) {
    const rect = parent.append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', h).attr('rx', r)
      .attr('fill', P.panel).attr('stroke', P.edge).attr('stroke-width', 1.2);
    const f = mix(P.panel, P.panelLit), s = mix(P.edge, P.edgeLit);
    on(litY, (a) => rect.attr('fill', f(a)).attr('stroke', s(a)));
    return rect;
  }
  function mono(x, y, text, size = 12, anchor = 'middle', litY = y) {
    const t = gBody.append('text').attr('class', 'mf-mono').attr('x', x).attr('y', y)
      .attr('text-anchor', anchor).attr('font-size', size).attr('fill', P.label).text(text);
    const c = mix(P.label, P.labelLit);
    on(litY, (a) => t.attr('fill', c(a)));
    return t;
  }
  function norm(y, label = 'LayerNorm') {
    const rect = box(352, y, 496, 16, 8, y);
    const text = mono(838, y + 11.5, label, 10, 'end', y);
    return { rect, text };
  }
  function adds(y) {
    LANES.forEach((x, i) => {
      const c = gBody.append('circle').attr('cx', x).attr('cy', y).attr('r', 10)
        .attr('fill', P.panel).attr('stroke', P.edge).attr('stroke-width', 1.2);
      const p = gBody.append('path').attr('d', `M${x - 5},${y} H${x + 5} M${x},${y - 5} V${y + 5}`)
        .attr('stroke', P.label).attr('stroke-width', 1.4);
      const s = mix(P.edge, P.teal), pc = mix(P.label, P.mint);
      on(y - 10 + i * 4, (a) => { c.attr('stroke', s(a)); p.attr('stroke', pc(a)); });
    });
  }

  /* ---------- tokens ---------- */
  const TOK_Y = 92;
  caps(352, 56, 'Input tokens');
  TOKENS.forEach((t, i) => {
    const x = LANES[i];
    const r = gBody.append('rect').attr('x', x - 45).attr('y', TOK_Y - 20).attr('width', 90).attr('height', 40).attr('rx', 10)
      .attr('fill', P.panel).attr('stroke', P.edge).attr('stroke-width', 1.3);
    const tx = gBody.append('text').attr('x', x).attr('y', TOK_Y).attr('dy', '0.36em').attr('text-anchor', 'middle')
      .attr('font-size', 16).attr('fill', P.label).text(t);
    const f = mix(P.panel, '#153a2c'), s = mix(P.edge, P.teal), c = mix(P.label, P.text);
    on(TOK_Y - 30 + i * 10, (a) => {
      r.attr('fill', f(a)).attr('stroke', s(a)).attr('filter', a > 0.6 ? 'url(#mf-glow)' : null);
      tx.attr('fill', c(a));
    });
    wire(`M${x},${TOK_Y + 20} V158`, TOK_Y + 10 + i * 10);
  });

  /* ---------- embeddings + positional encoding ---------- */
  const EMB_Y = 162, EMB_P = 17;
  gBody.append('text').attr('class', 'mf-caps').attr('font-size', 10).attr('fill', P.label)
    .attr('transform', `translate(366 ${EMB_Y + 68}) rotate(-90)`).attr('text-anchor', 'middle').text('embedding');
  LANES.forEach((x, i) => {
    for (let j = 0; j < 8; j++) {
      const v = rand() * 2 - 1;
      const target = v >= 0 ? heat(0.35 + 0.6 * v) : mix('#15241b', '#e0a44a')(Math.min(1, -v * 0.9));
      const cell = gBody.append('rect').attr('x', x - 13).attr('y', EMB_Y + j * EMB_P).attr('width', 26).attr('height', 15)
        .attr('rx', 3).attr('fill', P.cellOff);
      const c = mix(P.cellOff, target);
      on(EMB_Y + j * EMB_P + i * 3, (a) => cell.attr('fill', c(a)));
    }
    wire(`M${x},${EMB_Y + 8 * EMB_P} V322`, EMB_Y + 8 * EMB_P);
  });
  const PE_Y = 332;
  gWires.append('line').attr('x1', 836).attr('x2', LANES[0]).attr('y1', PE_Y).attr('y2', PE_Y)
    .attr('stroke', '#2c4637').attr('stroke-dasharray', '3 4');
  box(836, PE_Y - 14, 44, 28, 7, PE_Y);
  const sine = gBody.append('path').attr('fill', 'none').attr('stroke', P.label).attr('stroke-width', 1.5)
    .attr('d', d3.line()(d3.range(0, 1.001, 0.05).map((t) => [840 + t * 36, PE_Y - 7 * Math.sin(t * Math.PI * 3)])));
  { const c = mix(P.label, P.teal); on(PE_Y, (a) => sine.attr('stroke', c(a))); }
  caps(858, 306, 'position', 'middle');
  adds(PE_Y);

  /* ---------- residual stream: one lane per token ---------- */
  const laneLit = LANES.map((x) => {
    gLanes.append('line').attr('x1', x).attr('x2', x).attr('y1', LANE_TOP).attr('y2', LANE_END)
      .attr('stroke', P.wire).attr('stroke-width', 1.6);
    return gLanesLit.append('line').attr('x1', x).attr('x2', x).attr('y1', LANE_TOP).attr('y2', LANE_TOP)
      .attr('stroke', P.teal).attr('stroke-width', 2).attr('stroke-opacity', 0.75);
  });

  /* ---------- transformer block 1, in detail ---------- */
  const blockFrame = gFrame.append('rect').attr('x', 322).attr('y', 392).attr('width', 556).attr('height', 1140).attr('rx', 24)
    .attr('fill', 'rgba(95,217,184,0.012)').attr('stroke', P.edge).attr('stroke-dasharray', '5 7').attr('stroke-width', 1.2);
  { const s = mix(P.edge, '#3a7a62'); on(392, (a) => blockFrame.attr('stroke', s(a))); }
  caps(342, 414, 'Transformer block · layer 1');
  caps(858, 414, '× 12', 'end');
  norm(425);
  wire(`M${LANES[0]},410 H342 Q332,410 332,420 V970 Q332,980 342,980 H384`, 420).attr('marker-end', 'url(#mf-arrow)');

  // Masked multi-head attention
  const ATT_TOP = 456, ATT_BOT = 940;
  box(340, ATT_TOP, 520, ATT_BOT - ATT_TOP, 18, ATT_TOP, gPanels);
  caps(360, ATT_TOP + 24, 'Masked multi-head attention');
  const QKV = [['Q', 440], ['K', 600], ['V', 760]];
  QKV.forEach(([name, cx], n) => {
    const y = 494;
    LANES.forEach((lx, i) => wire(vlink(lx, 441, cx, y), 446 + i * 2 + n * 3, gWiresIn));
    box(cx - 58, y, 116, 44, 9, y + n * 6);
    gBody.append('text').attr('x', cx - 40).attr('y', y + 29).attr('text-anchor', 'middle').attr('font-size', 19)
      .attr('font-style', 'italic').attr('font-family', "'Source Serif 4',Georgia,serif").attr('fill', P.labelLit).text(name);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 6; c++) {
        const cell = gBody.append('rect').attr('x', cx - 22 + c * 11).attr('y', y + 8 + r * 10).attr('width', 9).attr('height', 8)
          .attr('rx', 1.5).attr('fill', P.cellOff);
        const col = mix(P.cellOff, heat(0.2 + rand() * 0.75));
        on(y + r * 6 + c * 3 + n * 6, (a) => cell.attr('fill', col(a)));
      }
    }
  });

  const W = [[1], [0.7, 0.3], [0.2, 0.3, 0.5], [0.25, 0.15, 0.35, 0.25], [0.3, 0.1, 0.05, 0.45, 0.1]];
  const MAT_X = 420, MAT_Y = 590, CELL = 38, PITCH = 42;
  const KX = 690, QX = 818;
  const rowY = (q) => MAT_Y + q * PITCH;
  const CARD = { x: MAT_X - 7, y: MAT_Y - 7, w: 5 * PITCH + 10, h: 5 * PITCH + 10 };
  wire(vlink(440, 538, 400, MAT_Y - 4), MAT_Y - 20, gWiresIn);
  wire(vlink(600, 538, MAT_X + 103, CARD.y - 30), MAT_Y - 30, gWiresIn);
  wire(vlink(760, 538, (KX + QX) / 2, MAT_Y - 2), MAT_Y - 10, gWiresIn);

  // Other heads stacked behind as cards whose top/right edges peek out:
  // this is one of 12 heads, not the only one.
  [3, 2, 1].forEach((h) => {
    const g = gBody.append('g').attr('opacity', 0);
    g.append('rect').attr('x', CARD.x + h * 10).attr('y', CARD.y - h * 10).attr('width', CARD.w).attr('height', CARD.h)
      .attr('rx', 9).attr('fill', '#12221a').attr('stroke', '#2c5343');
    const hr = mulberry32(100 + h);
    for (let q = 0; q < 5; q++) {
      const raw = d3.range(q + 1).map(() => hr() + 0.05);
      const sum = d3.sum(raw);
      for (let k = 0; k <= q; k++) {
        g.append('rect').attr('x', MAT_X + h * 10 + k * PITCH).attr('y', MAT_Y - h * 10 + q * PITCH)
          .attr('width', CELL).attr('height', CELL).attr('rx', 4).attr('fill', heat(raw[k] / sum)).attr('opacity', 0.5);
      }
    }
    on(MAT_Y - 30 + (3 - h) * 16, (a) => g.attr('opacity', a * (1 - h * 0.22)));
  });
  box(CARD.x, CARD.y, CARD.w, CARD.h, 9, MAT_Y);
  for (let q = 0; q < 5; q++) {
    const lab = gBody.append('text').attr('x', MAT_X - 10).attr('y', rowY(q) + 23).attr('text-anchor', 'end')
      .attr('font-size', 12).attr('fill', P.label).text(TOKENS[q]);
    const lc = mix(P.label, P.labelLit);
    on(rowY(q), (a) => lab.attr('fill', lc(a)));
    for (let k = 0; k < 5; k++) {
      const x = MAT_X + k * PITCH, y = rowY(q);
      if (k > q) {
        gBody.append('rect').attr('x', x).attr('y', y).attr('width', CELL).attr('height', CELL).attr('rx', 4)
          .attr('fill', 'url(#mf-hatch)').attr('stroke', '#18261d');
        continue;
      }
      const w = W[q][k];
      const cell = gBody.append('rect').attr('x', x).attr('y', y).attr('width', CELL).attr('height', CELL).attr('rx', 4)
        .attr('fill', P.cellOff);
      const val = gBody.append('text').attr('class', 'mf-mono').attr('x', x + CELL / 2).attr('y', y + CELL / 2 + 4)
        .attr('text-anchor', 'middle').attr('font-size', 10).attr('opacity', 0)
        .attr('fill', w > 0.4 ? '#0f1a14' : '#cfe3d6').text(w.toFixed(2).replace(/^0/, ''));
      const c = mix(P.cellOff, heat(w));
      on(y + k * 7, (a) => { cell.attr('fill', c(a)); val.attr('opacity', a); });
    }
  }
  const rowHi = gBody.append('rect').attr('x', MAT_X - 5).attr('y', MAT_Y - 5).attr('width', 5 * PITCH + 6)
    .attr('height', CELL + 10).attr('rx', 8).attr('fill', 'none').attr('stroke', P.amber).attr('stroke-width', 1.6).attr('opacity', 0);
  mono(MAT_X + 103, 830, 'softmax(QKᵀ / √d) · V', 12);

  // Arc view of the active query row: the same weights, read as "who looks at whom".
  caps((KX + QX) / 2, MAT_Y + 5 * PITCH + 12, 'head 1 of 12', 'middle');
  const keyDots = TOKENS.map((t, k) => {
    const y = rowY(k) + 19;
    const txt = gBody.append('text').attr('x', KX - 10).attr('y', y + 4).attr('text-anchor', 'end').attr('font-size', 11)
      .attr('fill', P.label).text(t);
    const dot = gBody.append('circle').attr('cx', KX).attr('cy', y).attr('r', 5).attr('fill', '#1b2e23').attr('stroke', P.edge);
    const tc = mix(P.label, P.labelLit), dc = mix('#1b2e23', P.teal);
    on(y, (a) => { txt.attr('fill', tc(a)); dot.attr('fill', dc(a)); });
    return dot;
  });
  const queryDots = TOKENS.map((_, q) => gBody.append('circle').attr('cx', QX).attr('cy', rowY(q) + 19).attr('r', 5)
    .attr('fill', '#1b2e23').attr('stroke', P.edge));
  const arcs = TOKENS.map(() => gBody.append('path').attr('fill', 'none').attr('stroke', P.mint)
    .attr('stroke-linecap', 'round').style('opacity', 0).style('transition', 'stroke-width .35s ease, opacity .35s ease'));
  const activeLabel = gBody.append('text').attr('x', QX + 11).attr('font-size', 11).attr('fill', P.amber).attr('opacity', 0);

  // Output projection back into the residual stream
  wire(vlink(MAT_X + 103, MAT_Y + 5 * PITCH, 600, 872), 800, gWiresIn);
  wire(vlink((KX + QX) / 2, MAT_Y + 5 * PITCH + 20, 600, 872), 800, gWiresIn);
  box(530, 872, 140, 36, 9, 872);
  mono(600, 895, 'Concat · W_O', 12);
  LANES.forEach((lx, i) => wire(vlink(600, 908, lx, 970), 912 + i * 3, gWiresIn));
  adds(980);

  // Feed-forward MLP: every connection drawn, 4x-wide hidden layer
  norm(1004);
  wire(`M${LANES[4]},990 H858 Q868,990 868,1000 V1494 Q868,1504 858,1504 H816`, 1000).attr('marker-end', 'url(#mf-arrow)');
  const MLP_TOP = 1040, MLP_BOT = 1470;
  box(340, MLP_TOP, 520, MLP_BOT - MLP_TOP, 18, MLP_TOP, gPanels);
  caps(360, MLP_TOP + 24, 'Feed-forward · MLP');
  const INP_Y = 1104, HID_Y = 1262, OUT_Y = 1420;
  const inp = d3.range(6).map((i) => ({ x: 600 + (i - 2.5) * 80, y: INP_Y, v: 0.35 + rand() * 0.65 }));
  const hid = d3.range(12).map((j) => ({ x: 600 + (j - 5.5) * 42, y: HID_Y, v: rand() < 0.3 ? 0.06 : 0.3 + rand() * 0.7 }));
  const out = d3.range(6).map((i) => ({ x: 600 + (i - 2.5) * 80, y: OUT_Y, v: 0.3 + rand() * 0.7 }));
  caps(846, HID_Y - 26, '4× wider · GELU', 'end');
  inp.forEach((n, i) => wire(`M${n.x},1020 V${INP_Y - 12}`, 1020 + i * 3, gWiresIn));
  const connect = (A, B) => A.forEach((a) => B.forEach((b) => {
    const w = rand() * 2 - 1;
    const l = gWiresIn.append('line').attr('x1', a.x).attr('y1', a.y).attr('x2', b.x).attr('y2', b.y)
      .attr('stroke', P.wire).attr('stroke-width', 0.8).attr('stroke-opacity', 0.7);
    const col = mix(P.wire, w > 0 ? P.teal : P.amber);
    on(a.y + (b.y - a.y) * rand() * 0.85, (x) => l.attr('stroke', col(x))
      .attr('stroke-opacity', 0.35 + 0.5 * x * Math.abs(w)).attr('stroke-width', 0.8 + 0.9 * x * Math.abs(w)));
  }));
  connect(inp, hid);
  connect(hid, out);
  const neuron = (n, r, litY) => {
    const c = gNeurons.append('circle').attr('cx', n.x).attr('cy', n.y).attr('r', r)
      .attr('fill', '#132219').attr('stroke', P.edge).attr('stroke-width', 1.3);
    const f = mix('#132219', heat(0.2 + 0.8 * n.v)), s = mix(P.edge, P.teal);
    on(litY, (a) => c.attr('fill', f(a)).attr('stroke', s(a)).attr('filter', a > 0.5 && n.v > 0.5 ? 'url(#mf-glow)' : null));
  };
  inp.forEach((n, i) => neuron(n, 12, INP_Y - 12 + i * 4));
  hid.forEach((n, j) => neuron(n, 10, HID_Y - 10 + j * 3));
  out.forEach((n, i) => neuron(n, 12, OUT_Y - 12 + i * 4));
  out.forEach((n, i) => wire(vlink(n.x, OUT_Y + 12, LANES[Math.round(i * 4 / 5)], 1494), OUT_Y + 14 + i * 3, gWiresIn));
  adds(1504);

  /* ---------- layers 2..12, compact: the agents' workbench ---------- */
  caps(342, 1562, 'Layers 2 – 12');
  const blocks = d3.range(5).map((i) => makeCompact(1580 + i * 80, `L${i + 2}`));
  const tail = { dy: 0 };
  const gTail = gBody.append('g');
  gTail.append('text').attr('x', 600).attr('y', 1994).attr('text-anchor', 'middle').attr('font-size', 20).attr('fill', P.label).text('⋮');
  gTail.append('rect').attr('x', 380).attr('y', 2010).attr('width', 440).attr('height', 44).attr('rx', 12)
    .attr('fill', P.panel).attr('stroke', P.edge).attr('opacity', 0.55);
  gTail.append('text').attr('class', 'mf-mono').attr('x', 394).attr('y', 2037).attr('font-size', 12).attr('fill', P.label).text('L12');

  function miniMatrix(parent, x, y, seed) {
    const r = mulberry32(seed);
    const cells = [];
    for (let q = 0; q < 5; q++) {
      for (let k = 0; k < 5; k++) {
        const cell = parent.append('rect').attr('x', x + k * 9.6).attr('y', y + q * 9.6).attr('width', 8).attr('height', 8)
          .attr('rx', 1.5).attr('fill', k <= q ? P.cellOff : '#121d16');
        if (k <= q) cells.push({ cell, c: mix(P.cellOff, heat(0.25 + r() * 0.7)) });
      }
    }
    return cells;
  }
  function makeCompact(y0, name) {
    const s = { dx: 0, dy: 0, o: 1, rot: 0, thin: 0, wide: 0 };
    const g = gBody.append('g');
    const rect = g.append('rect').attr('y', y0).attr('height', 60).attr('rx', 12)
      .attr('fill', P.panel).attr('stroke', P.edge).attr('stroke-width', 1.2);
    const label = g.append('text').attr('class', 'mf-mono').attr('y', y0 + 34).attr('font-size', 12).attr('fill', P.label).text(name);
    const extra = g.append('g').attr('opacity', 0);
    const cells = [...miniMatrix(g, 432, y0 + 6, y0), ...miniMatrix(extra, 376, y0 + 6, y0 + 1)];
    g.append('text').attr('class', 'mf-mono').attr('x', 494).attr('y', y0 + 34).attr('font-size', 10).attr('fill', P.label).text('attn');
    g.append('text').attr('class', 'mf-mono').attr('x', 604).attr('y', y0 + 34).attr('font-size', 10).attr('fill', P.label).text('mlp');
    const gLines = g.append('g'), gDots = g.append('g');
    const mk = (n, y) => d3.range(n).map(() => ({ x: 0, y, el: gDots.append('circle').attr('cy', y).attr('r', 3.2).attr('fill', '#1b2e23') }));
    const ins = mk(4, y0 + 12), hids = mk(8, y0 + 30), outs = mk(4, y0 + 48);
    const gone = new Set([1, 4, 6]);
    const lines = [];
    const link = (A, B) => A.forEach((a, i) => B.forEach((b, j) => lines.push({
      a, b, fades: (A === hids && gone.has(i)) || (B === hids && gone.has(j)),
      el: gLines.append('line').attr('y1', a.y).attr('y2', b.y).attr('stroke', P.wire).attr('stroke-width', 0.6),
    })));
    link(ins, hids);
    link(hids, outs);
    const rc = mix(P.edge, P.edgeLit), fc = mix(P.panel, P.panelLit), lc = mix(P.label, P.labelLit),
      dc = mix('#1b2e23', P.teal), wc = mix(P.wire, '#2f6e57');
    on(y0 + 10, (a) => {
      rect.attr('stroke', rc(a)).attr('fill', fc(a));
      label.attr('fill', lc(a));
      cells.forEach(({ cell, c }) => cell.attr('fill', c(a)));
      [...ins, ...hids, ...outs].forEach((d) => d.el.attr('fill', dc(a)));
      lines.forEach((l) => l.el.attr('stroke', wc(a)));
    });
    return { s, g, y0, rect, label, extra, ins, hids, outs, gone, lines };
  }
  function renderBlock(b) {
    const { s, y0 } = b;
    const left = 380 - 54 * s.wide, right = 820 - 74 * s.thin;
    b.g.attr('transform', `translate(${s.dx} ${s.dy}) rotate(${s.rot} 600 ${y0 + 30})`).attr('opacity', s.o);
    b.rect.attr('x', left).attr('width', right - left);
    b.label.attr('x', left + 14);
    b.extra.attr('opacity', s.wide);
    const mx = 700 - 38 * s.thin, hs = 13 * (1 - 0.5 * s.thin), os = 20 * (1 - 0.3 * s.thin);
    b.ins.forEach((d, i) => { d.x = mx + (i - 1.5) * os; d.el.attr('cx', d.x); });
    b.outs.forEach((d, i) => { d.x = mx + (i - 1.5) * os; d.el.attr('cx', d.x); });
    b.hids.forEach((d, j) => { d.x = mx + (j - 3.5) * hs; d.el.attr('cx', d.x).attr('opacity', b.gone.has(j) ? 1 - s.thin : 1); });
    b.lines.forEach((l) => l.el.attr('x1', l.a.x).attr('x2', l.b.x).attr('opacity', l.fades ? 1 - s.thin : 1));
  }
  const renderBlocks = () => { blocks.forEach(renderBlock); gTail.attr('transform', `translate(0 ${tail.dy})`); };
  renderBlocks();

  /* ---------- final norm (the kernel target), unembedding, softmax ---------- */
  const finalNorm = norm(2110);
  const kernel = { v: 0, a: 0 };
  const fnFill = mix(P.panel, P.panelLit), fnEdge = mix(P.edge, P.edgeLit), fnText = mix(P.label, P.labelLit);
  function renderKernel() {
    const { v, a } = kernel;
    finalNorm.rect.attr('fill', mix(fnFill(a), '#3a2c12')(v)).attr('stroke', mix(fnEdge(a), P.amber)(v))
      .attr('filter', v > 0.5 ? 'url(#mf-glow)' : null);
    finalNorm.text.text(v > 0.5 ? 'RMSNorm · fused Triton kernel' : 'LayerNorm').attr('fill', mix(fnText(a), P.amber)(v));
  }
  on(2110, (a) => { kernel.a = a; renderKernel(); });

  box(380, LANE_END, 440, 42, 10, LANE_END);
  mono(600, LANE_END + 26, 'W_U · unembed to 50,304 tokens', 12);
  caps(352, 2246, 'Next-token probabilities');
  const CANDS = [['faster', 0.61], ['better', 0.17], ['longer', 0.09], ['slower', 0.05], ['cleaner', 0.03]];
  CANDS.forEach(([t, p], i) => {
    const y = 2268 + i * 32;
    const txt = gBody.append('text').attr('x', 520).attr('y', y + 13).attr('text-anchor', 'end').attr('font-size', 14)
      .attr('fill', P.label).text(t);
    gBody.append('rect').attr('x', 532).attr('y', y).attr('width', 300).attr('height', 16).attr('rx', 4).attr('fill', '#142219');
    const bar = gBody.append('rect').attr('x', 532).attr('y', y).attr('width', 0).attr('height', 16).attr('rx', 4)
      .attr('fill', i === 0 ? P.teal : '#2f6e57').attr('filter', i === 0 ? 'url(#mf-glow)' : null);
    const val = gBody.append('text').attr('class', 'mf-mono').attr('x', 840).attr('y', y + 12.5).attr('font-size', 11)
      .attr('fill', P.labelLit).attr('opacity', 0).text(`${Math.round(p * 100)}%`);
    const tc = mix(P.label, i === 0 ? P.mint : P.labelLit);
    on(y - 10, (a) => { bar.attr('width', 300 * p * a); txt.attr('fill', tc(a)); val.attr('opacity', a); });
  });
  wire(`M600,2420 V2439`, 2420);
  const chip = gBody.append('rect').attr('x', 525).attr('y', 2439).attr('width', 150).attr('height', 46).attr('rx', 12)
    .attr('fill', P.panel).attr('stroke', P.edge).attr('stroke-width', 1.5);
  const chipText = gBody.append('text').attr('x', 600).attr('y', 2469).attr('text-anchor', 'middle').attr('font-size', 22)
    .attr('font-family', "'Source Serif 4',Georgia,serif").attr('fill', P.label).text('faster');
  {
    const f = mix(P.panel, '#17463a'), s = mix(P.edge, P.mint), c = mix(P.label, '#ffffff');
    on(2445, (a) => { chip.attr('fill', f(a)).attr('stroke', s(a)).attr('filter', a > 0.5 ? 'url(#mf-glow)' : null); chipText.attr('fill', c(a)); });
  }

  /* ---------- measured results: honest, proportional ---------- */
  caps(360, 2552, 'val loss · −7.21%');
  caps(640, 2552, 'step time · −4.1%');
  const lossY = d3.scaleLinear().domain([5.2, 7.6]).range([2694, 2572]);
  const curve = (final) => d3.line().curve(d3.curveBasis)(d3.range(0, 1.001, 0.02)
    .map((t) => [364 + t * 212, lossY(final + (7.4 - final) * Math.exp(-t * 4.2))]));
  gBody.append('rect').attr('x', 360).attr('y', 2566).attr('width', 220).attr('height', 134).attr('rx', 8)
    .attr('fill', 'none').attr('stroke', '#1c2e24');
  const drawIn = (path, litY) => {
    const len = path.node().getTotalLength();
    path.attr('stroke-dasharray', `${len} ${len}`).attr('stroke-dashoffset', len);
    on(litY, (a) => path.attr('stroke-dashoffset', len * (1 - a)));
  };
  drawIn(gBody.append('path').attr('d', curve(5.848)).attr('fill', 'none').attr('stroke', '#4b6a58').attr('stroke-width', 1.6), 2580);
  drawIn(gBody.append('path').attr('d', curve(5.426)).attr('fill', 'none').attr('stroke', P.teal).attr('stroke-width', 2.2)
    .attr('filter', 'url(#mf-glow)'), 2600);
  mono(586, lossY(5.848) + 4, '5.85', 11, 'start', 2640);
  const evolvedLoss = mono(586, lossY(5.426) + 4, '5.43', 11, 'start', 2650).attr('font-weight', 600);
  { const c = mix(P.label, P.mint); on(2650, (a) => evolvedLoss.attr('fill', c(a))); }
  mono(470, 2718, 'matched 300 s budget →', 10, 'middle', 2700);
  [['baseline · 8.95 ms', 8.95, 2596, '#4b6a58'], ['evolved · 8.58 ms', 8.58, 2646, P.teal]].forEach(([t, ms, y, col], i) => {
    mono(640, y - 6, t, 11, 'start', y - 10);
    gBody.append('rect').attr('x', 640).attr('y', y).attr('width', 210).attr('height', 18).attr('rx', 4).attr('fill', '#142219');
    const bar = gBody.append('rect').attr('x', 640).attr('y', y).attr('width', 0).attr('height', 18).attr('rx', 4).attr('fill', col)
      .attr('filter', i === 1 ? 'url(#mf-glow)' : null);
    on(y, (a) => bar.attr('width', (ms / 9) * 210 * a));
  });
  mono(600, 2752, 'measured · modern-lm 481M (recipe) · karpathy/nanochat (kernel)', 11, 'middle', 2740);
  const exitLine = gBody.append('rect').attr('x', 599).attr('y', 2772).attr('width', 2).attr('height', 128)
    .attr('fill', 'url(#mf-exit)').attr('opacity', 0);
  on(2772, (a) => exitLine.attr('opacity', a));

  /* ---------- scan beam, lane particles, MLP signals ---------- */
  const beamLine = gFx.append('rect').attr('x', 330).attr('width', 540).attr('height', 2).attr('fill', 'url(#mf-beam)')
    .attr('filter', 'url(#mf-glow)').attr('opacity', 0);
  const particles = LANES.flatMap((x) => d3.range(3).map(() => ({
    x, y: LANE_TOP + rand() * 300, v: 0.09 + rand() * 0.07,
    el: gParticles.append('circle').attr('cx', x).attr('r', 2.6).attr('fill', P.mint).attr('opacity', 0),
  })));
  const pick = (arr) => arr[Math.floor(rand() * arr.length)];
  const signals = d3.range(14).map(() => ({
    t: rand(), v: 0.0006 + rand() * 0.0005, path: [pick(inp), pick(hid), pick(out)],
    el: gSignals.append('circle').attr('r', 3).attr('fill', P.mint).attr('opacity', 0),
  }));

  /* ---------- agents ---------- */
  function makeAgent(href, name) {
    const s = { x: 0, y: 0, sx: 1, sy: 1, rot: 0, o: 0, bob: 0, name: 0 };
    const g = gAgents.append('g').attr('opacity', 0);
    g.append('ellipse').attr('cx', 0).attr('cy', 32).attr('rx', 22).attr('ry', 5).attr('fill', '#000').attr('opacity', 0.35);
    g.append('image').attr('href', href).attr('x', -48).attr('y', -48).attr('width', 96).attr('height', 96);
    const tag = g.append('text').attr('class', 'mf-mono').attr('y', 52).attr('text-anchor', 'middle').attr('font-size', 11)
      .attr('fill', P.labelLit).attr('opacity', 0).text(name);
    return { s, g, tag };
  }
  const agents = {
    gpt: makeAgent('assets/agent-gpt.png', 'GPT'),
    claude: makeAgent('assets/agent-claude.png', 'Claude'),
    glm: makeAgent('assets/agent-glm.png', 'GLM'),
  };
  const renderAgents = () => Object.values(agents).forEach(({ s, g, tag }) => {
    g.attr('transform', `translate(${s.x} ${s.y + s.bob}) rotate(${s.rot}) scale(${s.sx} ${s.sy})`).attr('opacity', s.o);
    tag.attr('opacity', s.name);
  });

  const seq = { played: false, running: false, tl: null, idles: [], extras: [] };

  function tag(x, y, text, color = P.amber) {
    const g = gFx.append('g').attr('opacity', 0);
    const w = text.length * 7.3 + 20;
    g.append('rect').attr('x', x - w / 2).attr('y', y - 15).attr('width', w).attr('height', 22).attr('rx', 11)
      .attr('fill', '#0b140f').attr('stroke', color).attr('stroke-opacity', 0.55);
    g.append('text').attr('class', 'mf-mono').attr('x', x).attr('y', y).attr('text-anchor', 'middle').attr('font-size', 12)
      .attr('fill', color).text(text);
    seq.extras.push(g);
    const st = { o: 0, dy: 8 };
    animate(st, { o: 1, dy: 0, duration: 420, ease: 'outBack', onUpdate: () => g.attr('opacity', st.o).attr('transform', `translate(0 ${st.dy})`) });
  }
  function sparks(x, y, color) {
    const ring = gFx.append('circle').attr('cx', x).attr('cy', y).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2);
    const rs = { t: 0 };
    animate(rs, { t: 1, duration: 480, ease: 'outQuad',
      onUpdate: () => ring.attr('r', 4 + 30 * rs.t).attr('opacity', 1 - rs.t), onComplete: () => ring.remove() });
    d3.range(10).forEach((i) => {
      const ang = (i / 10) * Math.PI * 2 + rand() * 0.4, dist = 22 + rand() * 26;
      const c = gFx.append('circle').attr('cx', x).attr('cy', y).attr('r', 3.2).attr('fill', color).attr('filter', 'url(#mf-glow)');
      const st = { t: 0 };
      animate(st, { t: 1, duration: 520 + rand() * 200, ease: 'outQuad',
        onUpdate: () => c.attr('cx', x + Math.cos(ang) * dist * st.t).attr('cy', y + Math.sin(ang) * dist * st.t)
          .attr('r', 3.2 * (1 - st.t)).attr('opacity', 1 - st.t),
        onComplete: () => c.remove() });
    });
  }
  function zap(x1, y1, x2, y2) {
    const pts = d3.range(7).map((i) => {
      const t = i / 6;
      return [x1 + (x2 - x1) * t + (i % 6 ? (rand() - 0.5) * 14 : 0), y1 + (y2 - y1) * t];
    });
    const bolt = gFx.append('path').attr('d', d3.line()(pts)).attr('fill', 'none').attr('stroke', P.amber)
      .attr('stroke-width', 2.2).attr('filter', 'url(#mf-glow)');
    const st = { o: 0 };
    animate(st, { o: [0, 1, 0.2, 1, 0], duration: 520, onUpdate: () => bolt.attr('opacity', st.o), onComplete: () => bolt.remove() });
    sparks(x2, y2, P.amber);
  }
  const squash = (sx, sy) => ({
    sx: [{ to: sx, duration: 70 }, { to: 2 - sx, duration: 130 }, { to: 1, duration: 360, ease: 'outElastic(1, .5)' }],
    sy: [{ to: sy, duration: 70 }, { to: 2 - sy, duration: 130 }, { to: 1, duration: 360, ease: 'outElastic(1, .5)' }],
  });

  function playAgents() {
    seq.played = true;
    seq.running = true;
    const G = agents.gpt.s, Cl = agents.claude.s, R = agents.glm.s;
    const [, L3, L4, L5, L6] = blocks.map((b) => b.s);
    const tl = createTimeline({
      defaults: { ease: 'outQuad' },
      onComplete: () => { seq.running = false; renderBlocks(); startIdle(); },
    });
    seq.tl = tl;

    // GPT rams layer 3's MLP from the right: the hidden layer gets thinner.
    tl.add(G, { o: [0, 1], x: [1130, 925], y: [1610, 1690], duration: 650 }, 0)
      .add(G, { x: 948, sx: 0.88, sy: 1.08, duration: 240, ease: 'outSine' })
      .add(G, { x: 852, sx: 1.16, sy: 0.9, duration: 150, ease: 'inQuad' })
      .call(() => sparks(824, 1690, P.teal))
      .add(G, { ...squash(0.7, 1.3), x: [{ to: 846, duration: 70 }, { to: 892, duration: 480, ease: 'outBack' }] })
      .add(L3, { dx: [{ to: -12, duration: 60 }, { to: 7, duration: 70 }, { to: -3, duration: 70 }, { to: 0, duration: 90 }] }, '<<')
      .add(L3, { thin: 1, duration: 640, ease: 'outBack' }, '<<')
      .call(() => tag(770, 1648, 'MLP 4× → 2×'), '<<+=200');

    // Claude grabs layer 4's attention side and hauls it wider: one more head.
    tl.add(Cl, { o: [0, 1], x: [40, 262], y: [1720, 1770], rot: [-18, 0], duration: 700 }, '+=120')
      .add(Cl, { x: 348, sx: 1.12, sy: 0.9, duration: 200, ease: 'inQuad' })
      .call(() => sparks(378, 1770, P.mint))
      .add(Cl, { ...squash(0.78, 1.22) })
      .add(Cl, { x: 292, rot: -10, duration: 640, ease: 'outBack' }, '<<+=180')
      .add(L4, { wide: 1, duration: 640, ease: 'outBack' }, '<<')
      .add(L4, { dx: [{ to: -8, duration: 90 }, { to: 3, duration: 100 }, { to: 0, duration: 140 }] }, '<<')
      .call(() => tag(430, 1728, '+1 head', P.mint), '<<+=260')
      .add(Cl, { rot: 0, duration: 320 });

    // GLM drops onto layer 5 and shoves it out of the stack.
    tl.add(R, { o: [0, 1], x: [520, 520], y: [1450, 1470], duration: 220 }, '+=120')
      .add(R, { y: 1788, duration: 480, ease: 'inQuad' })
      .call(() => sparks(520, 1820, P.amber))
      .add(R, { ...squash(1.3, 0.72) })
      .add(L5, { dy: [{ to: 8, duration: 80 }, { to: 0, duration: 140 }] }, '<<')
      .add(R, { x: 492, duration: 180, ease: 'outSine' }, '<<+=260')
      .add(R, { x: 560, sx: 1.18, sy: 0.88, duration: 130, ease: 'inQuad' })
      .add(L5, { dx: 640, rot: 18, o: 0, duration: 720, ease: 'inQuad' }, '<<+=40')
      .call(() => tag(880, 1850, '−1 layer'), '<<')
      .add(R, { sx: 1, sy: 1, x: 540, duration: 300, ease: 'outBack' }, '<<')
      .add([L6, tail], { dy: -80, duration: 560, ease: 'outBack' }, '<<+=420');

    // Then hops down to the final norm and fuses it into a Triton kernel.
    tl.add(R, { x: 895, y: [{ to: 1720, duration: 340, ease: 'outQuad' }, { to: 2098, duration: 560, ease: 'inQuad' }], duration: 900 }, '+=260')
      .add(R, { ...squash(1.22, 0.8) })
      .call(() => zap(872, 2094, 846, 2116), '<<')
      .add(kernel, { v: 1, duration: 600, onUpdate: renderKernel }, '<<+=120')
      .call(() => tag(730, 2096, 'Triton · −4.1% step'), '<<+=200');
  }

  function startIdle() {
    Object.values(agents).forEach(({ s }, i) => {
      seq.idles.push(animate(s, { bob: [0, -6], duration: 1100 + i * 180, loop: true, alternate: true, ease: 'inOutSine' }));
      seq.idles.push(animate(s, { name: 1, duration: 400 }));
    });
  }

  const INITIAL = {
    gpt: { x: 1130, y: 1610 }, claude: { x: 40, y: 1720 }, glm: { x: 520, y: 1450 },
  };
  function resetAgents() {
    seq.tl?.cancel();
    seq.idles.forEach((a) => a.cancel());
    seq.idles = [];
    seq.extras.forEach((g) => g.remove());
    seq.extras = [];
    Object.entries(agents).forEach(([k, { s }]) => Object.assign(s, { ...INITIAL[k], sx: 1, sy: 1, rot: 0, o: 0, bob: 0, name: 0 }));
    blocks.forEach((b) => Object.assign(b.s, { dx: 0, dy: 0, o: 1, rot: 0, thin: 0, wide: 0 }));
    tail.dy = 0;
    kernel.v = 0;
    renderKernel();
    renderBlocks();
    seq.played = false;
    seq.running = false;
  }
  resetAgents();

  function finalState() {
    Object.assign(agents.gpt.s, { x: 892, y: 1690, o: 1, name: 1 });
    Object.assign(agents.claude.s, { x: 292, y: 1770, o: 1, name: 1 });
    Object.assign(agents.glm.s, { x: 895, y: 2098, o: 1, name: 1 });
    Object.assign(blocks[1].s, { thin: 1 });
    Object.assign(blocks[2].s, { wide: 1 });
    Object.assign(blocks[3].s, { o: 0 });
    blocks[4].s.dy = -80;
    tail.dy = -80;
    kernel.v = 1;
    renderKernel();
    renderBlocks();
    renderAgents();
  }

  /* ---------- side notes, anchored to component heights ---------- */
  const notes = [...document.querySelectorAll('#fig-notes .fig-note')].map((el) => {
    const y = Number(el.dataset.y);
    el.style.top = `${(y / VB_H) * 100}%`;
    return { el, y };
  });

  /* ---------- the loop ---------- */
  let headTarget = -400, head = -400, lastHead = null, lastT = 0, running = false;
  let activeQ = -2;
  const rowHiState = { y: MAT_Y - 5, o: 0 };

  function measure() {
    const r = svgNode.getBoundingClientRect();
    headTarget = reduceMotion ? VB_H : (innerHeight * 0.62 - r.top) / (r.width / vbW);
  }
  function applyLit() {
    for (const L of lit) {
      const a = Math.max(0, Math.min(1, (head - L.y) / RAMP));
      if (a !== L.a) { L.a = a; L.fn(a); }
    }
  }
  function updateAttention(t) {
    let q = -1;
    for (let i = 0; i < 5; i++) if (head > rowY(i) + 24) q = i;
    // Once the read head is past the matrix, attention keeps "looking around".
    if (head > MAT_Y + 5 * PITCH + 60 && !reduceMotion) q = Math.floor(t / 1300) % 5;
    if (q === activeQ) return;
    activeQ = q;
    arcs.forEach((p, k) => {
      const w = q >= 0 && k <= q ? W[q][k] : 0;
      const yq = rowY(Math.max(q, 0)) + 19, yk = rowY(k) + 19;
      p.attr('d', `M${QX},${yq} C${QX - 58},${yq} ${KX + 58},${yk} ${KX},${yk}`)
        .style('stroke-width', `${1 + 10 * w}px`).style('opacity', q < 0 || k > q ? 0 : 0.25 + 0.75 * w);
    });
    queryDots.forEach((d, i) => d.attr('fill', i === q ? P.amber : i < q ? '#2f6e57' : '#1b2e23'));
    activeLabel.attr('opacity', q < 0 ? 0 : 1).attr('y', rowY(Math.max(q, 0)) + 23).text(q < 0 ? '' : TOKENS[q]);
    animate(rowHiState, {
      y: rowY(Math.max(q, 0)) - 5, o: q < 0 ? 0 : 1, duration: reduceMotion ? 0 : 320, ease: 'outQuad',
      onUpdate: () => rowHi.attr('y', rowHiState.y).attr('opacity', rowHiState.o),
    });
  }
  function updateFlow(dt) {
    const limit = Math.min(head, LANE_END);
    laneLit.forEach((l) => l.attr('y2', Math.max(LANE_TOP, limit)));
    beamLine.attr('y', head - 1).attr('opacity', head > 40 && head < 2780 ? 0.75 : 0);
    particles.forEach((p) => {
      if (!reduceMotion) p.y += p.v * dt;
      if (p.y > limit) p.y = LANE_TOP + rand() * 40;
      p.el.attr('cy', p.y).attr('opacity', limit > LANE_TOP + 20 ? 0.9 : 0);
    });
    const mlpOn = head > INP_Y && !reduceMotion;
    signals.forEach((s) => {
      if (!mlpOn) { s.el.attr('opacity', 0); return; }
      s.t += s.v * dt;
      if (s.t >= 1) { s.t = 0; s.path = [pick(inp), pick(hid), pick(out)]; }
      const [a, b] = s.t < 0.5 ? [s.path[0], s.path[1]] : [s.path[1], s.path[2]];
      const u = s.t < 0.5 ? s.t * 2 : (s.t - 0.5) * 2;
      const y = a.y + (b.y - a.y) * u;
      s.el.attr('cx', a.x + (b.x - a.x) * u).attr('cy', y).attr('opacity', y < head ? 0.95 : 0);
    });
  }
  function frame(t) {
    const dt = Math.min(50, t - lastT || 16);
    lastT = t;
    measure();
    head += (headTarget - head) * (reduceMotion ? 1 : 0.16);
    if (lastHead === null || Math.abs(head - lastHead) > 0.3) { applyLit(); lastHead = head; }
    updateFlow(dt);
    updateAttention(t);
    notes.forEach((n) => n.el.classList.toggle('on', head > n.y - 40));
    if (!reduceMotion) {
      if (!seq.played && head > 1780) playAgents();
      else if (seq.played && head < 1450) resetAgents();
      if (seq.running) renderBlocks();
    }
    renderAgents();
    if (running) requestAnimationFrame(frame);
  }

  if (reduceMotion) finalState();
  new IntersectionObserver(([entry]) => {
    const was = running;
    running = entry.isIntersecting;
    if (running && !was) requestAnimationFrame(frame);
  }, { rootMargin: '200px 0px' }).observe(svgNode);
}
