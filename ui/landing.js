'use strict';
/* Walkthrough visuals use window.TOPK_RUNS from landing-data.js; the repository
   graph uses window.TOPK_BENCHMARKS from the published ten-repository summary. */
(() => {
  const R = window.TOPK_RUNS;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (s, el = document) => el.querySelector(s);
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const sleep = ms => new Promise(r => setTimeout(r, reduce ? 0 : ms));
  const once = (node, cb, threshold = .3) => {
    const io = new IntersectionObserver(es => {
      if (es.some(e => e.isIntersecting)) { io.disconnect(); cb(); }
    }, { threshold });
    io.observe(node);
  };
  const watch = (node, cb) => new IntersectionObserver(es => cb(es[es.length - 1].isIntersecting), { threshold: 0 }).observe(node);
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pct = g => (g >= 0 ? '−' : '+') + Math.abs(g).toFixed(2) + '%';
  const HANDOFF_KEY = 'topk_start_handoff_v1'; // Also consumed by front.js on start.html.

  // Canvas palette (RGB triplets) — mirrors the CSS variables.
  const C = { ink: '237,237,240', gray: '107,107,117', coral: '239,122,109', amber: '242,181,96', accent: '139,151,255', accent2: '196,202,255' };
  const FATE = { unfinished: C.gray, nocompile: C.coral, wrong: C.coral, slower: C.amber, noise: C.amber, accepted: C.accent };
  const kernelFates = R.kernel.candidates.filter(c => c.kind !== 'seed').map(c => c.kind);
  const archFinalists = R.arch.candidates.filter(c => c.gen === 4 && Number.isFinite(c.gain)).sort((a, b) => b.gain - a.gain);
  const archFinal = archFinalists[0];
  const archGain = $('#hero-arch-gain'), kernelGain = $('#hero-kernel-gain');
  archGain.dataset.count = archFinal.gain;
  kernelGain.dataset.count = R.kernel.paired_median * 100;
  $('#hero-arch-model').textContent = `val loss · ${R.arch.model.split('·').pop().trim()} model`;

  /* ---------------- chrome ---------------- */
  document.querySelectorAll('.reveal').forEach(n => once(n, () => n.classList.add('in'), .12));
  const nav = $('#top');
  const onScroll = () => nav.classList.toggle('scrolled', scrollY > 24);
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  document.querySelectorAll('form.repo').forEach(f => f.addEventListener('submit', e => {
    e.preventDefault();
    const input = f.querySelector('input'), note = f.querySelector('.note');
    const raw = input.value.trim();
    if (!raw) { note.textContent = 'Paste the URL of a PyTorch training repo.'; input.focus(); return; }
    let url;
    try {
      url = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`);
      if (url.protocol === 'http:') url.protocol = 'https:';
      if (url.protocol !== 'https:' || !['github.com', 'www.github.com'].includes(url.hostname)
        || url.username || url.password || url.pathname.split('/').filter(Boolean).length < 2) throw new Error('Invalid GitHub URL');
    } catch { note.textContent = 'Paste a GitHub repository URL, such as github.com/owner/repo.'; input.focus(); return; }
    try {
      sessionStorage.setItem(HANDOFF_KEY, JSON.stringify({ repo: url.href, mode: f.querySelector('select').value, requestKey: crypto.randomUUID(), attempted: false, createdAt: Date.now() }));
    } catch { note.textContent = 'This browser could not save the run request. Use a secure browser tab and allow session storage.'; return; }
    location.assign('start.html');
  }));

  /* ---------------- tabs: one view per panel at first glance ---------------- */
  document.querySelectorAll('[data-tabs]').forEach(box => {
    const tabs = [...box.querySelectorAll('[data-tab]')], panes = [...box.querySelectorAll('[data-pane]')];
    const show = key => {
      tabs.forEach(t => { const on = t.dataset.tab === key; t.classList.toggle('on', on); t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; });
      panes.forEach(p => { p.hidden = p.dataset.pane !== key; });
      dispatchEvent(new Event('resize'));   // anything measured while hidden re-fits now
    };
    tabs.forEach((t, i) => {
      t.tabIndex = t.classList.contains('on') ? 0 : -1;
      t.addEventListener('click', () => show(t.dataset.tab));
      t.addEventListener('keydown', e => {
        let step = 0;
        if (e.key === 'ArrowRight') step = 1;
        if (e.key === 'ArrowLeft') step = -1;
        if (!step) return;
        const n = tabs[(i + step + tabs.length) % tabs.length];
        show(n.dataset.tab); n.focus();
      });
    });
  });

  const tip = $('#tip');
  const place = (x, y) => {
    const r = tip.getBoundingClientRect();
    let left = x + 16, top = y + 16;
    if (left + r.width > innerWidth - 12) left = x - r.width - 16;
    if (top + r.height > innerHeight - 12) top = y - r.height - 16;
    tip.style.left = Math.max(12, left) + 'px'; tip.style.top = Math.max(12, top) + 'px';
  };
  const bindTip = (node, html) => {
    node.addEventListener('pointerenter', () => { tip.innerHTML = html(); tip.hidden = false; });
    node.addEventListener('pointermove', e => place(e.clientX, e.clientY));
    node.addEventListener('pointerleave', () => { tip.hidden = true; });
    node.addEventListener('focus', () => { tip.innerHTML = html(); tip.hidden = false; const b = node.getBoundingClientRect(); place(b.right, b.top); });
    node.addEventListener('blur', () => { tip.hidden = true; });
  };

  /* ---------------- tiny 3D: a camera and a canvas loop ---------------- */
  function camera(dist) {
    const cam = { yaw: 0, pitch: 0, dist, scale: 1, cx: 0, cy: 0 };
    cam.project = (x, y, z) => {
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const d = z2 + cam.dist, f = cam.scale / d;
      return [cam.cx + x1 * f, cam.cy - y1 * f, d, cam.dist / d];   // screen x, screen y, depth, size factor
    };
    return cam;
  }
  function animate(cv, host, { resize, step, draw, prewarm = 0 }) {
    const ctx = cv.getContext('2d');
    let raf = 0, last = 0, on = true;
    const fit = () => {
      const dpr = Math.min(2, devicePixelRatio || 1), w = cv.clientWidth, h = cv.clientHeight;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      resize(w, h);
    };
    const frame = t => {
      const dt = Math.min(.05, (t - last) / 1000 || 0); last = t;
      step(dt); draw(ctx);
      raf = on ? requestAnimationFrame(frame) : 0;
    };
    const start = () => { if (!raf && !reduce) { last = performance.now(); raf = requestAnimationFrame(frame); } };
    fit();
    for (let i = 0; i < prewarm; i++) step(1 / 30);
    draw(ctx);
    addEventListener('resize', () => { fit(); draw(ctx); });
    watch(host, v => { on = v; if (v) start(); });
    start();
  }
  const pointer = host => {
    const m = { x: 0, y: 0 };
    host.addEventListener('pointermove', e => { const r = host.getBoundingClientRect(); m.x = (e.clientX - r.left) / r.width - .5; m.y = (e.clientY - r.top) / r.height - .5; });
    host.addEventListener('pointerleave', () => { m.x = 0; m.y = 0; });
    return m;
  };

  /* ---------------- hero: four checks as rings in space ----------------
     Each particle is one real kernel candidate, released in the order the run
     produced them. It stops at the ring (check) it actually failed; its colour
     is revealed only then. Survivors fly through all four and glow. */
  (function hero() {
    const cv = $('#gates'), host = $('#hero'), cam = camera(7.5), mouse = pointer(host);
    const RX = [-2.1, -0.7, 0.7, 2.1], RR = .92;
    const GATE = { unfinished: 0, nocompile: 1, wrong: 2, slower: 4, noise: 4, accepted: 5 };
    let W = 0, H = 0, parts = [], sparks = [], clock = 0, spawnAt = 0, next = 0;
    const pulse = [0, 0, 0, 0], pulseCol = [C.ink, C.ink, C.ink, C.ink], spin = [0, 1.7, 3.1, 4.4];
    const stopOf = p => {
      if (p.g === 0) return p.fz;
      if (p.g === 5) return Infinity;
      return RX[p.g - 1];
    };
    const depthA = d => Math.max(.16, Math.min(1, (10 - d) / 4.5));
    cam.yaw = .62; cam.pitch = .2;

    function spawn() {
      for (let i = 0; i < 8; i++) {
        const kind = kernelFates[next++ % kernelFates.length], a = Math.random() * 6.2832, r = RR * .78 * Math.sqrt(Math.random());
        parts.push({ kind, g: GATE[kind], fz: -3.3 + Math.random(), x: -4 - i * .28 - Math.random() * .3,
          y: r * Math.cos(a), z: r * Math.sin(a), v: .5 + Math.random() * .18, passed: 0, dead: false, a: 1 });
      }
    }
    function step(dt) {
      clock += dt;
      const ease = Math.min(1, dt * 2);
      cam.yaw += (.62 + .2 * Math.sin(clock * .11) + mouse.x * .35 - cam.yaw) * ease;
      cam.pitch += (.2 + .07 * Math.sin(clock * .083) - mouse.y * .2 - cam.pitch) * ease;
      for (let i = 0; i < 4; i++) { spin[i] += dt * (.35 + i * .08) * (i % 2 ? -1 : 1); pulse[i] = Math.max(0, pulse[i] - dt * 1.6); }
      if (clock >= spawnAt) { spawn(); spawnAt = clock + 2.2; }
      for (const p of parts) {
        if (p.dead) { p.a -= dt * (p.g === 0 ? 1.3 : .7); continue; }
        const stop = stopOf(p);
        p.x += p.v * dt;
        while (p.passed < 4 && p.x >= RX[p.passed] && RX[p.passed] < stop) {
          pulse[p.passed] = Math.max(pulse[p.passed], .3); pulseCol[p.passed] = C.ink; p.passed++;
        }
        if (p.x >= stop) {
          p.x = stop; p.dead = true;
          if (p.g > 0) {
            const gi = p.g - 1; pulse[gi] = 1; pulseCol[gi] = FATE[p.kind];
            for (let k = 0; k < 9; k++) sparks.push({ x: p.x, y: p.y, z: p.z, vx: (Math.random() - .7) * .5, vy: (Math.random() - .5) * .6, vz: (Math.random() - .5) * .6, a: 1, c: FATE[p.kind] });
          }
        }
      }
      parts = parts.filter(p => p.a > 0 && p.x < 4.6);
      for (const s of sparks) { s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt; s.a -= dt * 1.3; }
      sparks = sparks.filter(s => s.a > 0);
    }
    function draw(ctx) {
      ctx.clearRect(0, 0, W, H);
      ctx.globalCompositeOperation = 'lighter';
      // the axis the candidates travel along
      const a0 = cam.project(-4.2, 0, 0), a1 = cam.project(4.4, 0, 0);
      ctx.strokeStyle = `rgba(${C.ink},.05)`; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(a0[0], a0[1]); ctx.lineTo(a1[0], a1[1]); ctx.stroke();
      // rings, each with a brighter arc sweeping round it
      for (let i = 0; i < 4; i++) {
        let prev = cam.project(RX[i], RR, 0);
        for (let j = 1; j <= 72; j++) {
          const t = j / 72 * 6.2832, cur = cam.project(RX[i], RR * Math.cos(t), RR * Math.sin(t));
          const rel = ((t - spin[i]) % 6.2832 + 6.2832) % 6.2832, arc = rel < 1.1 ? 1 - rel / 1.1 : 0;
          const a = (.12 + pulse[i] * .55 + arc * .5) * depthA((prev[2] + cur[2]) / 2);
          let col = C.ink;
          if (arc > 0) col = C.accent2;
          if (pulse[i] > .05) col = pulseCol[i];
          ctx.strokeStyle = `rgba(${col},${a.toFixed(3)})`; ctx.lineWidth = 1 + pulse[i] * 1.2 + arc * .7;
          ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(cur[0], cur[1]); ctx.stroke();
          prev = cur;
        }
      }
      for (const p of parts) {
        const kept = p.g === 5 && p.passed >= 4;
        const col = p.dead || kept ? FATE[p.kind] : C.ink;
        const h = cam.project(p.x, p.y, p.z);
        let alpha = .62;
        if (kept) alpha = 1;
        if (p.dead) alpha = .95;
        const a = alpha * Math.max(0, p.a) * depthA(h[2]);
        if (!p.dead) {
          const tl = cam.project(p.x - (kept ? .9 : .35), p.y, p.z);
          const g = ctx.createLinearGradient(tl[0], tl[1], h[0], h[1]);
          g.addColorStop(0, `rgba(${col},0)`); g.addColorStop(1, `rgba(${col},${a.toFixed(3)})`);
          ctx.strokeStyle = g; ctx.lineWidth = (kept ? 2.4 : 1.5) * h[3];
          ctx.beginPath(); ctx.moveTo(tl[0], tl[1]); ctx.lineTo(h[0], h[1]); ctx.stroke();
        }
        if (kept) { ctx.shadowColor = `rgba(${C.accent},.95)`; ctx.shadowBlur = 16; }
        ctx.fillStyle = `rgba(${col},${a.toFixed(3)})`;
        ctx.beginPath(); ctx.arc(h[0], h[1], (kept ? 3.4 : 2.2) * h[3], 0, 6.2832); ctx.fill();
        ctx.shadowBlur = 0;
      }
      for (const s of sparks) {
        const h = cam.project(s.x, s.y, s.z);
        ctx.fillStyle = `rgba(${s.c},${(s.a * .8 * depthA(h[2])).toFixed(3)})`;
        ctx.fillRect(h[0] - .8, h[1] - .8, 1.7, 1.7);
      }
      ctx.globalCompositeOperation = 'source-over';
    }
    animate(cv, host, {
      resize: (w, h) => { W = w; H = h; cam.scale = Math.min(w * 1.05, h * 1.9); cam.cx = w / 2; cam.cy = h * .52; },
      step, draw, prewarm: 330,
    });
  })();

  /* ---------------- the architecture search, in raw validation loss ----------------
     Lower is better and is drawn lower. Later generations train longer (60 s, then
     120 s, then 300 s), so loss also falls from training time alone. The baseline's
     own loss at each budget is drawn as a staircase: a candidate beats the baseline
     only if it sits below the step for its own budget. */
  (function arch() {
    const A = R.arch, svg = $('#arch-svg');
    const proposals = A.candidates.filter(c => c.gen >= 1 && c.gen <= 3);
    $('#arch-summary').textContent = `${proposals.length} proposals across ${new Set(proposals.map(c => c.gen)).size} generations. Lower is better; hover any point to see what an agent tried.`;
    $('#crash-total').textContent = proposals.length;
    // Fixed display scale and generation labels for recorded run 75890bd0.
    const W = 1000, H = 500, X = [80, 290, 490, 690, 880], CRASH = 44, TOP = 84, BOT = 418, HI = 7.0, LO = 5.3;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const y = l => TOP + (HI - l) / (HI - LO) * (BOT - TOP);
    const GEN = [['Baseline', 'original model'], ['Gen 1', 'architecture · 60 s'], ['Gen 2', 'architecture · 60 s'], ['Gen 3', 'hyperparameters · 120 s'], ['Finals', 're-trained · 300 s']];
    const B = A.baseline, L3 = v => v.toFixed(3);

    const axis = S('g', { class: 'axis', 'aria-hidden': 'true' }, svg);
    S('text', { x: 40, y: 16, class: 'title' }, axis).textContent = 'Validation loss · lower is better';
    for (const t of [5.4, 5.8, 6.2, 6.6, 7.0]) {
      S('line', { x1: 40, x2: W - 8, y1: y(t), y2: y(t), 'stroke-dasharray': '2 6' }, axis);
      S('text', { x: 34, y: y(t) + 4, 'text-anchor': 'end' }, axis).textContent = t.toFixed(1);
    }
    const lane = S('g', { class: 'crash-lane', 'aria-hidden': 'true' }, svg);
    S('line', { x1: 40, x2: W - 8, y1: CRASH, y2: CRASH, stroke: '#2e2224', 'stroke-dasharray': '1 5' }, lane);
    S('text', { x: 34, y: CRASH + 4, 'text-anchor': 'end' }, lane).textContent = 'crash';
    // The baseline, trained for each budget, as a staircase.
    const m23 = (X[2] + X[3]) / 2, m34 = (X[3] + X[4]) / 2;
    const stair = S('g', { class: 'stair', 'aria-hidden': 'true' }, svg);
    S('path', { d: `M40,${y(B[60])} H${m23} V${y(B[120])} H${m34} V${y(B[300])} H${W - 8}`, class: 'stair-line' }, stair);
    [[m23 - 8, y(B[60]) - 7, `baseline ${L3(B[60])} · 60 s`], [m34 - 8, y(B[120]) - 7, `${L3(B[120])} · 120 s`], [W - 10, y(B[300]) - 7, `${L3(B[300])} · 300 s`]]
      .forEach(([x, yy, t]) => { S('text', { x, y: yy, 'text-anchor': 'end', class: 'stair-lab' }, stair).textContent = t; });
    const labels = GEN.map((g, i) => {
      const gl = S('g', { class: 'col-label', 'aria-hidden': 'true' }, svg);
      S('text', { x: X[i], y: 466, 'text-anchor': 'middle', class: 't' }, gl).textContent = g[0];
      S('text', { x: X[i], y: 484, 'text-anchor': 'middle', class: 's' }, gl).textContent = g[1];
      return gl;
    });

    const nodes = new Map();
    const base = { id: 'base', gen: 0, x: X[0], y: y(B[60]), base: true };
    nodes.set('base', base);
    const cands = A.candidates.filter(c => c.phase !== 'baseline');
    for (let g = 1; g <= 4; g++) {
      const list = cands.filter(c => c.gen === g).sort((a, b) => a.id - b.id);
      const gap = g === 4 ? 44 : 17;
      list.forEach((c, i) => {
        const crash = c.loss === null || c.loss === undefined;
        nodes.set(c.id, { ...c, crash, x: X[g] + (i - (list.length - 1) / 2) * gap, y: crash ? CRASH : y(c.loss) });
      });
    }
    const parentOf = n => (n.parent === null || n.parent === undefined || n.parent <= 3) ? 'base' : n.parent;
    const finals = archFinalists;
    const win = nodes.get(archFinal.id);
    const lead3 = nodes.get(cands.filter(c => c.gen === 3 && c.gain !== null && c.gain !== undefined).sort((a, b) => b.gain - a.gain)[0].id);
    const leadFinal = finals.map(f => nodes.get(f.id)).find(f => f.parent === lead3.id);
    const chain = id => { const out = []; let n = nodes.get(id); while (n && !n.base) { out.push(n); n = nodes.get(parentOf(n)); } return out; };

    const gEdges = S('g', { 'aria-hidden': 'true' }, svg), gNodes = S('g', {}, svg), gTags = S('g', { 'aria-hidden': 'true' }, svg);
    const curve = (a, b) => { const mx = (a.x + b.x) / 2; return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`; };
    for (const n of nodes.values()) if (!n.base) n.edge = S('path', { d: curve(nodes.get(parentOf(n)), n), class: 'edge' + (n.crash ? ' crash' : '') }, gEdges);
    const nodeDetail = (n, spoken = false) => {
      if (n.crash) return n.err;
      if (n.base) {
        if (spoken) return `Validation loss ${L3(B[60])} after 60 seconds, ${L3(B[120])} after 120 seconds, ${L3(B[300])} after 300 seconds.`;
        return `Val loss ${L3(B[60])} after 60 s, ${L3(B[120])} after 120 s, ${L3(B[300])} after 300 s.`;
      }
      if (spoken) return `Validation loss ${L3(n.loss)}, ${pct(n.gain)} versus the ${n.secs} second baseline.`;
      return `val loss ${L3(n.loss)} · ${pct(n.gain)} vs. the ${n.secs} s baseline`;
    };
    const tipFor = n => {
      if (n.base) return `<div class="h">Baseline · the repo's model + AdamW</div>${nodeDetail(n)}`;
      const head = `<div class="h">#${n.id} · ${GEN[n.gen][0]} · ${GEN[n.gen][1]}</div>`;
      return head + esc(n.strategy) + (n.crash
        ? `<div class="r bad">${esc(nodeDetail(n))}</div>`
        : `<div class="r">${nodeDetail(n)}</div>`);
    };
    for (const n of nodes.values()) {
      const status = [n === win && 'Reported result.', n === lead3 && 'Led at 120 seconds.', n === leadFinal && 'Retested at 300 seconds.'].filter(Boolean).join(' ');
      const spoken = n.base ? `Baseline: ${nodeDetail(n, true)}` : [`Candidate ${n.id}, ${GEN[n.gen][0]}: ${n.strategy}.`, `${n.crash ? 'Crashed: ' : ''}${nodeDetail(n, true)}`, status].filter(Boolean).join(' ');
      const g = S('g', { class: 'node' + (n.base ? ' base' : '') + (n.crash ? ' crash' : ''), tabindex: 0, role: 'img', 'aria-label': spoken }, gNodes);
      if (n.crash) {
        S('line', { x1: n.x - 4.5, y1: n.y - 4.5, x2: n.x + 4.5, y2: n.y + 4.5 }, g);
        S('line', { x1: n.x - 4.5, y1: n.y + 4.5, x2: n.x + 4.5, y2: n.y - 4.5 }, g);
      } else S('circle', { cx: n.x, cy: n.y, r: n.base ? 7 : 5.5, class: 'dot' }, g);
      S('circle', { cx: n.x, cy: n.y, r: 12, class: 'hit' }, g);
      n.el = g;
      bindTip(g, () => tipFor(n));
    }
    const tag = (n, text, cls, dx, dy, anchor) => {
      const t = S('g', { class: 'tag ' + cls }, gTags);
      S('text', { x: n.x + dx, y: n.y + dy, 'text-anchor': anchor }, t).textContent = text;
      return t;
    };
    const tags = [tag(win, `${L3(win.loss)} · reported`, 'win', 0, 22, 'middle'), tag(lead3, `led at 120 s: ${L3(lead3.loss)}`, 'lead', 0, 22, 'middle')];
    if (leadFinal) tags.push(tag(leadFinal, `${L3(leadFinal.loss)} after 300 s`, 'lead', 11, 4, 'start'));
    const trav = S('circle', { r: 4, class: 'traveler', opacity: 0 }, svg);

    // "What crashed" tab: the real last line of each crash, typed out the first time it's opened.
    const list = $('#crash-list'), crashes = [...nodes.values()].filter(n => n.crash && n.gen >= 1 && n.gen <= 3);
    $('#crash-n').textContent = crashes.length;
    const typeInto = (el, msg) => new Promise(done => {
      let i = 0;
      const go = () => { el.textContent = msg.slice(0, ++i); if (i < msg.length) setTimeout(go, 9); else done(); };
      go();
    });
    for (const n of crashes) {
      const li = document.createElement('li');
      li.tabIndex = 0;
      li.setAttribute('aria-label', `${GEN[n.gen][0]}, candidate ${n.id}: ${n.strategy}. Error: ${n.err}`);
      li.innerHTML = `<span class="g">${GEN[n.gen][0]} · #${n.id}</span><code></code>`;
      bindTip(li, () => `<div class="h">What the agent tried</div>${esc(n.strategy)}`);
      list.appendChild(li);
      n.code = li.querySelector('code');
      if (reduce) n.code.textContent = n.err;
    }
    if (!reduce) once(list, async () => { for (const n of crashes) { await typeInto(n.code, n.err); await sleep(90); } }, .1);

    const readout = $('#arch-readout');
    let token = 0, visible = false, played = false, travRaf = 0;
    function travel() {
      cancelAnimationFrame(travRaf);
      if (reduce || !played || !visible) { trav.setAttribute('opacity', 0); return; }
      const edges = chain(win.id).reverse().map(n => n.edge), lens = edges.map(e => e.getTotalLength());
      const total = lens.reduce((a, b) => a + b, 0), t0 = performance.now(), period = 3000, rest = 1600;
      const frame = now => {
        const el = (now - t0) % (period + rest);
        if (el > period) trav.setAttribute('opacity', 0);
        else {
          let s = el / period * total, k = 0;
          while (k < lens.length - 1 && s > lens[k]) { s -= lens[k]; k++; }
          const p = edges[k].getPointAtLength(Math.min(s, lens[k]));
          trav.setAttribute('cx', p.x.toFixed(1)); trav.setAttribute('cy', p.y.toFixed(1));
          trav.setAttribute('opacity', Math.min(1, Math.sin(el / period * Math.PI) * 1.8).toFixed(2));
        }
        travRaf = requestAnimationFrame(frame);
      };
      travRaf = requestAnimationFrame(frame);
    }
    function reset() {
      token++; played = false; travel();
      for (const n of nodes.values()) {
        n.el.classList.remove('on', 'win', 'lead');
        if (!n.edge) continue;
        n.edge.classList.remove('win', 'lead');
        n.edge.style.transition = 'none';
        if (!n.crash) { const L = n.edge.getTotalLength(); n.edge.style.strokeDasharray = L; n.edge.style.strokeDashoffset = L; }
        n.edge.style.opacity = 0;
      }
      tags.forEach(t => t.classList.remove('on'));
      labels.forEach(l => l.classList.remove('on'));
      readout.innerHTML = '&nbsp;';
    }
    function drawEdge(n) {
      const e = n.edge;
      e.getBoundingClientRect();
      e.style.transition = reduce ? 'none' : 'stroke-dashoffset .6s cubic-bezier(.3,.7,.2,1), opacity .3s, stroke .5s, stroke-width .5s';
      e.style.opacity = n.crash ? .75 : 1;
      e.style.strokeDashoffset = 0;
    }
    async function play() {
      reset(); const my = token;
      labels[0].classList.add('on'); base.el.classList.add('on');
      await sleep(450);
      if (my !== token) return;
      for (let g = 1; g <= 4; g++) {
        if (my !== token) return;
        labels[g].classList.add('on');
        const list = [...nodes.values()].filter(n => n.gen === g && !n.base);
        for (const n of list) {
          if (my !== token) return;
          drawEdge(n); await sleep(130);
          if (my !== token) return;
          n.el.classList.add('on');
          await sleep(80);
          if (my !== token) return;
        }
        const ok = list.filter(n => !n.crash), best = ok.reduce((a, b) => b.loss < a.loss ? b : a, ok[0]);
        if (my !== token) return;
        if (g < 4 && best) readout.innerHTML = `${GEN[g][0]} · best <b>${L3(best.loss)}</b> vs. baseline ${L3(B[best.secs])}`;
        await sleep(620);
        if (my !== token) return;
      }
      if (my !== token) return;
      base.el.classList.add('win');
      for (const n of chain(win.id).reverse()) {
        if (my !== token) return;
        n.el.classList.add('win'); n.edge.classList.add('win');
        await sleep(170);
        if (my !== token) return;
      }
      if (my !== token) return;
      lead3.el.classList.add('lead');
      if (leadFinal) { leadFinal.el.classList.add('lead'); leadFinal.edge.classList.add('lead'); }
      tags.forEach(t => t.classList.add('on'));
      readout.innerHTML = `Reported <b>${L3(B[300])} → ${L3(win.loss)}</b> after 300 s (${pct(win.gain)})`
        + (leadFinal ? ` · the 120 s leader ended at <span style="color:var(--amber)">${L3(leadFinal.loss)}</span>` : '');
      played = true; travel();
    }
    reset();
    watch(svg, v => { visible = v; travel(); });
    once(svg, play, .35);
    $('#arch-replay').addEventListener('click', play);
  })();

  /* ---------------- the kernel checks, as a funnel (plays once; ↻ replays) ---------------- */
  (function kernels() {
    const K = R.kernel, all = K.candidates.filter(c => c.kind !== 'seed');
    const n = k => all.filter(c => c.kind === k).length;
    const compiled = all.length - n('unfinished') - n('nocompile'), correct = compiled - n('wrong'), kept = n('accepted');
    $('#k-total').textContent = all.length;
    $('#k-rejected').textContent = all.length - kept;
    const FINAL = [all.length, compiled, correct, correct, kept];
    const ol = $('#stages');
    const nums = ['Proposed', 'Compiles', 'Matches eager', 'Enters timing', 'Faster in the step'].map(l => {
      const li = document.createElement('li');
      li.innerHTML = `<b>0</b><span class="l">${l}</span>`;
      ol.appendChild(li);
      return li.querySelector('b');
    });

    const svg = $('#funnel-svg'), W = 1000, H = 200, LANE = 56, CX = [100, 300, 500, 700, 900], GX = [null, 200, 400, 600, 800];
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    for (let i = 1; i < 5; i++) S('line', { x1: GX[i], x2: GX[i], y1: 12, y2: H - 6, class: 'gate' }, svg);
    const gDots = S('g', {}, svg);
    const FAIL = { unfinished: 1, nocompile: 1, wrong: 2, slower: 4, noise: 4 };
    const hex = rgb => '#' + rgb.split(',').map(part => Number(part).toString(16).padStart(2, '0')).join('');
    const HEX = Object.fromEntries(Object.entries(FATE).map(([kind, color]) => [kind, hex(color)]));

    let dots = [], counts, piles, keptN, T, raf = 0, lastT = 0;
    function build() {
      gDots.innerHTML = ''; dots = []; counts = [0, 0, 0, 0, 0]; piles = [0, 0, 0, 0, 0]; keptN = 0; T = 0;
      nums.forEach(b => { b.textContent = '0'; });
      const gens = [...new Set(all.map(c => c.gen))].sort((a, b) => a - b);
      gens.forEach((g, gi) => all.filter(c => c.gen === g).forEach((c, i) => {
        const el = S('circle', { r: c.kind === 'accepted' ? 5 : 4, cx: -20, cy: LANE, fill: '#e4e4e8', opacity: 0 }, gDots);
        dots.push({ c, el, x: 10, y: LANE + (i - 3.5) * 9, t0: gi * 1.25 + i * .09, state: 'wait', passed: -1, vy: 0,
          drop: c.kind === 'accepted' ? Infinity : GX[FAIL[c.kind]] - 14 });
      }));
    }
    const bump = i => {
      nums[i].textContent = ++counts[i];
      if (!reduce) nums[i].animate([{ transform: 'scale(1.14)' }, { transform: 'scale(1)' }], { duration: 260, easing: 'ease-out' });
    };
    function step(dt) {
      T += dt; let active = false;
      for (const d of dots) {
        if (d.state === 'wait') {
          active = true;
          if (T < d.t0) continue;
          d.state = 'run'; d.el.setAttribute('opacity', 1);
        }
        if (d.state === 'run') {
          active = true;
          d.x += 330 * dt;
          while (d.passed < 4 && d.x >= CX[d.passed + 1]) bump(++d.passed);
          if (d.x >= d.drop) {
            const gate = FAIL[d.c.kind], slot = piles[gate]++;
            d.state = 'fall'; d.el.setAttribute('fill', HEX[d.c.kind]);
            d.tx = GX[gate] + ((slot % 7) - 3) * 9; d.ty = H - 12 - Math.floor(slot / 7) * 9;
          } else if (d.c.kind === 'accepted' && d.passed === 4) {
            d.state = 'kept'; d.x = CX[4] + (keptN++ ? 10 : -10); d.y = LANE;
            d.el.setAttribute('fill', HEX.accepted); d.el.classList.add('glow');
          }
        } else if (d.state === 'fall') {
          active = true;
          d.vy += 1100 * dt; d.y += d.vy * dt; d.x += (d.tx - d.x) * Math.min(1, dt * 9);
          if (d.y >= d.ty) { d.y = d.ty; d.x = d.tx; d.state = 'pile'; d.el.setAttribute('opacity', .8); }
        }
        d.el.setAttribute('cx', d.x.toFixed(1)); d.el.setAttribute('cy', d.y.toFixed(1));
      }
      return active;
    }
    const settle = () => nums.forEach((b, i) => { b.textContent = FINAL[i]; });
    function frame(t) {
      const dt = Math.min(.05, (t - lastT) / 1000 || 0); lastT = t;
      if (step(dt)) { raf = requestAnimationFrame(frame); return; }
      raf = 0; settle();
    }
    function play() {
      cancelAnimationFrame(raf); build();
      if (reduce) { for (let i = 0; i < 4000 && step(1 / 30); i++); settle(); return; }
      lastT = performance.now(); raf = requestAnimationFrame(frame);
    }
    const box = $('.funnel');
    once(box, play, .35);
    $('#k-replay').addEventListener('click', play);
  })();

  /* ---------------- 03: cards tilt toward the pointer ---------------- */
  if (!reduce) document.querySelectorAll('.card').forEach(c => {
    c.addEventListener('pointermove', e => {
      const r = c.getBoundingClientRect(), px = (e.clientX - r.left) / r.width - .5, py = (e.clientY - r.top) / r.height - .5;
      c.style.transform = `rotateY(${(px * 12).toFixed(2)}deg) rotateX(${(-py * 12).toFixed(2)}deg) translateY(-4px)`;
    });
    c.addEventListener('pointerleave', () => { c.style.transform = ''; });
  });

  /* ---------------- repos: accepted results from the published ten-repo benchmark ---------------- */
  (function web() {
    const box = $('#web'), stage = $('#web-stage'), cv = $('#web-canvas'), layer = $('#web-nodes'), hub = $('#hub'), card = $('#run-card');
    if (!box) return;
    const benchmark = window.TOPK_BENCHMARKS;
    if (!benchmark?.results?.length) return;
    $('#benchmark-evidence').href = benchmark.evidence_url;
    const loss = value => {
      if (value === 0) return '0';
      if (value < 1e-6) return value.toExponential(2).replace('e-', 'e−');
      if (value < 1e-3) return value.toFixed(8);
      return value.toFixed(5);
    };
    const go = url => {
      const input = $('#repo-hero');
      if (url) input.value = url;
      if (reduce) { $('#hero').scrollIntoView(); input.focus({ preventScroll: true }); return; }
      box.classList.add('expanding');
      setTimeout(() => $('#hero').scrollIntoView({ behavior: 'smooth', block: 'start' }), 520);
      setTimeout(() => { input.focus({ preventScroll: true }); box.classList.remove('expanding'); }, 1400);
    };
    const RUNS = benchmark.results.map(result => {
      const metrics = [];
      if (result.kernel) {
        const k = result.kernel;
        metrics.push({ value: `${k.improvement_pct.toFixed(3)}%`, label: 'faster kernel step',
          detail: `${k.baseline_ms.toFixed(4)} → ${k.candidate_ms.toFixed(4)} ms versus the compiled baseline; ${k.generations} kernel generations.` });
      }
      if (result.architecture) {
        const a = result.architecture;
        let detail = `${loss(a.baseline_val_loss)} → ${loss(a.candidate_val_loss)} after equal ${a.budget_s} s training; structural change verified.`;
        if (a.caveat === 'near-zero') detail += ' Near-zero synthetic loss makes the relative gain large.';
        if (a.caveat === 'loss-floor') detail += ' This synthetic-task floor effect is not evidence of downstream quality.';
        metrics.push({ value: a.caveat === 'loss-floor' ? `${loss(a.baseline_val_loss)} → ${loss(a.candidate_val_loss)}` : `${a.improvement_pct.toFixed(3)}%`,
          label: a.caveat === 'loss-floor' ? 'validation loss' : 'lower validation loss', detail });
      }
      let chip = result.kernel ? 'kernel' : 'architecture';
      if (result.kernel && result.architecture) chip = 'kernel + architecture';
      if (result.architecture?.caveat === 'loss-floor') chip = 'architecture · loss floor';
      return { name: result.name, short: result.short, chip, metrics };
    });
    const RIN = 1.5;
    const nodes = [];
    RUNS.forEach((r, i) => {
      const a = i / RUNS.length * 6.2832 + .5;
      nodes.push({ ...r, idx: i, x: Math.cos(a) * RIN, y: Math.sin(i * 1.7) * .14, z: Math.sin(a) * RIN });
    });
    const O = { x: 0, y: 0, z: 0 }, links = [];
    nodes.forEach(n => links.push([O, n]));

    hub.addEventListener('click', () => go(''));
    nodes.forEach(n => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'rnode run';
      b.innerHTML = `<i aria-hidden="true"></i><span>${esc(n.short)}</span><em class="chip">${esc(n.chip)}</em>`;
      b.setAttribute('aria-label', `${n.name}: accepted ${n.chip} improvement. Show details.`);
      b.addEventListener('click', () => show(n.idx, true));
      layer.appendChild(b);
      n.el = b;
    });

    // Anchor each node on its dot's centre, not the middle of the dot+label block, so the
    // links end exactly on the dot. Measured from layout, so it follows CSS and font changes.
    const measure = () => nodes.forEach(n => {
      const dot = n.el.querySelector('i');
      n.ax = dot.offsetLeft + dot.offsetWidth / 2;
      n.ay = dot.offsetTop + dot.offsetHeight / 2;
      n.el.style.transformOrigin = `${n.ax}px ${n.ay}px`;
    });
    measure();
    if (document.fonts) document.fonts.ready.then(measure);

    // The run card: cycles through measured runs until someone picks one.
    let sel = 0, t = 0, lastSwitch = 0, userAt = -1e9, redrawStatic = null;
    let hovered = false, focused = false;
    function show(i, user) {
      sel = i; lastSwitch = t;
      if (user) userAt = t;
      const restorePagerFocus = user && card.contains(document.activeElement) && document.activeElement.matches('.pager button');
      const r = RUNS[i];
      card.setAttribute('aria-live', user ? 'polite' : 'off');
      card.innerHTML = `<div class="body"><span class="k">Accepted benchmark result</span><h3>${esc(r.name)}</h3>`
        + r.metrics.map(m => `<div class="run-metric"><div class="big">${esc(m.value)}<small>${esc(m.label)}</small></div><p>${esc(m.detail)}</p></div>`).join('')
        + `<p class="search-detail">Kernel search checked candidate correctness and full-step time against torch.compile. Architecture search tested structural changes, then compared held-out loss at the same training budget. Only improved domains are shown for this project.</p></div>`
        + `<div class="act"><a class="go" href="${esc(benchmark.evidence_url)}" target="_blank" rel="noopener noreferrer">View measured evidence ↗</a><div class="pager">`
        + RUNS.map((q, k) => `<button type="button" class="${k === i ? 'on' : ''}" aria-label="Show ${esc(q.name)}"></button>`).join('')
        + '</div></div>';
      card.querySelectorAll('.pager button').forEach((b, k) => b.addEventListener('click', () => show(k, true)));
      if (restorePagerFocus) card.querySelectorAll('.pager button')[i].focus();
      nodes.forEach(n => n.el.classList.toggle('sel', n.idx === i));
      if (reduce && redrawStatic) redrawStatic();
    }
    card.addEventListener('pointerenter', () => { hovered = true; userAt = t; });
    card.addEventListener('pointerleave', () => { hovered = false; });
    card.addEventListener('focusin', () => { focused = true; userAt = t; });
    card.addEventListener('focusout', e => { if (!card.contains(e.relatedTarget)) focused = false; });
    show(0, false);

    const cam = camera(7), mouse = pointer(stage);
    let W = 0, H = 0, yaw = 0;
    cam.pitch = -.62;   // look down on the web from above
    const depthOf = d => Math.max(0, Math.min(1, (8.9 - d) / 3.8));
    // Yaw that brings a node to the front, a little right of the hub so it doesn't hide behind it.
    const frontYaw = n => Math.atan2(n.z, n.x) + Math.PI / 2 - .55;
    function step(dt) {
      t += dt;
      if (!reduce && !hovered && !focused && t - lastSwitch > 6.5 && t - userAt > 15) show((sel + 1) % RUNS.length, false);
      const target = frontYaw(nodes[sel]) + .08 * Math.sin(t * .35);
      const diff = ((target - yaw) % 6.2832 + 9.4248) % 6.2832 - 3.1416;
      yaw += diff * Math.min(1, dt * 1.6);
      cam.yaw = yaw + mouse.x * .35;
      cam.pitch += (-.62 + mouse.y * .18 - cam.pitch) * Math.min(1, dt * 2);
    }
    function ring(ctx, r) {
      let prev = cam.project(r, 0, 0);
      for (let j = 1; j <= 90; j++) {
        const a = j / 90 * 6.2832, cur = cam.project(Math.cos(a) * r, 0, Math.sin(a) * r);
        ctx.strokeStyle = `rgba(${C.accent2},${(.03 + .09 * depthOf((prev[2] + cur[2]) / 2)).toFixed(3)})`;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(cur[0], cur[1]); ctx.stroke();
        prev = cur;
      }
    }
    function draw(ctx) {
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ring(ctx, RIN);
      for (const [a, b] of links) {
        const pa = cam.project(a.x, a.y, a.z), pb = cam.project(b.x, b.y, b.z), dep = (depthOf(pa[2]) + depthOf(pb[2])) / 2;
        const hot = b.idx === sel;
        ctx.lineWidth = hot ? 2.2 : 1.4;
        ctx.strokeStyle = `rgba(${C.accent},${((hot ? .55 : .2) + .4 * dep).toFixed(3)})`;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      }
      nodes.forEach((n, i) => {
        const k = (t * .35 + i / nodes.length) % 1, p = cam.project(n.x * k, n.y * k, n.z * k), hot = n.idx === sel;
        ctx.fillStyle = `rgba(${C.accent2},${(Math.sin(k * Math.PI) * (hot ? 1 : .6)).toFixed(3)})`;
        ctx.beginPath(); ctx.arc(p[0], p[1], (hot ? 3.2 : 2.2) * p[3], 0, 6.2832); ctx.fill();
      });
      for (const n of nodes) {
        const p = cam.project(n.x, n.y, n.z), dep = depthOf(p[2]);
        n.el.style.transform = `translate(${(p[0] - n.ax).toFixed(1)}px,${(p[1] - n.ay).toFixed(1)}px) scale(${(.7 + .42 * p[3]).toFixed(3)})`;
        n.el.style.opacity = (.55 + .45 * dep).toFixed(3);
        n.el.style.zIndex = String(Math.round(dep * 100));
      }
    }
    animate(cv, stage, {
      resize: (w, h) => {
        W = w; H = h;
        const narrow = w < 700;
        cam.scale = Math.min(w * 1.15, h * 1.65);
        cam.cx = narrow ? w / 2 : w * .65; cam.cy = h * (narrow ? .54 : .57);
        hub.style.left = cam.cx + 'px'; hub.style.top = cam.cy + 'px';
        measure();
      },
      step, draw,
    });
    redrawStatic = () => {
      yaw = frontYaw(nodes[sel]);
      cam.yaw = yaw;
      cam.pitch = -.62;
      draw(cv.getContext('2d'));
    };
    if (reduce) redrawStatic();
  })();

  /* ================= v3 additions ================= */

  /* Hero results count up once visible */
  document.querySelectorAll('[data-count]').forEach(b => once(b, () => {
    const v = +b.dataset.count, dec = +(b.dataset.dec || 2), t0 = performance.now(), dur = reduce ? 0 : 1400;
    const f = now => {
      const k = dur ? Math.min(1, (now - t0) / dur) : 1, e = 1 - Math.pow(1 - k, 3);
      b.textContent = '−' + (v * e).toFixed(dec) + '%';
      if (k < 1) requestAnimationFrame(f);
    };
    requestAnimationFrame(f);
  }, .5));

  /* 01: one generation of the real run, replayed from its agent-call timestamps */
  (function timeline() {
    const T = window.TOPK_TIMELINE, svg = $('#tl-svg');
    if (!T || !svg) return;
    const W = 1000, L = 104, RT = 990, H = 286;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const Y = { plan: 20, agent: 50, gap: 17, gpu: 200, cur: 232, axis: 272 };
    const agentY = i => Y.agent + i * Y.gap;
    const defs = S('defs', {}, svg);
    const pat = S('pattern', { id: 'tl-stripes', width: 8, height: 8, patternUnits: 'userSpaceOnUse' }, defs);
    S('rect', { width: 8, height: 8, fill: 'rgba(139,151,255,.10)' }, pat);
    S('path', { d: 'M-2,10 L10,-2 M6,10 L10,6 M-2,2 L2,-2', stroke: 'rgba(139,151,255,.38)', 'stroke-width': 2 }, pat);
    const gLab = S('g', {}, svg);
    [['Planner', Y.plan], ['Sub-agents', agentY(3.5)], ['GPU', Y.gpu], ['Curator', Y.cur]].forEach(([t, yy]) => {
      S('text', { x: 0, y: yy + 4, class: 'lab' }, gLab).textContent = t;
    });
    S('text', { x: 0, y: Y.gpu + 17, class: 'tick' }, gLab).textContent = 'train & score';
    const gGrid = S('g', {}, svg), gBars = S('g', {}, svg), gOut = S('g', {}, svg);
    const head = S('line', { class: 'playhead', y1: 6, y2: Y.cur + 14 }, svg);
    const clock = $('#tl-clock'), spend = $('#tl-spend'), tabs = [...document.querySelectorAll('#tl-tabs button')];
    const SWEEP = 7000, HOLD = 2600, MANUAL_HOLD = 30000;
    let gi = 0, sel = null, items = [], raf = 0, visible = false, t0 = 0, manualUntil = 0;
    const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

    function build(i) {
      gi = i; tabs.forEach((b, k) => { b.classList.toggle('on', k === i); b.setAttribute('aria-pressed', String(k === i)); });
      gGrid.innerHTML = ''; gBars.innerHTML = ''; gOut.innerHTML = ''; items = [];
      const g = T.gens[i], start = g.planner[0], end = g.curate[0] + g.curate[1], span = end - start;
      const X = t => L + (t - start) / span * (RT - L);
      const every = span > 700 ? 2 : 1;
      for (let m = 0; m * 60 <= span; m += every) {
        const x = X(start + m * 60);
        S('line', { x1: x, x2: x, y1: 8, y2: Y.cur + 12, class: 'grid' }, gGrid);
        S('text', { x, y: Y.axis, class: 'tick', 'text-anchor': 'middle' }, gGrid).textContent = `${m} min`;
      }
      const bar = (s, d, y, h, cls, extra = {}) => {
        const x0 = X(s), w = Math.max(X(s + d) - x0, 3);
        const r = S('rect', { x: x0, y: y - h / 2, width: 0, height: h, rx: Math.min(3, h / 2), class: cls, ...extra }, gBars);
        items.push({ r, w, s, e: s + d });
        return r;
      };
      const reveal = (el, at) => { el.style.opacity = 0; el.style.transition = 'opacity .35s'; items.push({ el, s: at }); };
      bar(g.planner[0], g.planner[1], Y.plan, 10, 'b-plan');
      const ok = g.agents.filter(a => !a.crash), best = ok.reduce((a, b) => b.gain > a.gain ? b : a, ok[0]);
      g.agents.forEach((a, k) => {
        const r = bar(a.s, a.d, agentY(k), 9, 'b-agent');
        bindTip(r, () => `<div class="h">Sub-agent · wrote for ${Math.round(a.d)} s</div>${esc(a.strategy)}<div class="r ${a.crash ? 'bad' : ''}">${a.crash ? 'crashed when it ran' : pct(a.gain) + ' val loss vs. baseline'}</div>`);
        const x = X(g.train[1]) - 10, y0 = agentY(k);
        let m;
        if (a.crash) { m = S('text', { x, y: y0 + 4.5, class: 'o-crash', 'text-anchor': 'middle' }, gOut); m.textContent = '×'; }
        else m = S('circle', { cx: x, cy: y0, r: a === best ? 4.5 : 3, class: a === best ? 'o-best' : 'o-ok' }, gOut);
        reveal(m, g.train[1] - 1);
        if (a === best) {
          const t = S('text', { x: x - 11, y: y0 + 4, class: 'o-lab', 'text-anchor': 'end' }, gOut);
          t.textContent = `best ${pct(a.gain)}`;
          reveal(t, g.train[1] - 1);
        }
      });
      bar(g.train[0], g.train[1] - g.train[0], Y.gpu, 13, 'b-gpu', { fill: 'url(#tl-stripes)' });
      bar(g.curate[0], g.curate[1], Y.cur, 10, 'b-cur');
      spend.textContent = g.spend_before !== null && g.spend_before !== undefined ? `$${g.spend_before.toFixed(2)}` : '—';
      sel = { X, start, end, span };
    }
    function update(ts) {
      for (const it of items) {
        if (it.r) it.r.setAttribute('width', (it.w * Math.max(0, Math.min(1, (ts - it.s) / Math.max(.001, it.e - it.s)))).toFixed(1));
        else it.el.style.opacity = ts >= it.s ? 1 : 0;
      }
      const x = sel.X(Math.min(ts, sel.end)).toFixed(1);
      head.setAttribute('x1', x); head.setAttribute('x2', x);
      clock.textContent = mmss(Math.max(0, ts - sel.start));
    }
    function frame(now) {
      const el = now - t0;
      pat.setAttribute('patternTransform', `translate(${((now / 60) % 8).toFixed(2)},0)`);
      if (el < SWEEP) update(sel.start + el / SWEEP * sel.span);
      else {
        update(sel.end);
        if (el > SWEEP + HOLD && now > manualUntil) { build((gi + 1) % T.gens.length); t0 = now; }
      }
      raf = visible ? requestAnimationFrame(frame) : 0;
    }
    const start = () => { if (!raf && !reduce) { t0 = performance.now(); raf = requestAnimationFrame(frame); } };
    tabs.forEach((b, k) => b.addEventListener('click', () => { build(k); t0 = performance.now(); manualUntil = t0 + MANUAL_HOLD; update(reduce ? sel.end : sel.start); }));
    build(0); update(reduce ? sel.end : sel.start);
    watch(svg, v => { visible = v; if (v) start(); });
  })();

  /* 02: the recipe that won, walked from the winner back through its parents */
  (function recipe() {
    const A = R.arch, host = $('#steps'), byId = new Map(A.candidates.map(c => [c.id, c]));
    // Short names for the four real strategies in recorded run 75890bd0 (full text on hover).
    const TITLE = { 8: 'Parallel attention + MLP block', 16: 'One shared key/value head', 24: 'Cyclic learning-rate schedule', 29: 'Re-trained for the full 300 s' };
    const GEN = ['Baseline', 'Gen 1', 'Gen 2', 'Gen 3', 'Finals'];
    const win = archFinal;
    const path = [];
    for (let c = win; c && c.gen > 0; c = byId.get(c.parent)) path.unshift(c);
    path.forEach((c, i) => {
      const el = document.createElement('div');
      el.className = 'step' + (i === path.length - 1 ? ' last' : '');
      el.tabIndex = 0;
      el.innerHTML = `<span class="g">${GEN[c.gen]} · #${c.id}</span><h4>${esc(TITLE[c.id] || c.strategy)}</h4><span class="v">${pct(c.gain)} after ${c.secs} s</span>`;
      bindTip(el, () => `<div class="h">#${c.id} · ${GEN[c.gen]}</div>${esc(c.strategy)}`);
      host.appendChild(el);
    });

    // Light up each circle as the glide dot reaches it; the last one stays lit
    // until the loop resets, since there's no next circle to hand off to.
    const glide = host.querySelector('.glide'), steps = [...host.querySelectorAll('.step')];
    if (glide && steps.length && !reduce) {
      let offsets = steps.map(s => s.offsetLeft), raf = 0, vis = false;
      addEventListener('resize', () => { offsets = steps.map(s => s.offsetLeft); });
      const frame = () => {
        if (getComputedStyle(glide).display === 'none') {
          steps.forEach(s => s.classList.remove('lit'));
        } else {
          const x = parseFloat(getComputedStyle(glide).left) || 0;
          let idx = -1;
          for (let i = 0; i < offsets.length; i++) if (x >= offsets[i] - 2) idx = i;
          steps.forEach((s, i) => s.classList.toggle('lit', i === idx));
        }
        raf = vis ? requestAnimationFrame(frame) : 0;
      };
      watch(host, v => { vis = v; if (v && !raf) raf = requestAnimationFrame(frame); });
    }
  })();

  /* 03: a small looping picture for each check */
  (function checks() {
    const wave = (x, t, h) => Math.sin(x * .045 + t * 1.6) * h * .22 + Math.sin(x * .11 + t * .9) * h * .07;
    const DRAW = {
      compile(ctx, w, h, t) {
        const n = 14, gap = 4, cw = (w - gap * (n - 1)) / n, ch = 16, y = h / 2 - ch / 2, p = (t * 7) % 24;
        const flash = p > 17 ? Math.max(0, 1 - (p - 17) / 5) : 0;
        for (let i = 0; i < n; i++) {
          ctx.fillStyle = p > i ? `rgba(${C.accent2},${(.45 + flash * .55).toFixed(2)})` : 'rgba(255,255,255,.06)';
          ctx.fillRect(i * (cw + gap), y, cw, ch);
        }
      },
      eager(ctx, w, h, t) {
        const mid = h / 2;
        ctx.fillStyle = `rgba(${C.accent},.09)`;
        ctx.beginPath();
        for (let x = 0; x <= w; x += 3) ctx.lineTo(x, mid + wave(x, t, h) - 5);
        for (let x = w; x >= 0; x -= 3) ctx.lineTo(x, mid + wave(x, t, h) + 5);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = `rgba(${C.ink},.5)`; ctx.beginPath();
        for (let x = 0; x <= w; x += 3) ctx.lineTo(x, mid + wave(x, t, h));
        ctx.stroke();
        ctx.strokeStyle = `rgba(${C.accent},.95)`; ctx.beginPath();
        for (let x = 0; x <= w; x += 3) ctx.lineTo(x, mid + wave(x, t, h) + Math.sin(x * .7 + t * 6) * 1.2);
        ctx.stroke();
      },
      alone(ctx, w, h, t) {
        const k = (t % 2.8) / 2.8, run = Math.min(1, k / .8), rows = [[h * .32, 1, C.ink, .4], [h * .7, .955, C.accent, .95]];
        for (const [y, pace, col, a] of rows) {
          ctx.fillStyle = 'rgba(255,255,255,.05)'; ctx.fillRect(0, y - 6, w, 12);
          ctx.fillStyle = `rgba(${col},${a})`; ctx.fillRect(0, y - 6, w * Math.min(1, run / pace), 12);
        }
      },
      step(ctx, w, h, t) {
        const n = 12, gap = 4, bw = (w - gap * (n - 1)) / n, shown = ((t % 3.8) / 3.8) * n * 1.3;
        for (let i = 0; i < n && i < shown; i++) {
          const cand = i % 2 === 1, bh = h * (cand ? .52 : .6) * (.95 + .05 * Math.sin(i * 1.9));
          ctx.fillStyle = cand ? `rgba(${C.accent},.9)` : `rgba(${C.ink},.3)`;
          ctx.fillRect(i * (bw + gap), h - bh - 2, bw, bh);
        }
      },
    };
    document.querySelectorAll('.check canvas').forEach(cv => {
      const draw = DRAW[cv.dataset.kind], ctx = cv.getContext('2d');
      let w = 0, h = 0, raf = 0, vis = false;
      const fit = () => {
        const dpr = Math.min(2, devicePixelRatio || 1); w = cv.clientWidth; h = cv.clientHeight;
        cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      };
      const paint = t => { ctx.clearRect(0, 0, w, h); draw(ctx, w, h, t); };
      const frame = now => { paint(now / 1000); raf = vis ? requestAnimationFrame(frame) : 0; };
      fit(); paint(2.2);
      addEventListener('resize', () => { fit(); paint(2.2); });
      watch(cv, v => { vis = v; if (v && !raf && !reduce) raf = requestAnimationFrame(frame); });
    });
  })();

  /* 03: what shipped, and how the final number was measured */
  (function shipped() {
    const K = R.kernel, byId = new Map(K.candidates.map(c => [c.id, c])), path = [];
    for (let c = K.candidates.filter(c => c.kind === 'accepted').sort((a, b) => b.gen - a.gen)[0]; c; c = byId.get(c.parent)) path.unshift(c);
    const lin = $('#k-lineage');
    path.forEach((c, i) => {
      const li = document.createElement('li'), ms = Number(c.ms), inc = Number(c.inc);
      if (i === path.length - 1) li.className = 'best';
      if (c.kind === 'seed') {
        li.innerHTML = `<span class="ms">${ms.toFixed(2)} ms</span><span class="id">starting point</span><p>torch.compile (inductor), the incumbent to beat.</p>`;
      } else if (c.kind === 'accepted') {
        li.innerHTML = `<span class="ms">${ms.toFixed(2)} ms</span><span class="id">#${esc(c.id)}</span><span class="dx">${pct((inc - ms) / inc * 100)} vs. incumbent</span><p>${esc(c.strategy)}</p>`;
      } else {
        li.innerHTML = `<span class="ms">${ms.toFixed(2)} ms</span><span class="id">#${esc(c.id)} · within noise, kept as a parent</span><p>${esc(c.strategy)}</p>`;
      }
      lin.appendChild(li);
    });
    K.shipped_kernels.forEach(k => { const c = document.createElement('code'); c.textContent = k; $('#k-chips').appendChild(c); });
    $('#k-fine').textContent = `Triton kernels installed in the backward pass. ${K.full_state_checks.passed} of ${K.full_state_checks.total} full-state checks passed.`;

    const bs = $('#blocks-svg'), BW = 460, BH = 176, L = 58, RR = 440, MAXG = .08;
    bs.setAttribute('viewBox', `0 0 ${BW} ${BH}`);
    const bx = g => L + g / MAXG * (RR - L);
    for (const t of [0, .02, .04, .06, .08]) S('text', { x: bx(t), y: BH - 4, 'text-anchor': 'middle', class: 'axis-t' }, bs).textContent = t ? `−${Math.round(t * 100)}%` : '0%';
    S('line', { x1: bx(K.threshold), x2: bx(K.threshold), y1: 18, y2: BH - 18, class: 'thr' }, bs);
    S('text', { x: bx(K.threshold) - 4, y: 12, 'text-anchor': 'end', class: 'lab t' }, bs).textContent = `${Math.round(K.threshold * 100)}% bar`;
    K.blocks.forEach((b, i) => {
      const yy = 34 + i * 30, row = S('g', { class: 'row' }, bs);
      S('text', { x: 0, y: yy + 4 }, row).textContent = `block ${i + 1}`;
      S('line', { x1: bx(0), x2: bx(b.gain), y1: yy, y2: yy, class: 'stem', style: `--d:${i * .12}s` }, row);
      const pt = S('circle', { cx: bx(b.gain), cy: yy, r: 4.5, class: 'pt', style: `--d:${i * .12}s` }, row);
      bindTip(pt, () => `<div class="h">Block ${i + 1}</div>Original ${b.baseline.toFixed(2)} ms · evolved ${b.candidate.toFixed(2)} ms<div class="r">${pct(b.gain * 100)} step time</div>`);
    });
    S('line', { x1: bx(K.paired_median), x2: bx(K.paired_median), y1: 18, y2: BH - 18, class: 'med' }, bs);
    S('text', { x: bx(K.paired_median) + 4, y: 12, class: 'lab m' }, bs).textContent = `reported ${pct(K.paired_median * 100)}`;
    bs.setAttribute('aria-label', `Four interleaved timing blocks; the reported paired median is ${(K.paired_median * 100).toFixed(1)}%.`);
    $('#blocks-cap').innerHTML = `Original and evolved steps, timed back to back in four blocks. We report the paired median, <b>${pct(K.paired_median * 100)}</b> (${K.baseline_ms.toFixed(2)} → ${K.final_ms.toFixed(2)} ms), not the best block (${pct(K.best_block * 100)}).`;
    once(bs, () => bs.classList.add('on'), .4);
  })();

  /* 04: models on a slowly turning 3D carousel (tiles always face the viewer) */
  (function models() {
    const stage = $('#carousel');
    if (!stage) return;
    const MODELS = [
      { img: 'assets/agent-claude.png', name: 'Claude', via: 'Anthropic API or Claude login' },
      { img: 'assets/agent-gpt.png', name: 'GPT', via: 'OpenAI API or ChatGPT login' },
      { img: 'assets/agent-kimi.png', name: 'Kimi K2.7', via: 'W&B Inference · the run above' },
      { img: 'assets/agent-deepseek.png', name: 'DeepSeek V4 Flash', via: 'W&B Inference · default' },
      { img: 'assets/agent-glm.png', name: 'GLM', via: 'W&B Inference' },
    ];
    const tiles = MODELS.map(m => {
      const el = document.createElement('div');
      el.className = 'tile';
      el.innerHTML = `<img src="${esc(m.img)}" alt=""><b>${esc(m.name)}</b><span>${esc(m.via)}</span>`;
      stage.appendChild(el);
      return el;
    });
    let a = 0, raf = 0, vis = false, last = 0, drag = null;
    stage.addEventListener('pointerleave', () => { drag = null; });
    stage.addEventListener('pointerdown', e => { drag = { x: e.clientX, a }; });
    addEventListener('pointerup', () => { drag = null; });
    stage.addEventListener('pointermove', e => { if (drag) { a = drag.a + (e.clientX - drag.x) / 260; place(); } });
    function place() {
      const R = Math.min(420, stage.clientWidth * .34);
      tiles.forEach((el, i) => {
        const ang = a + i / tiles.length * 6.2832, x = Math.sin(ang) * R, z = Math.cos(ang) * R, depth = (z + R) / (2 * R);
        el.style.transform = `translate(-50%,-50%) translate3d(${x.toFixed(1)}px,${(-Math.cos(ang) * 16).toFixed(1)}px,${(z - R).toFixed(1)}px)`;
        el.style.opacity = (.22 + .78 * depth).toFixed(3);
        el.style.zIndex = Math.round(depth * 100);
        el.style.filter = depth > .8 ? 'none' : `blur(${((.8 - depth) * 2.2).toFixed(2)}px)`;
      });
    }
    const frame = now => {
      const dt = Math.min(.05, (now - last) / 1000); last = now;
      if (!drag) a += dt * .22;
      place();
      raf = vis ? requestAnimationFrame(frame) : 0;
    };
    place();
    addEventListener('resize', place);
    watch(stage, v => { vis = v; if (v && !raf && !reduce) { last = performance.now(); raf = requestAnimationFrame(frame); } });
  })();

})();
