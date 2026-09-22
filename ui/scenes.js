import { animate, stagger, utils } from 'https://cdn.jsdelivr.net/npm/animejs/+esm';

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ------------------------------------------------------------------ *
 * Native scroll. The model section (model-figure.js) is one tall figure;
 * the sections after it animate in once as they reach the viewport.
 * ------------------------------------------------------------------ */
const sections = [...document.querySelectorAll('.scene')];
const dotsNav = document.getElementById('scene-dots');

sections.forEach((section) => {
  const dot = document.createElement('button');
  dot.type = 'button';
  dot.setAttribute('aria-label', 'Go to ' + section.dataset.scene);
  dot.addEventListener('click', () => section.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' }));
  dotsNav.appendChild(dot);
});

function paintDots() {
  const mid = innerHeight / 2;
  const i = sections.findIndex((s) => { const r = s.getBoundingClientRect(); return r.top <= mid && r.bottom > mid; });
  if (i < 0) return;
  [...dotsNav.children].forEach((dot, j) => dot.classList.toggle('active', j === i));
  dotsNav.classList.toggle('on-dark', sections[i].classList.contains('scene-dark'));
}
addEventListener('scroll', paintDots, { passive: true });
paintDots();

function onceVisible(el, fn, threshold = 0.25) {
  if (!el) return;
  const io = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    io.disconnect();
    fn();
  }, { threshold });
  io.observe(el);
}

function scrollToSearch() {
  document.querySelector('#scene-search').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' });
  const input = document.querySelector('#repository');
  if (input) setTimeout(() => input.focus({ preventScroll: true }), reduceMotion ? 0 : 700);
}
document.querySelector('#cta-add-yours')?.addEventListener('click', scrollToSearch);

/* Hero */
if (!reduceMotion) {
  animate('.model-hero-sponsors img', { opacity: [0, 1], translateY: [10, 0], delay: stagger(80), duration: 600 });
  animate('.model-kicker', { opacity: [0, 1], translateY: [10, 0], delay: 200, duration: 600 });
  animate('.model-title', { opacity: [0, 1], translateY: [22, 0], delay: 320, duration: 900, ease: 'outQuart' });
  animate('.model-stats', { opacity: [0, 1], translateY: [14, 0], delay: 700, duration: 700 });
  animate('.model-scroll-hint', { opacity: [0, 0.9], delay: 1100, duration: 700 });
}

/* Ambient drifting blobs */
if (!reduceMotion) {
  document.querySelectorAll('.ambient span').forEach((el, i) => {
    animate(el, {
      translateX: () => utils.random(-50, 50),
      translateY: () => utils.random(-40, 40),
      scale: [1, 1.18],
      duration: () => utils.random(7000, 12000),
      loop: true,
      alternate: true,
      ease: 'inOutSine',
      delay: i * 260,
    });
  });
}

/* CTA: the top-K bars grow in */
onceVisible(document.querySelector('#scene-cta'), () => {
  const bars = document.querySelectorAll('.topk-bar');
  bars.forEach((bar) => {
    bar.style.transformBox = 'fill-box';
    bar.style.transformOrigin = 'center bottom';
    bar.style.transform = reduceMotion ? 'scaleY(1)' : 'scaleY(0)';
  });
  if (reduceMotion) return;
  animate(bars, { scaleY: [0, 1], delay: stagger(60), duration: 500, ease: 'outQuad' });
  animate('.cta-copy', { opacity: [0, 1], delay: 500 });
});

/* ------------------------------------------------------------------ *
 * The repo web: a mesh where nodes connect to their nearest neighbors,
 * with Top-Kernel as a bigger, clickable node near the center.
 * ------------------------------------------------------------------ */
onceVisible(document.querySelector('#scene-web'), webEnter, 0.2);

function webEnter() {
  const stage = document.querySelector('#web-stage');
  const svgEl = document.querySelector('#web-svg');
  const tooltip = document.querySelector('#web-tooltip');
  const repos = window.TOPK_REPOS || [];
  if (!stage || !svgEl || !repos.length) return;

  const VB_W = 900, VB_H = 560;
  svgEl.setAttribute('viewBox', `0 0 ${VB_W} ${VB_H}`);

  const hub = { id: 'hub', name: 'Top-Kernel', isHub: true, r: 34,
    x: VB_W / 2 + (Math.random() - 0.5) * 40, y: VB_H / 2 + (Math.random() - 0.5) * 40 };
  const nodes = [hub, ...repos.map((repo) => ({
    ...repo,
    r: Math.max(11, Math.min(26, 10 + repo.pct * 1.05)),
    x: VB_W / 2 + (Math.random() - 0.5) * 760,
    y: VB_H / 2 + (Math.random() - 0.5) * 460,
  }))];

  for (let iter = 0; iter < 260; iter++) {
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      let fx = (VB_W / 2 - a.x) * 0.0012;
      let fy = (VB_H / 2 - a.y) * 0.0012;
      for (let j = 0; j < nodes.length; j++) {
        if (i === j) continue;
        const b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.max(14, Math.hypot(dx, dy));
        const minDist = a.r + b.r + 20;
        if (dist < minDist) {
          const push = (minDist - dist) / dist * 0.6;
          fx += dx * push * 0.5;
          fy += dy * push * 0.5;
        }
      }
      a.x = Math.max(a.r + 6, Math.min(VB_W - a.r - 6, a.x + fx));
      a.y = Math.max(a.r + 6, Math.min(VB_H - a.r - 6, a.y + fy));
    }
  }

  // Mesh: every node links to its two nearest neighbors.
  const edgeKeys = new Set();
  const edges = [];
  nodes.forEach((a, i) => {
    nodes
      .map((b, j) => ({ j, d: i === j ? Infinity : Math.hypot(a.x - b.x, a.y - b.y) }))
      .sort((p, q) => p.d - q.d)
      .slice(0, 2)
      .forEach(({ j }) => {
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (edgeKeys.has(key)) return;
        edgeKeys.add(key);
        edges.push([nodes[i], nodes[j]]);
      });
  });

  const NS = 'http://www.w3.org/2000/svg';
  const pan = document.createElementNS(NS, 'g');
  svgEl.appendChild(pan);

  edges.forEach(([a, b]) => {
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('class', 'web-link');
    line.setAttribute('x1', String(a.x));
    line.setAttribute('y1', String(a.y));
    line.setAttribute('x2', String(b.x));
    line.setAttribute('y2', String(b.y));
    pan.appendChild(line);
  });

  nodes.forEach((node) => {
    // Position lives on this outer group's SVG `transform` attribute; anime animates
    // scale/drift via inline CSS `transform` on the inner group, never on this one.
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', node.isHub ? 'web-hub' : 'web-node' + (node.verified ? ' verified' : ''));
    g.setAttribute('tabindex', '0');
    g.setAttribute('transform', `translate(${node.x} ${node.y})`);

    const inner = document.createElementNS(NS, 'g');
    inner.style.transform = 'scale(0)';
    inner.style.transformBox = 'fill-box';
    inner.style.transformOrigin = 'center';
    g.appendChild(inner);

    if (!node.isHub) {
      const halo = document.createElementNS(NS, 'circle');
      halo.setAttribute('r', String(node.r * 1.7));
      halo.setAttribute('fill', 'none');
      halo.setAttribute('stroke', node.verified ? '#7be8c6' : '#5fd9b8');
      halo.setAttribute('stroke-opacity', '0.18');
      inner.appendChild(halo);
      node._halo = halo;
    }
    const circle = document.createElementNS(NS, 'circle');
    circle.setAttribute('r', String(node.r));
    inner.appendChild(circle);
    if (node.isHub || node.r > 15) {
      const label = document.createElementNS(NS, 'text');
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('dy', node.isHub ? '4' : String(node.r + 12));
      label.textContent = node.isHub ? 'TOP-K' : (node.name.split('/')[1] || node.name);
      inner.appendChild(label);
    }

    if (node.isHub) {
      g.setAttribute('role', 'button');
      g.setAttribute('aria-label', 'Start with your repository');
      g.addEventListener('click', scrollToSearch);
      g.addEventListener('keydown', (event) => { if (event.key === 'Enter') scrollToSearch(); });
    } else {
      const show = () => {
        tooltip.hidden = false;
        tooltip.innerHTML = `<b>${node.name}</b>${node.pct}% training-step improvement` +
          `<span class="tag${node.verified ? '' : ' sample'}">${node.verified ? 'Verified run' : 'Illustrative sample'}</span>` +
          (node.note ? `<div style="margin-top:6px;color:#93a394">${node.note}</div>` : '');
        const rect = g.getBoundingClientRect();
        const stageRect = stage.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width / 2 - stageRect.left) + 'px';
        tooltip.style.top = (rect.top - stageRect.top) + 'px';
      };
      const hide = () => { tooltip.hidden = true; };
      g.addEventListener('pointerenter', show);
      g.addEventListener('pointerleave', hide);
      g.addEventListener('focus', show);
      g.addEventListener('blur', hide);
    }
    pan.appendChild(g);
    node._inner = inner;
  });

  if (reduceMotion) {
    nodes.forEach((node) => { node._inner.style.transform = 'scale(1)'; });
    return;
  }

  animate(nodes.map((n) => n._inner), { scale: [0, 1], delay: stagger(16, { from: 'center' }), ease: 'outElastic(1, .6)' });
  nodes.forEach((node) => {
    if (node._halo) {
      animate(node._halo, {
        strokeOpacity: [0.1, 0.32], duration: 1600 + Math.random() * 1400,
        loop: true, alternate: true, delay: 900 + Math.random() * 1200, ease: 'inOutSine',
      });
    }
    animate(node._inner, {
      translateX: () => utils.random(-4, 4), translateY: () => utils.random(-4, 4),
      duration: () => utils.random(3200, 5200), loop: true, alternate: true,
      delay: 900 + Math.random() * 1500, ease: 'inOutSine',
    });
  });

  let dragging = false, moved = false, startX = 0, startY = 0, panX = 0, panY = 0;
  stage.addEventListener('pointerdown', (event) => {
    dragging = true; moved = false;
    startX = event.clientX; startY = event.clientY;
    stage.classList.add('dragging');
  });
  addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = event.clientX - startX, dy = event.clientY - startY;
    if (Math.hypot(dx, dy) > 4) moved = true;
    startX = event.clientX; startY = event.clientY;
    panX = Math.max(-260, Math.min(260, panX + dx));
    panY = Math.max(-180, Math.min(180, panY + dy));
    pan.setAttribute('transform', `translate(${panX} ${panY})`);
  });
  addEventListener('pointerup', () => {
    dragging = false;
    stage.classList.remove('dragging');
    if (moved) tooltip.hidden = true;
  });
}
