'use strict';
(async () => {
  const $ = selector => document.querySelector(selector);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = value => !Number.isFinite(value) ? '—' : value === 0 ? '0' : Math.abs(value) < .001 ? value.toExponential(4) : value.toFixed(4);
  const views = ['exploration', 'dashboard', 'results', 'wandb', 'marimo', 'aria'];
  const url = new URL(location.href), repo = url.searchParams.get('project');
  let project, current, elapsed = 0, timer, lastFrame = '', selected = null;
  let view = views.includes(url.searchParams.get('view')) ? url.searchParams.get('view') : 'exploration';
  let startedAt = Date.now();
  const storageKey = 'topk-project-replay:' + repo;
  const storeStart = () => {try {sessionStorage.setItem(storageKey, String(startedAt));} catch { /* Playback also works without storage. */ }};
  function safeLink(selector, value) {
    const link = $(selector);
    try {const target = new URL(value);if (target.protocol !== 'https:' || target.username || target.password) throw Error();link.href = target.href;}
    catch {link.hidden = true;}
  }
  function selectView(next, writeHistory = true) {
    if (!views.includes(next) || next === 'results' && !current?.complete) return;
    view = next;
    for (const name of views) $('#' + name + '-view').hidden = name !== view;
    document.querySelectorAll('.replay-tabs [data-view]').forEach(button => {
      if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    if (writeHistory) {url.searchParams.set('view', view);history.replaceState(null, '', url);}
    if (view === 'marimo') notebook();
  }
  const valueOf = (row, side) => row[side === 'architecture' ? 'val_loss' : 'step_time_ms'];
  const phase = row => row.phase === 'finals' ? 'Finals' : row.phase === 'hyperparam' ? 'Hyperparameters' : row.phase === 'architecture' ? 'Architecture' : 'Kernel';
  function plot(rows, side, baseline = null, budget = null) {
    const measured = rows.filter(r => Number.isFinite(valueOf(r, side)));
    if (!measured.length && !Number.isFinite(baseline)) return '';
    const values = measured.map(r => valueOf(r, side)).concat(Number.isFinite(baseline) ? [baseline] : []);
    const low = Math.min(...values) * .95, high = Math.max(...values) * 1.05, range = high - low || 1;
    const y = value => 18 + (high - value) / range * 150;
    const maxOrdinal = Math.max(1, ...rows.map(r => r.ordinal));
    const points = measured.map(r => `<circle cx="${82 + (r.ordinal - 1) * 350 / Math.max(1, maxOrdinal - 1)}" cy="${y(valueOf(r, side))}" r="4" fill="${r.accepted ? 'var(--success)' : 'var(--warning)'}"><title>${esc(phase(r))} #${r.ordinal}: ${fmt(valueOf(r, side))}</title></circle>`).join('');
    return `<svg viewBox="0 0 470 195" role="img" aria-label="${side === 'architecture' ? 'Validation loss' : 'Training-step milliseconds'}${budget ? ' at ' + budget + ' seconds' : ''}">${[0,.5,1].map(t => `<line x1="80" x2="455" y1="${18+t*150}" y2="${18+t*150}" stroke="var(--line)"/><text x="0" y="${22+t*150}" fill="var(--muted)" font-size="10">${fmt(high-t*range)}</text>`).join('')}${Number.isFinite(baseline) ? `<path d="M80 ${y(baseline)} H455" stroke="var(--muted)" stroke-dasharray="5 5"><title>Baseline: ${fmt(baseline)}</title></path>` : ''}${points}<text x="80" y="192" fill="var(--muted)" font-size="10">Candidate evaluations →</text></svg>`;
  }
  function card(side, final = false) {
    const mode = current.modes[side], result = mode.result, architecture = side === 'architecture';
    const base = result[architecture ? 'baseline_val_loss' : 'baseline_ms'];
    const candidate = result[architecture ? 'candidate_val_loss' : 'candidate_ms'];
    const comparison = base > 0 && Number.isFinite(candidate);
    const floor = architecture && project.caveat === 'loss-floor';
    const gain = comparison ? 100 * (1 - candidate / base) : null;
    let charts = '';
    if (final && comparison) {
      const max = Math.max(base, candidate), width = candidate / max * 100;
      charts = `<div class="comparison-label"><span>Baseline</span><span>${fmt(base)}</span></div><div class="result-bar"><span style="width:${base/max*100}%"></span></div><div class="comparison-label"><span>Candidate</span><span>${fmt(candidate)}</span></div><div class="result-bar improved"><span style="width:${width}%"></span></div>`;
    } else if (architecture) {
      const budgets = [...new Set(mode.candidates.filter(r => Number.isFinite(r.val_loss)).map(r => r.train_secs).concat(comparison ? [result.final_budget_s] : []))].sort((a,b) => b-a);
      charts = budgets.map((b,i) => {const graph = plot(mode.candidates.filter(r => r.train_secs === b), side, b === result.final_budget_s ? base : null, b);return i ? `<details class="screening-chart"><summary>${b}s screening</summary>${graph}</details>` : `<p class="caption">${b}s training · held-out validation loss</p>${graph}`;}).join('');
    } else charts = plot(mode.candidates, side, base);
    if (!charts) charts = `<div class="empty">${mode.status === 'waiting' ? 'Waiting for evaluation' : mode.status === 'running' ? 'Evaluating candidates…' : 'No accepted ' + side + ' result'}</div>`;
    return `<article class="metric-card"><h3>${architecture ? 'Architecture' : 'Kernels'}</h3>${comparison ? `<div class="numbers"><div><small>Baseline</small><strong>${fmt(base)}</strong></div><div><small>Candidate</small><strong>${fmt(candidate)}</strong></div></div><p class="gain">${floor ? 'Loss floor reached' : gain.toFixed(3) + '% lower ' + (architecture ? 'validation loss' : 'step time')}</p>` : ''}${charts}${comparison ? `<p class="caption">${architecture ? 'Equal ' + result.final_budget_s + 's training budget' : 'Full training step · milliseconds'}</p>` : ''}${architecture && project.caveat ? `<p class="caption caveat">${floor ? 'Synthetic-task loss floor; downstream quality not established.' : 'Near-zero synthetic loss; relative reduction is sensitive to scale.'}</p>` : ''}</article>`;
  }
  function detail() {
    if (!selected) return;
    const [side, ordinal] = selected.split(':');
    const row = current.modes[side].candidates.find(r => r.ordinal === Number(ordinal));
    if (!row) {$('#candidate-detail').hidden = true;return;}
    $('#candidate-title').textContent = phase(row) + ' · candidate ' + ordinal;
    const fields = [['State', row.state], ['Generation', row.generation], ['Training budget', row.train_secs == null ? null : row.train_secs + 's'], ['Validation loss', row.val_loss == null ? null : fmt(row.val_loss)], ['Step time', row.step_time_ms == null ? null : fmt(row.step_time_ms) + ' ms'], ['Gate', row.gate_reached], ['Compile check', row.compile_ok == null ? null : row.compile_ok ? 'Passed' : 'Failed'], ['Correctness check', row.correct_ok == null ? null : row.correct_ok ? 'Passed' : 'Not passed'], ['Model', current.modes[side].model]];
    $('#candidate-metrics').innerHTML = fields.filter(([,v]) => v != null).map(([k,v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
    $('#candidate-detail').hidden = false;
  }
  function exploration() {
    const focused = document.activeElement?.dataset?.candidate;
    $('#exploration-graph').innerHTML = ['kernel','architecture'].map(side => {
      const mode = current.modes[side], groups = new Map();
      for (const row of mode.candidates) {const key = row.phase === 'finals' ? 'Finals' : 'Generation ' + row.generation;if (!groups.has(key)) groups.set(key, []);groups.get(key).push(row);}
      return `<section class="generation-lane"><div class="lane-heading"><h3>${side === 'architecture' ? 'Architecture' : 'Kernel'} search</h3><small>${esc(mode.model)}</small></div><div class="generation-track">${[...groups].map(([name,rows]) => `<div class="generation-column"><span>${esc(name)}</span>${rows.map(r => `<button class="candidate-node ${r.state}" data-candidate="${side}:${r.ordinal}" aria-label="${esc(phase(r))} candidate ${r.ordinal}: ${r.state}"><span>${esc(phase(r))} ${r.ordinal}<br><small>${Number.isFinite(valueOf(r,side)) ? fmt(valueOf(r,side)) + (side === 'kernel' ? ' ms' : ' loss') : r.state === 'running' ? 'Evaluating…' : 'Check failed'}</small></span><span aria-hidden="true">${r.state === 'running' ? '◌' : r.accepted ? '✓' : '×'}</span></button>`).join('')}</div>`).join('') || `<span class="quiet">${mode.status === 'waiting' ? 'Waiting for evaluation' : mode.status === 'running' ? 'Preparing evaluation…' : 'Stopped before candidate measurement'}</span>`}</div></section>`;
    }).join('');
    if (focused) {const button = [...document.querySelectorAll('[data-candidate]')].find(b => b.dataset.candidate === focused);button?.focus({preventScroll:true});}
    detail();
  }
  function notebook() {
    const frame = $('#project-notebook');
    frame.hidden = !current?.complete;
    $('#notebook-progress').hidden = !!current?.complete;
    $('#notebook-source').hidden = !current?.complete;
    if (current?.complete && view === 'marimo' && !frame.getAttribute('src')) frame.src = project.notebook;
  }
  function render() {
    elapsed = Math.max(0, Date.now() - startedAt);
    current = ProjectReplay.snapshot(project, elapsed);
    $('#replay-progress>span').style.width = current.progress * 100 + '%';
    $('#replay-progress').setAttribute('aria-valuenow', Math.round(current.progress * 100));
    const frameKey = JSON.stringify({complete:current.complete,modes:current.modes});
    if (frameKey === lastFrame) return;
    lastFrame = frameKey;
    const state = current.complete ? 'Run complete' : Object.values(current.modes).some(m => m.status === 'running') ? 'Evaluating candidates' : elapsed < 1000 ? 'Preparing run' : 'Finalizing results';
    $('#run-state').textContent = state;
    document.querySelectorAll('[data-view="results"]').forEach(b => {b.disabled = !current.complete;});
    exploration();
    const charts = card('architecture') + card('kernel');
    for (const id of ['dashboard-metrics','wandb-charts','notebook-progress','aria-charts']) $('#' + id).innerHTML = charts;
    $('#final-metrics').innerHTML = current.complete ? card('architecture', true) + card('kernel', true) : '';
    const rows = Object.entries(current.modes).flatMap(([side,mode]) => mode.candidates.filter(r => r.state !== 'running').map(r => ({...r, side})));
    $('#evaluation-table').innerHTML = `<table><thead><tr><th>Search</th><th>Candidate</th><th>Budget</th><th>Measurement</th><th>Gate</th><th>Result</th></tr></thead><tbody>${rows.map(r => `<tr><td>${esc(r.side)}</td><td>${r.ordinal}</td><td>${r.train_secs ? r.train_secs + 's' : '—'}</td><td>${fmt(valueOf(r,r.side))}${Number.isFinite(r.step_time_ms) ? ' ms' : ''}</td><td>${r.gate_reached ?? '—'}</td><td class="${r.accepted ? 'accepted' : 'rejected'}">${r.accepted ? 'Accepted' : 'Not accepted'}</td></tr>`).join('')}</tbody></table>`;
    const miniSide = current.modes.architecture.candidates.some(r => Number.isFinite(r.val_loss)) ? 'architecture' : 'kernel';
    const miniMode = current.modes[miniSide], miniBudget = miniMode.result.final_budget_s || miniMode.candidates.find(r => Number.isFinite(r.val_loss))?.train_secs;
    const miniRows = miniMode.candidates.filter(r => miniSide === 'kernel' || r.train_secs === miniBudget);
    const mini = plot(miniRows, miniSide, miniMode.result[miniSide === 'architecture' ? 'baseline_val_loss' : 'baseline_ms'], miniBudget);
    for (const id of ['wandb-mini','marimo-mini','aria-mini']) {
      const node = $('#' + id);node.setAttribute('viewBox', '0 0 470 195');
      node.setAttribute('aria-label', miniSide === 'architecture' ? 'Validation loss measurements' : 'Training-step measurements');
      node.innerHTML = mini.replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '');
    }
    $('#aria-prompt').value = `Analyze ${project.repo} using ${project.wandb_url}. Compare kernel step time and held-out validation loss at matching training budgets. Explain accepted changes and failed checks. Data: deterministic synthetic batches. Do not infer downstream quality from near-zero losses.\n\n${JSON.stringify(Object.fromEntries(Object.entries(current.modes).map(([side,m])=>[side,m.result])))}`;
    notebook();
    if (current.complete && view === 'exploration') selectView('dashboard');
  }
  document.addEventListener('click', event => {
    const navigation = event.target.closest('[data-view]');
    if (navigation && !navigation.disabled) selectView(navigation.dataset.view);
    const candidate = event.target.closest('[data-candidate]');
    if (candidate) {selected = candidate.dataset.candidate;detail();}
  });
  $('#close-candidate').onclick = () => {selected = null;$('#candidate-detail').hidden = true;};
  document.addEventListener('keydown', event => {if (event.key === 'Escape') $('#close-candidate').click();});
  $('#restart-run').onclick = () => {startedAt = Date.now();storeStart();lastFrame = '';selected = null;$('#candidate-detail').hidden = true;$('#project-notebook').removeAttribute('src');render();selectView('exploration');};
  $('#copy-analysis').onclick = async () => {try {await navigator.clipboard.writeText($('#aria-prompt').value);$('#copy-status').textContent = 'Copied';} catch {$('#aria-prompt').focus();$('#aria-prompt').select();$('#copy-status').textContent = 'Select and copy the prompt';}};
  try {
    const response = await fetch('assets/project-replays.json');
    if (!response.ok) throw Error('Run evidence unavailable.');
    const data = await response.json();project = data.projects.find(p => p.repo === repo);
    if (!project) throw Error('Choose a project from the homepage.');
    if (url.searchParams.get('start') !== '1') {try {const saved = Number(sessionStorage.getItem(storageKey));if (Number.isFinite(saved) && saved > 0 && saved <= Date.now()) startedAt = saved;} catch {}}
    storeStart();url.searchParams.delete('start');history.replaceState(null, '', url);
    $('#project-title').textContent = project.name;document.title = project.name + ' · Replay run · Top-Kernel';
    $('#project-repo').textContent = project.repo;safeLink('#project-repo','https://github.com/' + project.repo);
    safeLink('#evidence-link', project.evidence_url);safeLink('#wandb-source', project.wandb_url);safeLink('#aria-source', project.wandb_url);
    $('#notebook-source').href = project.notebook;
    $('#restart-run').disabled = false;
    render();if (view === 'results' && !current.complete) view = 'dashboard';selectView(view);
    timer = setInterval(render, 250);
  } catch (error) {$('#project-title').textContent = 'Run unavailable';$('#replay-error').textContent = error.message;$('#replay-error').hidden = false;$('#restart-run').disabled = true;document.querySelectorAll('[data-view]').forEach(b => {b.disabled = true;});}
  window.addEventListener('pagehide', () => clearInterval(timer));
  window.addEventListener('pageshow', event => {if (event.persisted && project) {render();clearInterval(timer);timer = setInterval(render, 250);}});
})();
