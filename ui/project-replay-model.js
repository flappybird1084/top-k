/* Playback only: reveal immutable public measurements; never dispatch a job. */
(function (root) {
  'use strict';
  const duration = 28000, lead = 1000, finish = 2000;
  function schedule(project) {
    const events = [];
    for (const side of ['kernel', 'architecture']) {
      const rows = project.modes[side]?.candidates || [];
      if (rows.length) rows.forEach(row => events.push({side, row}));
      else events.push({side, row: null});
    }
    const slot = (duration - lead - finish) / events.length;
    return events.map((event, i) => ({...event, start: lead + slot * i, end: lead + slot * (i + 1)}));
  }
  function snapshot(project, elapsed) {
    const events = schedule(project), complete = elapsed >= duration;
    const modes = {};
    for (const side of ['kernel', 'architecture']) {
      const source = project.modes[side], own = events.filter(e => e.side === side);
      const started = elapsed >= own[0].start, done = elapsed >= own.at(-1).end;
      const rows = own.filter(e => e.row && elapsed >= e.start).map(e => {
        if (elapsed < e.end) return {ordinal: e.row.ordinal, generation: e.row.generation,
          phase: e.row.phase, train_secs: e.row.train_secs, state: 'running'};
        let state = 'failed';
        if (e.row.accepted) state = 'improved';
        else if (e.row.correct_ok) state = 'dropped';
        return {...e.row, state};
      });
      let status = 'waiting';
      if (done) status = source.status === 'done' ? 'complete' : 'failed';
      else if (started) status = 'running';
      modes[side] = {status,
        // Calibration precedes the candidate search; final candidate outcomes stay gated below.
        baseline: Object.fromEntries(Object.entries(source.result).filter(([key]) => ['baseline_ms', 'baseline_val_loss', 'final_budget_s'].includes(key))),
        candidates: rows, result: done ? {...source.result} : {}, model: source.model,
        data_source: source.data_source, commit: source.commit};
    }
    return {complete, progress: Math.max(0, Math.min(1, elapsed / duration)), modes};
  }
  // Keep every recorded training budget separate; never interpolate training history.
  function series(mode, side) {
    const metric = side === 'architecture' ? 'val_loss' : 'step_time_ms';
    const rows = mode.candidates.filter(row => Number.isFinite(row[metric]));
    const baseline = mode.baseline || mode.result;
    const budgets = side === 'kernel' ? [null] : [...new Set(rows.map(row => row.train_secs).concat(Number.isFinite(baseline.baseline_val_loss) ? [baseline.final_budget_s] : []))].sort((a,b) => a-b);
    return budgets.map(budget => ({budget, metric,
      baseline: side === 'kernel' ? baseline.baseline_ms : budget === baseline.final_budget_s ? baseline.baseline_val_loss : null,
      rows: rows.filter(row => side === 'kernel' || row.train_secs === budget)
    })).filter(group => group.rows.length || Number.isFinite(group.baseline));
  }
  function runPair(project, elapsed) {
    const current = snapshot(project, elapsed), pair = {};
    for (const side of ['architecture', 'kernel']) {
      const mode = current.modes[side];
      const rows = mode.candidates.map(row => ({...row, id: row.id || row.ordinal,
        display_id: row.id || row.ordinal,
        strategy: row.strategy || `${row.phase || 'Kernel'} evaluation ${row.ordinal}`}));
      pair[side] = {repo:'https://github.com/' + project.repo, data:mode.data_source,
        agent_model:mode.model, subagent_model:mode.model,
        status:mode.status === 'complete' ? 'complete' : mode.status,
        rows, message:mode.status, architecture:{candidates: side === 'architecture' ? rows : []},
        candidates:side === 'kernel' ? rows : []};
      if (current.complete && mode.result.measured) {
        const metric = side === 'architecture' ? 'val_loss' : 'step_time_ms';
        const value = mode.result[side === 'architecture' ? 'candidate_val_loss' : 'candidate_ms'];
        const winner = rows.find(row => row.accepted && row[metric] === value && (side === 'kernel' || row.train_secs === mode.result.final_budget_s));
        pair[side].final_result = {title:'Final ' + side, parent_id:winner?.id,
          metric: Number(value).toPrecision(6) + (side === 'kernel' ? ' ms' : ' validation loss'),
          description:'Recorded final measurement', state: winner ? 'improved' : 'dropped',
          note:side === 'architecture' ? mode.result.final_budget_s + 's training' : 'Full training step'};
      }
    }
    return pair;
  }
  const api = {duration, schedule, snapshot, series, runPair};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ProjectReplay = api;
})(typeof window === 'undefined' ? globalThis : window);
