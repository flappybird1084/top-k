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
        return {...e.row, state: e.row.accepted ? 'improved' : e.row.correct_ok ? 'dropped' : 'failed'};
      });
      modes[side] = {status: !started ? 'waiting' : !done ? 'running' : source.status === 'done' ? 'complete' : 'failed',
        candidates: rows, result: done ? {...source.result} : {}, model: source.model,
        data_source: source.data_source, commit: source.commit};
    }
    return {complete, progress: Math.max(0, Math.min(1, elapsed / duration)), modes};
  }
  const api = {duration, schedule, snapshot};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ProjectReplay = api;
})(typeof window === 'undefined' ? globalThis : window);
