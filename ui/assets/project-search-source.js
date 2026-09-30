'use strict';
// Only this public container installs the recorded source. Live pages keep their API path.
window.topkProjectReplay = (() => {
  const params = new URLSearchParams(location.search);
  const repo = params.get('project');
  const requestedStart = Number(params.get('started'));
  const start = Number.isFinite(requestedStart) && requestedStart > 0 && requestedStart <= Date.now() ? requestedStart : Date.now();
  let evidence;
  function load() {
    if (!evidence) evidence = fetch('project-replays.json', {cache:'no-store'}).then(async response => {
      if (!response.ok) throw Error('Run evidence unavailable');
      const project = (await response.json()).projects.find(p => p.repo === repo);
      if (!project) throw Error('Project unavailable');
      return project;
    }).catch(error => {evidence = null;throw error;});
    return evidence;
  }
  return {
    async pair() {return ProjectReplay.runPair(await load(), Math.max(0, Date.now() - start));},
    destination: '../replay.html?project=' + encodeURIComponent(repo) + '&view=dashboard'
  };
})();
