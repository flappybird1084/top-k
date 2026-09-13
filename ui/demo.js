'use strict';
// Recorded results never advance with the diagram playback clock.
function demoSnapshot(view){return structuredClone(window.RECORDED_RUNS[view==='architecture'?'architecture':'kernel']);}
