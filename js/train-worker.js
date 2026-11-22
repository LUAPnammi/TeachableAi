// Training Web Worker. Runs model training off the main thread so the UI stays smooth
// and the Stop button always responds. Imports the same engine code the app uses.
//
// Messages in:  { type: 'start', job } | { type: 'stop' }
// Messages out: { type: 'ready' } | { type: 'progress', info } | { type: 'done', result } | { type: 'error', message }

import { HybridNGramEngine, TinyAttentionLM, TinyNeuralLM } from './engines.js';
import { trainNeural } from './trainer.js';

let stopRequested = false;
let running = false;

const yieldNow = () => new Promise(resolve => setTimeout(resolve, 0));

self.onmessage = (event) => {
  const msg = event.data || {};
  if (msg.type === 'stop') { stopRequested = true; return; }
  if (msg.type === 'start' && !running) run(msg.job);
};

async function run(job) {
  running = true;
  stopRequested = false;
  try {
    const cfg = job.config || {};
    if (job.engineType === 'ngram') {
      const engine = new HybridNGramEngine();
      await engine.trainAsync(job.text, cfg.indexLimit || 50000, () => stopRequested);
      self.postMessage({ type: 'done', result: { kind: 'ngram', stopped: stopRequested, json: engine.toJSON() } });
      return;
    }

    const engine = job.engineType === 'attention'
      ? new TinyAttentionLM(cfg.contextSize, cfg.embDim, cfg.hiddenDim)
      : new TinyNeuralLM(cfg.contextSize, cfg.embDim, cfg.hiddenDim);

    const outcome = await trainNeural(engine, job.text, cfg, {
      shouldStop: () => stopRequested,
      yieldNow,
      yieldEveryMs: 30,
      onProgress: (info) => {
        // lossPoints is the same growing array each time; send a copy.
        self.postMessage({ type: 'progress', info: { ...info, lossPoints: info.lossPoints.slice() } });
      }
    });

    const state = engine.exportTrainedState();
    const transfer = [state.E.buffer, state.W1.buffer, state.b1.buffer, state.W2.buffer, state.b2.buffer];
    self.postMessage({ type: 'done', result: { kind: 'neural', ...outcome, state } }, transfer);
  } catch (err) {
    self.postMessage({ type: 'error', message: (err && err.message) || String(err) });
  } finally {
    running = false;
  }
}

self.postMessage({ type: 'ready' });
