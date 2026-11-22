// Shared neural training loop. Runs the same way inside the Web Worker (train-worker.js) and,
// as a fallback, on the main thread. It only touches the engine it is given and reports back
// through hooks, so it has no DOM or app dependencies.

const defaultYield = () => new Promise(resolve => setTimeout(resolve, 0));

/**
 * Trains `engine` (TinyNeuralLM or TinyAttentionLM) on `text` with mini-batches.
 *
 * cfg: epochs, learningRate, lrFloor (fraction of the rate kept at the end of the cosine schedule),
 *      augment, maxExamples, vocabLimit, validate (hold out 10% and keep the best checkpoint),
 *      earlyStop, patience, batchSize (default 8 or 16 by dataset size), lrBoost (default 1.5; batch gradients are averaged),
 *      contextSize, hiddenDim, embDim.
 * hooks: shouldStop() -> bool, yieldNow() -> Promise, onProgress(info), yieldEveryMs.
 *
 * Returns { completed, lossPoints, stopped, earlyStopped, examples }. The caller decides what to
 * do with the engine afterwards (the engine already holds the trained weights).
 */
export async function trainNeural(engine, text, cfg = {}, hooks = {}) {
  const shouldStop = hooks.shouldStop || (() => false);
  const yieldNow = hooks.yieldNow || defaultYield;
  const onProgress = hooks.onProgress || (() => {});
  const yieldEveryMs = hooks.yieldEveryMs || 25;
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  const epochs = Math.max(1, cfg.epochs | 0);
  const baseLr = cfg.learningRate ?? 0.08;
  const lrFloor = cfg.lrFloor ?? 0.2;
  const lrBoost = cfg.lrBoost ?? 1.5;
  const validate = cfg.validate !== false;
  const earlyStop = cfg.earlyStop !== false;
  const patienceLimit = cfg.patience ?? 6;

  if (cfg.contextSize) engine.contextSize = cfg.contextSize;
  if (cfg.hiddenDim) engine.hiddenDim = cfg.hiddenDim;
  if (cfg.embDim) engine.embDim = cfg.embDim;

  const data = engine.prepareDataset(text, cfg.augment !== false, cfg.maxExamples ?? Infinity, cfg.vocabLimit ?? Infinity);
  const { X, Y } = data;
  const n = data.count;
  const all = new Int32Array(n);
  for (let i = 0; i < n; i++) all[i] = i;

  let valIdx = new Int32Array(0);
  let trainIdx = all;
  if (validate && n >= 2) {
    const v = [], t = [];
    for (let i = 0; i < n; i++) (i % 10 === 0 ? v : t).push(i);
    valIdx = Int32Array.from(v);
    trainIdx = t.length ? Int32Array.from(t) : all;
  }
  const trainN = trainIdx.length;
  const order = Int32Array.from(trainIdx);
  // Small datasets get smaller batches so each epoch still makes enough updates.
  const batchSize = Math.max(1, (cfg.batchSize | 0) || (trainN < 600 ? 8 : 16));

  const lossPoints = [];
  let completed = 0, bestVal = Infinity, patience = 0, bestWeights = null;
  let stopped = false, earlyStopped = false;
  let lastYield = now();
  let lastReport = 0;

  const snapshot = () => ({ E: engine.E.slice(), W1: engine.W1.slice(), b1: engine.b1.slice(), W2: engine.W2.slice(), b2: engine.b2.slice() });
  const previewLoss = () => engine.evalLoss(X, Y, trainIdx, 0, Math.min(32, trainN));
  const validationLoss = () => (valIdx.length ? engine.evalLoss(X, Y, valIdx, 0, valIdx.length) : previewLoss());

  for (let epoch = 1; epoch <= epochs; epoch++) {
    if (shouldStop()) { stopped = true; break; }
    const lr = baseLr * lrBoost * (lrFloor + (1 - lrFloor) * 0.5 * (1 + Math.cos(Math.PI * epoch / epochs)));

    // Fisher-Yates shuffle each epoch.
    for (let i = trainN - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    let epochLoss = 0;
    for (let start = 0; start < trainN; start += batchSize) {
      const end = Math.min(trainN, start + batchSize);
      epochLoss += engine.trainBatch(X, Y, order, start, end, lr);
      if (!Number.isFinite(epochLoss)) {
        throw new Error('Training diverged (the loss became NaN). Try a lower learning rate or a smaller hidden size.');
      }
      const t = now();
      if (t - lastYield >= yieldEveryMs) {
        await yieldNow();
        lastYield = now();
        if (shouldStop()) { stopped = true; break; }
        if (lastYield - lastReport > 120) {
          lastReport = lastYield;
          onProgress({ epoch, epochs, fraction: ((epoch - 1) + end / trainN) / epochs, lossPoints, partial: true });
        }
      }
    }
    if (stopped) break;

    const avg = epochLoss / Math.max(1, trainN);
    const val = validationLoss();
    lossPoints.push(avg);
    completed++;
    lastReport = now();
    onProgress({ epoch, epochs, fraction: epoch / epochs, avg, val, lossPoints, partial: false });

    if (validate) {
      if (val < bestVal - 0.0005) {
        bestVal = val;
        patience = 0;
        bestWeights = snapshot();
      } else {
        patience++;
      }
      if (earlyStop && patience >= patienceLimit) { earlyStopped = true; break; }
    }
    await yieldNow();
    lastYield = now();
  }

  // Keep the best validation checkpoint, unless the run was stopped by the user.
  if (bestWeights && !stopped) Object.assign(engine, bestWeights);
  engine.isTrained = true;
  engine.rawCorpus = text;

  return { completed, lossPoints, stopped, earlyStopped, examples: n };
}
