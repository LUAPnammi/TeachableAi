// Train Studio: training loop, corpus cleaning tools, pre-flight check, Corpus Library packs, undo.
// Methods here are copied onto AIApp in main.js, so `this` is the app.
import { CORPUS_LIBRARY } from './corpus-library.js';
import { HybridNGramEngine, TinyAttentionLM, TinyNeuralLM } from './engines.js';
import { trainNeural } from './trainer.js';

export class AppTraining {
  updateTrainingProfileNote() {
    const profile = document.getElementById('training-profile')?.value || 'balanced';
    const notes = {
      phone: 'Phone-safe limits: 240-token vocabulary, 450 examples, 24 hidden units, up to 20 epochs. CPU only; no GPU needed. Trains in a background worker.',
      balanced: 'Balanced limits: 520-token vocabulary, 1,100 examples, 48 hidden units, up to 45 epochs. CPU-first training.',
      large: 'Large CPU limits: 900-token vocabulary, 2,200 examples, 64 hidden units, up to 80 epochs. Avoid this profile on phones.'
    };
    const note = document.getElementById('training-profile-note');
    if (note) note.textContent = notes[profile] || notes.balanced;
  }

  createEngineForType(type) {
    return type === 'neural' ? new TinyNeuralLM() : (type === 'attention' ? new TinyAttentionLM() : new HybridNGramEngine());
  }

  /* ============ Background training (Web Worker with main-thread fallback) ============ */

  // Runs one training job and resolves with its result. Never touches the live model: the caller
  // applies the result with applyTrainingResult(). opts.primary marks the job the Stop button controls.
  runTrainingJob(job, onProgress = () => {}, opts = {}) {
    if (!this.trainingWorkerUnavailable && typeof Worker !== 'undefined') {
      return this._runInWorker(job, onProgress, opts).catch(err => {
        if (err && err.workerBoot) {
          this.trainingWorkerUnavailable = true;
          console.warn('Training worker unavailable, falling back to the main thread:', err.message);
          this.showToast('Background training is not available in this browser, so training runs on the main thread instead.');
          return this._runOnMainThread(job, onProgress, opts);
        }
        throw err;
      });
    }
    return this._runOnMainThread(job, onProgress, opts);
  }

  _runInWorker(job, onProgress, opts) {
    return new Promise((resolve, reject) => {
      let worker;
      try {
        worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
      } catch (err) {
        const e = new Error('Could not start the training worker: ' + (err.message || err));
        e.workerBoot = true;
        reject(e);
        return;
      }
      let booted = false, settled = false;
      if (opts.primary) this.trainingWorker = worker;
      const finish = (fn, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(bootTimer);
        clearTimeout(this._hardStopTimer);
        try { worker.terminate(); } catch (e) {}
        if (this.trainingWorker === worker) this.trainingWorker = null;
        fn(value);
      };
      const bootTimer = setTimeout(() => {
        if (booted) return;
        const e = new Error('The training worker did not start in time.');
        e.workerBoot = true;
        finish(reject, e);
      }, 15000);
      // Used by the Stop button if the worker does not answer in time.
      worker.abortJob = () => finish(resolve, { kind: 'aborted', aborted: true, stopped: true, completed: 0, lossPoints: [] });
      worker.onmessage = (ev) => {
        const m = ev.data || {};
        if (m.type === 'ready') { booted = true; clearTimeout(bootTimer); }
        else if (m.type === 'progress') { booted = true; try { onProgress(m.info); } catch (e) { console.warn(e); } }
        else if (m.type === 'done') finish(resolve, m.result);
        else if (m.type === 'error') finish(reject, new Error(m.message || 'Training failed'));
      };
      worker.onerror = (ev) => {
        if (ev && ev.preventDefault) ev.preventDefault();
        const e = new Error((ev && ev.message) || 'The training worker crashed.');
        if (!booted) e.workerBoot = true;
        finish(reject, e);
      };
      worker.postMessage({ type: 'start', job });
    });
  }

  async _runOnMainThread(job, onProgress, opts) {
    const cfg = job.config || {};
    const shouldStop = () => Boolean(opts.primary && this.stopRequested);
    if (job.engineType === 'ngram') {
      const engine = new HybridNGramEngine();
      await engine.trainAsync(job.text, cfg.indexLimit || 50000, shouldStop);
      return { kind: 'ngram', stopped: shouldStop(), json: engine.toJSON() };
    }
    const engine = job.engineType === 'attention'
      ? new TinyAttentionLM(cfg.contextSize, cfg.embDim, cfg.hiddenDim)
      : new TinyNeuralLM(cfg.contextSize, cfg.embDim, cfg.hiddenDim);
    const outcome = await trainNeural(engine, job.text, cfg, {
      shouldStop,
      yieldNow: () => new Promise(resolve => setTimeout(resolve, 0)),
      yieldEveryMs: 10,
      onProgress: info => onProgress({ ...info, lossPoints: info.lossPoints.slice() })
    });
    return { kind: 'neural', ...outcome, state: engine.exportTrainedState() };
  }

  // The Stop button: ask the worker to finish its current slice and hand back what it has learned.
  // If it has not answered after 3 seconds, the worker is terminated and the model stays as it was.
  requestStopTraining() {
    if (!this.isTraining) return;
    this.stopRequested = true;
    const statusText = document.getElementById('train-status-text');
    if (statusText) statusText.innerText = 'Stopping…';
    const worker = this.trainingWorker;
    if (worker) {
      worker.postMessage({ type: 'stop' });
      clearTimeout(this._hardStopTimer);
      this._hardStopTimer = setTimeout(() => { if (worker.abortJob) worker.abortJob(); }, 3000);
    }
  }

  // Copies a finished job's result into the model. Returns false when there is nothing to apply
  // (job aborted, or stopped before the first epoch finished), in which case the model is unchanged.
  applyTrainingResult(modelObj, result, text, { freshEngine = false, resetCounters = false, newType = null } = {}) {
    if (!result || result.aborted) return false;
    if (result.kind === 'neural' && !(result.completed > 0)) return false;
    let engine = modelObj.engine;
    if (freshEngine) {
      if (newType) modelObj.type = newType;
      engine = modelObj.engine = this.createEngineForType(modelObj.type);
    }
    if (resetCounters) { engine.lossHistory = []; engine.epochsTrained = 0; }
    if (result.kind === 'neural') {
      engine.applyTrainedState(result.state);
      for (const l of result.lossPoints) engine.lossHistory.push(l);
      engine.epochsTrained = (engine.epochsTrained || 0) + result.completed;
    } else {
      engine.fromJSON(result.json);
    }
    engine.isTrained = true;
    engine.rawCorpus = text;
    if (engine.memory) engine.memory.indexText(text);
    return true;
  }

  async startTraining() {
    if (this.isRestoring) this.isRestoring = false;
    if (this.isTraining) return;
    const inputText = document.getElementById('training-text').value.trim();
    if (!inputText) {
      this.showToast("Please enter text or load a pack from Corpus Library first!", "error");
      return;
    }

    const fromScratch = Boolean(document.getElementById('param-from-scratch')?.checked);
    const activeForMerge = this.getActiveModelObj();
    const text = fromScratch ? inputText : this.mergeCorpus(activeForMerge?.engine?.rawCorpus || '', inputText);
    document.getElementById('training-text').value = text;
    this.updateCorpusStats();

    if (text.length > 500000) {
      this.showToast('Merged corpus is over 500,000 characters. Split it into focused datasets for safer phone/CPU training.', 'error');
      return;
    }

    this.isTraining = true;
    this.stopRequested = false;

    document.getElementById('start-train-btn').disabled = true;
    document.getElementById('stop-train-btn').disabled = false;
    document.getElementById('train-status-dot').className = "w-2 h-2 rounded-full bg-amber-400 animate-ping";
    document.getElementById('train-status-text').innerText = fromScratch ? "Training fresh from scratch..." : "Training active model...";
    document.getElementById('validation-loss-val').innerText = 'N/A';

    try {
      await this._runTrainingFlow(text, fromScratch);
    } catch (err) {
      // Make sure the UI never stays locked after a failure, then let the caller report it.
      this.isTraining = false;
      this.stopRequested = false;
      document.getElementById('start-train-btn').disabled = false;
      document.getElementById('stop-train-btn').disabled = true;
      document.getElementById('train-status-dot').className = 'w-2 h-2 rounded-full bg-rose-500';
      document.getElementById('train-status-text').innerText = 'Training failed';
      throw err;
    }
  }

  async _runTrainingFlow(text, fromScratch) {
    const activeObj = this.getActiveModelObj();
    try { this.lastTrainingSnapshot = { modelId: activeObj.id, name: activeObj.name, json: activeObj.engine.toJSON({ weights: 'f32' }) }; } catch (snapErr) { this.lastTrainingSnapshot = null; }
    // Train Studio only offers "neural" and "ngram"; an Attention model stays an Attention model.
    let selectedEngineType = document.querySelector('input[name="engine-type"]:checked').value;
    if (selectedEngineType === 'neural' && activeObj.type === 'attention') selectedEngineType = 'attention';
    const swapEngine = activeObj.type !== selectedEngineType || fromScratch;

    const epochs = parseInt(document.getElementById('param-epochs').value);
    const baseLr = parseFloat(document.getElementById('param-lr').value);
    const contextSize = parseInt(document.getElementById('param-context').value);
    const hiddenDim = parseInt(document.getElementById('param-hidden').value);
    const augment = document.getElementById('param-augment').checked;
    const profile = document.getElementById('training-profile').value || 'balanced';

    let job, config = null;
    if (selectedEngineType === 'neural' || selectedEngineType === 'attention') {
      const profileLimits = {
        phone: { vocabLimit: 240, maxExamples: 450, hiddenCap: 24, epochCap: 20, profileLabel: 'Phone / low-memory CPU' },
        balanced: { vocabLimit: 520, maxExamples: 1100, hiddenCap: 48, epochCap: 45, profileLabel: 'Balanced CPU' },
        large: { vocabLimit: 900, maxExamples: 2200, hiddenCap: 64, epochCap: 80, profileLabel: 'Large CPU' }
      };
      const limits = profileLimits[profile] || profileLimits.balanced;
      config = {
        profile,
        profileLabel: limits.profileLabel,
        epochs: Math.min(epochs, limits.epochCap),
        learningRate: Math.min(baseLr, profile === 'phone' ? 0.09 : 0.16),
        contextSize: Math.min(16, Math.max(2, contextSize)),
        hiddenDim: Math.min(hiddenDim, limits.hiddenCap),
        embDim: 20,
        vocabLimit: limits.vocabLimit,
        maxExamples: limits.maxExamples,
        augment,
        validate: true,
        earlyStop: true,
        lrFloor: 0.2
      };
      if (config.epochs < epochs) this.showToast(`Profile limited training to ${config.epochs} epochs for device safety.`);
      if (config.hiddenDim < hiddenDim) this.showToast(`Hidden size limited to ${config.hiddenDim} for this hardware profile.`);
      document.getElementById('chart-max-epoch').innerText = 'Epoch ' + config.epochs;
      job = { engineType: selectedEngineType, text, config };
    } else {
      const indexLimit = profile === 'phone' ? 25000 : profile === 'large' ? 100000 : 50000;
      job = { engineType: 'ngram', text, config: { indexLimit } };
    }

    const onProgress = (info) => {
      document.getElementById('train-progress-bar').style.width = (info.fraction * 100) + '%';
      if (!info.partial) {
        document.getElementById('current-loss-val').innerText = info.avg.toFixed(4);
        document.getElementById('validation-loss-val').innerText = info.val.toFixed(4);
      }
      if (!this.stopRequested) {
        document.getElementById('train-status-text').innerText = `${config.profileLabel} · epoch ${Math.min(info.epoch, config.epochs)}/${config.epochs} · ${this.trainingWorkerUnavailable ? 'main thread' : 'background'}`;
      }
      this.drawLossChart(info.lossPoints);
    };

    const result = await this.runTrainingJob(job, onProgress, { primary: true });

    const applied = this.applyTrainingResult(activeObj, result, text, { freshEngine: swapEngine, newType: selectedEngineType });

    if (result.kind === 'ngram') {
      document.getElementById('train-progress-bar').style.width = this.stopRequested ? '0%' : '100%';
      document.getElementById('current-loss-val').innerText = 'Indexed';
      document.getElementById('validation-loss-val').innerText = 'N/A';
    } else if (applied) {
      if (result.lossPoints.length) this.drawLossChart(result.lossPoints);
      if (result.earlyStopped) document.getElementById('train-status-text').innerText += ' · early stop';
    }

    const stopped = this.stopRequested;
    this.isTraining = false;
    document.getElementById('start-train-btn').disabled = false;
    document.getElementById('stop-train-btn').disabled = true;
    document.getElementById('train-status-dot').className = "w-2 h-2 rounded-full bg-emerald-500";
    let statusMessage = stopped ? 'Training stopped safely.' : 'Training complete!';
    let toast = stopped ? 'Training stopped safely; completed progress was saved.' : "Model '" + activeObj.name + "' is trained and ready in Chat Studio.";
    if (stopped && !applied) {
      statusMessage = 'Training stopped. Model left unchanged.';
      toast = 'Training stopped before the first epoch finished, so the model was left as it was.';
    }
    document.getElementById('train-status-text').innerText = statusMessage;
    this.stopRequested = false;

    if (applied) {
      this.saveModelsToStorage();
      this.updateStatsDisplay();
      this.refreshMemoryInspector();
      if (this.nudgeBackup) this.nudgeBackup();
      // If this model has a held-out test set, re-score it right away so progress is measurable.
      if (this.rescoreAfterTraining) this.rescoreAfterTraining(activeObj).catch(err => console.warn('Re-scoring failed:', err));
    }
    this.showToast(toast);
  }

  drawLossChart(lossPoints) {
    if (lossPoints.length < 2) return;
    const svg = document.getElementById('loss-polyline');
    const maxLoss = Math.max(...lossPoints, 1.0);
    const minLoss = Math.min(...lossPoints, 0.0);
    const range = Math.max(maxLoss - minLoss, 0.001);

    const pts = lossPoints.map((l, i) => {
      const x = (i / (lossPoints.length - 1)) * 100;
      const y = 90 - ((l - minLoss) / range) * 80;
      return x.toFixed(1) + ',' + y.toFixed(1);
    }).join(' ');

    svg.setAttribute('points', pts);
  }

  handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      document.getElementById('training-text').value = evt.target.result;
      this.updateCorpusStats();
      this.showToast('Loaded ' + file.name);
    };
    reader.readAsText(file);
  }

  updateCorpusStats() {
    const text = document.getElementById('training-text').value;
    const tokens = text.split(/\s+/).filter(Boolean).length;
    const qaCount = (text.match(/^user\s*:/gim) || []).length;
    document.getElementById('corpus-stats').innerText = 'Tokens: ' + tokens + ' | Q&A Pairs: ' + qaCount;
    const lines = text.split(/\r?\n/).map(x => x.trim()).filter(Boolean);
    const duplicates = lines.length - new Set(lines).size;
    const aiCount = (text.match(/^(ai|assistant|bot)\s*:/gim) || []).length;
    const quality = document.getElementById('corpus-quality');
    if (quality) quality.textContent = `${text.length.toLocaleString()} chars · ${lines.length.toLocaleString()} lines · ${duplicates} duplicate lines · ${Math.abs(qaCount - aiCount)} unmatched user/AI tag count`;
  }

  cleanCorpusDuplicates() {
    const area = document.getElementById('training-text');
    const lines = area.value.replace(/\r\n/g, '\n').split('\n');
    const seen = new Set(), cleaned = [];
    for (const line of lines) {
      const key = line.trim().toLowerCase();
      if (key && seen.has(key)) continue;
      if (key) seen.add(key);
      cleaned.push(line);
    }
    area.value = cleaned.join('\n').replace(/\n{3,}/g, '\n\n').trim();
    this.updateCorpusStats();
    this.showToast('Removed exact duplicate lines. Review the corpus before training.');
  }

  autoFormatCorpus() {
    const area = document.getElementById('training-text');
    let text = area.value;
    if (!text.trim()) return;
    text = text
      .replace(/\r\n/g, '\n')
      .replace(/^(user|human|q)\s*:\s*/gim, 'user: ')
      .replace(/^(ai|bot|assistant|a)\s*:\s*/gim, 'ai: ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    area.value = text;
    this.updateCorpusStats();
    this.showToast("Formatted dialogue tags and spacing!");
  }

  renderCorpusLibrary(category) {
    const grid = document.getElementById('library-grid');
    const items = category === 'all'
      ? CORPUS_LIBRARY
      : CORPUS_LIBRARY.filter(item => item.category === category);

    grid.innerHTML = items.map(item => `
      <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-slate-700 flex flex-col justify-between gap-3 transition">
        <div class="space-y-2">
          <div class="flex items-center justify-between gap-2">
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-pink-500/15 text-pink-300 border border-pink-500/30 font-mono">${item.badge}</span>
            <span class="text-[10px] text-slate-500 font-mono">~${item.text.split(/\s+/).length} words</span>
          </div>
          <h3 class="text-sm font-bold text-white">${item.title}</h3>
          <p class="text-xs text-slate-400 leading-relaxed">${item.description}</p>
          <pre class="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80 text-[11px] font-mono text-slate-300 max-h-32 overflow-y-auto whitespace-pre-wrap">${this.escapeHTML(item.text)}</pre>
        </div>

        <div class="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800/80">
          <button onclick="app.copyLibraryPack('${item.id}')" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition">
            <i class="fa-regular fa-copy"></i> Copy
          </button>
          <button onclick="app.appendLibraryPack('${item.id}', false)" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-brand-300 text-xs font-medium flex items-center gap-1.5 transition">
            <i class="fa-solid fa-plus"></i> Append to Trainer
          </button>
          <button onclick="app.appendLibraryPack('${item.id}', true)" class="ml-auto px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold flex items-center gap-1.5 transition">
            <i class="fa-solid fa-bolt"></i> Load & Train
          </button>
        </div>
      </div>
    `).join('');
  }

  copyLibraryPack(id) {
    const pack = CORPUS_LIBRARY.find(p => p.id === id);
    if (!pack) return;
    navigator.clipboard.writeText(pack.text);
    this.showToast('Copied "' + pack.title + '" to clipboard!');
  }

  appendLibraryPack(id, autoTrain = false) {
    const pack = CORPUS_LIBRARY.find(p => p.id === id);
    if (!pack) return;
    const area = document.getElementById('training-text');
    area.value = this.mergeCorpus(area.value, pack.text);
    this.updateCorpusStats();
    this.switchTab('train-tab');
    if (autoTrain) {
      this.startTraining();
    } else {
      this.showToast('Added "' + pack.title + '" to Training Studio!');
    }
  }

  /* ============ Corpus pre-flight check, auto-fix, CSV export, undo training ============ */
  runCorpusPreflight() {
    const area = document.getElementById('training-text');
    const text = area.value.replace(/\r\n/g, '\n').trim();
    if (!text) { this.showToast('The corpus is empty; nothing to check.', 'error'); return 'The corpus is empty; nothing to check.'; }
    const blocks = text.split(/\n\s*\n/).map(b => b.trim()).filter(Boolean);
    const keyOf = b => { const m = b.match(/^user\s*:\s*(.+?)$/im); return m ? m[1].trim().toLowerCase().replace(/\s+/g, ' ') : null; };
    const pairBlocks = blocks.filter(b => /^user\s*:/im.test(b) && /\bai\s*:/im.test(b));
    const userOnlyBlocks = blocks.filter(b => /^user\s*:/im.test(b) && !/\bai\s*:/im.test(b));
    const exactDupes = blocks.length - new Set(blocks.map(b => b.toLowerCase().replace(/[\s.:;]/g, ' ').replace(/\s+/g, ' ').trim())).size;
    const byQ = new Map();
    for (const b of pairBlocks) { const k = keyOf(b); if (k) { if (!byQ.has(k)) byQ.set(k, []); byQ.get(k).push(b); } }
    let conflicts = 0, conflictSamples = [];
    for (const [q, bs] of byQ) {
      const ans = new Set(bs.map(b => (b.match(/^\s*(?:ai|bot|assistant|a)\s*:\s*(.+)$/im) || [])[1] || '').map(x => x.trim().toLowerCase().replace(/\s+/g, ' ')));
      if (ans.size > 1) { conflicts++; if (conflictSamples.length < 3) conflictSamples.push(q); }
    }
    const shortAnswers = pairBlocks.filter(b => { const m = b.match(/^\s*(?:ai|bot|assistant|a)\s*:\s*(.+)$/im); return !m || m[1].trim().split(/\s+/).filter(Boolean).length < 3; }).length;
    const paragraphs = this.parseParagraphs(text).length;
    const qualityEl = document.getElementById('corpus-quality');
    const problems = exactDupes + conflicts + shortAnswers + userOnlyBlocks;
    const report = [
      problems ? 'Pre-flight found ' + problems + ' issue(s):' : 'Pre-flight check passed: no blocking issues found.',
      exactDupes ? '\n· ' + exactDupes + ' duplicate block(s)' : '',
      conflicts ? '\n· ' + conflicts + ' question(s) with conflicting answers' + (conflictSamples.length ? ' (e.g. "' + conflictSamples.join('", "') + '")' : '') : '',
      shortAnswers ? '\n· ' + shortAnswers + ' pair(s) with empty or very short answers' : '',
      userOnlyBlocks ? '\n· ' + userOnlyBlocks + ' user line(s) with no answer' : '',
      '\n\nTotals: ' + blocks.length + ' blocks · ' + pairBlocks.length + ' Q&A pairs · ' + paragraphs + ' paragraph(s)'
    ].filter(Boolean).join('');
    if (qualityEl) qualityEl.textContent = 'Pre-flight: ' + pairBlocks.length + ' pairs · ' + exactDupes + ' dupes · ' + conflicts + ' conflicts · ' + shortAnswers + ' short answers · ' + paragraphs + ' paragraphs';
    this.showToast(problems ? 'Pre-flight found ' + problems + ' issue(s). Use Auto-fix, then review.' : 'Pre-flight check passed.');
    return report;
  }

  autoFixCorpus() {
    const area = document.getElementById('training-text');
    const text = area.value.replace(/\r\n/g, '\n').trim();
    if (!text) { this.showToast('The corpus is empty; nothing to fix.', 'error'); return 'The corpus is empty; nothing to fix.'; }
    const blocks = text.split(/\n\s*\n/).map(b => b.trim()).filter(Boolean);
    const keyOf = b => { const m = b.match(/^user\s*:\s*(.+?)$/im); return m ? m[1].trim().toLowerCase().replace(/\s+/g, ' ') : null; };
    const seenExact = new Set();
    const latestByQ = new Map();
    const keptNonQA = [];
    let removedDupes = 0, resolvedConflicts = 0;
    for (const b of blocks) {
      const key = b.toLowerCase().replace(/[\s.:;]/g, ' ').replace(/\s+/g, ' ').trim();
      if (seenExact.has(key)) { removedDupes++; continue; }
      seenExact.add(key);
      const k = keyOf(b);
      if (k) {
        if (latestByQ.has(k)) resolvedConflicts++;
        latestByQ.set(k, b);
      } else {
        keptNonQA.push(b);
      }
    }
    const fixedText = ([...latestByQ.values(), ...keptNonQA]).join('\n\n');
    if (fixedText === text && !removedDupes && !resolvedConflicts) {
      this.showToast('Auto-fix found nothing more to clean; exact duplicate lines were already removed.');
      return 'Auto-fix found nothing more to clean.';
    }
    area.value = fixedText;
    this.updateCorpusStats();
    const msg = 'Auto-fix: removed ' + removedDupes + ' duplicate block(s), kept the newest answer for ' + resolvedConflicts + ' conflicting question(s). Review, then train.';
    this.showToast(msg);
    return msg;
  }

  exportCorpusCSV() {
    const pairs = this.parseDialoguePairs(document.getElementById('training-text').value);
    if (!pairs.length) { this.showToast('No user:/ai: pairs found to export.', 'error'); return; }
    const esc = v => '"' + String(v).replace(/"/g, '""') + '"';
    const csv = 'question,answer\n' + pairs.map(p => esc(p.q) + ',' + esc(p.a)).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (this.getActiveModelObj()?.name || 'corpus').replace(/[^\w-]+/g, '_').toLowerCase() + '-dataset.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 3000);
    this.showToast('Exported ' + pairs.length + ' Q&A pairs as CSV.');
  }

  undoLastTraining() {
    const snap = this.lastTrainingSnapshot;
    if (!snap) { this.showToast('No training run to undo yet.', 'error'); return; }
    if (this.isTraining) { this.showToast('Wait for training to finish first.', 'error'); return; }
    const m = this.models.find(x => x.id === snap.modelId);
    if (!m) { this.showToast('That model no longer exists.', 'error'); return; }
    try {
      const eng = m.type === 'neural' ? new TinyNeuralLM() : (m.type === 'attention' ? new TinyAttentionLM() : new HybridNGramEngine());
      eng.fromJSON(snap.json);
      m.engine = eng;
      this.lastTrainingSnapshot = null;
      this.saveModelsToStorage();
      this.updateStatsDisplay();
      this.refreshMemoryInspector();
      this.showToast('Undid the last training run of "' + (snap.name || m.name) + '"; previous weights restored.');
    } catch (err) {
      this.showToast('Could not undo: ' + (err.message || err), 'error');
    }
  }

  exportCorpus() {
    const text = document.getElementById('training-text').value;
    if (!text.trim()) { this.showToast('There is no corpus to export.', 'error'); return; }
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' }), url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = 'training-corpus-0.2.2.txt'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
