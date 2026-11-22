// Model management: create, select, duplicate, delete, import/export, progression and stats.
// Methods here are copied onto AIApp in main.js, so `this` is the app.
import { HybridNGramEngine, TinyAttentionLM, TinyNeuralLM } from './engines.js';

export class AppModels {
  mergeCorpus(existing, incoming, replaceMatchingQA = true) {
    const merged = String(existing || '').trim().split(/\n\s*\n/).map(x => x.trim()).filter(Boolean);
    const keyOf = block => {
      const match = block.match(/^user\s*:\s*(.+?)(?:\r?\n|$)/im);
      return match ? match[1].trim().toLowerCase().replace(/\s+/g, ' ') : null;
    };
    const seen = new Set(merged.map(x => x.toLowerCase().replace(/\s+/g, ' ')));
    for (const raw of String(incoming || '').trim().split(/\n\s*\n/)) {
      const clean = raw.trim();
      if (!clean) continue;
      const exact = clean.toLowerCase().replace(/\s+/g, ' ');
      if (seen.has(exact)) continue;
      const qKey = keyOf(clean);
      let replaced = false;
      if (qKey) {
        const idx = merged.findIndex(old => keyOf(old) === qKey);
        if (idx >= 0) {
          if (replaceMatchingQA) {
            seen.delete(merged[idx].toLowerCase().replace(/\s+/g, ' '));
            merged[idx] = clean;
            seen.add(exact);
          }
          replaced = true;
        }
      }
      if (!replaced) {
        merged.push(clean);
        seen.add(exact);
      }
    }
    return merged.join('\n\n');
  }

  async quickTrainModel(modelObj, text, epochs = 60, fromScratch = false) {
    const cumulativeText = fromScratch ? String(text || '').trim() : this.mergeCorpus(modelObj.engine.rawCorpus, text);
    if (modelObj.type === 'neural' || modelObj.type === 'attention') {
      if (fromScratch) {
        modelObj.engine.lossHistory = [];
        modelObj.engine.epochsTrained = 0;
      }
      const dataset = modelObj.engine.prepareDataset(cumulativeText, true);
      for (let ep = 1; ep <= epochs; ep++) {
        const lr = 0.09 * (0.25 + 0.75 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs)));
        let epochLoss = 0;
        for (let i = 0; i < dataset.X.length; i++) {
          const idx = (i * 7919 + ep * 104729) % Math.max(1, dataset.X.length);
          epochLoss += modelObj.engine.trainStep(dataset.X[idx], dataset.Y[idx], lr);
          if (i > 0 && i % 32 === 0) await new Promise(resolve => setTimeout(resolve, 0));
        }
        modelObj.engine.lossHistory.push(epochLoss / Math.max(1, dataset.X.length));
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      modelObj.engine.epochsTrained += epochs;
      modelObj.engine.isTrained = true;
      modelObj.engine.rawCorpus = cumulativeText;
      modelObj.engine.memory.indexText(cumulativeText);
    } else {
      await modelObj.engine.trainAsync(cumulativeText.slice(0, 500000), 500000);
    }
    this.saveModelsToStorage();
    this.updateStatsDisplay();
  }

  getActiveModelObj() {
    return this.models.find(m => m.id === this.activeModelId) || this.models[0];
  }

  createNewModel(name, type) {
    this.referenceMode = false;
    this.activeReferenceId = null;
    const newModel = {
      id: 'model-' + Date.now() + '-' + Math.floor(Math.random() * 1000),
      name: name || 'Custom AI',
      type: type || 'neural',
      engine: type === 'neural' ? new TinyNeuralLM() : (type === 'attention' ? new TinyAttentionLM() : new HybridNGramEngine()),
      createdAt: new Date().toLocaleDateString()
    };
    this.models.push(newModel);
    this.activeModelId = newModel.id;
    this.saveModelsToStorage();
    this.refreshModelSelect();
    this.updateStatsDisplay();
    return newModel;
  }

  deleteModel(id) {
    if (this.models.length <= 1) {
      this.showToast("Cannot delete the only remaining model!", "error");
      return;
    }
    this.models = this.models.filter(m => m.id !== id);
    if (this.activeModelId === id) {
      this.activeModelId = this.models[0].id;
    }
    this.saveModelsToStorage();
    this.refreshModelSelect();
    this.updateStatsDisplay();
    this.showToast("Model deleted.");
  }

  duplicateModel(id) {
    const src = this.models.find(m => m.id === id);
    if (!src) return;
    const copy = this.createNewModel(src.name + ' (Copy)', src.type);
    copy.engine.fromJSON(src.engine.toJSON());
    this.saveModelsToStorage();
    this.updateStatsDisplay();
    this.showToast('Duplicated ' + src.name + '!');
  }

  selectModel(id) {
    this.referenceMode = false;
    this.activeReferenceId = null;
    this.refHistory = [];
    this.activeModelId = id;
    const active = this.getActiveModelObj();
    document.getElementById('training-text').value = active.engine.rawCorpus || '';
    this.updateCorpusStats();
    this.refreshModelSelect();
    this.updateStatsDisplay();
    this.refreshMemoryInspector();
    this.showToast('Active model: ' + active.name);
  }

  refreshModelSelect() {
    const select = document.getElementById('active-model-select');
    const currentValue = select.value;
    const refs = [
      ['__ref_smollm2_360__', 'SmolLM2 360M Instruct · assistant'],
      ['__ref_qwen05__', 'Qwen2.5 0.5B Instruct · assistant'],
      ['__ref_smollm2_135__', 'SmolLM2 135M Instruct · assistant']
    ];
    select.innerHTML = '<option value="" disabled>Choose model…</option>' + this.models.map(m => `
      <option style="background:#0f172a;color:#f8fafc" value="${m.id}" ${m.id === this.activeModelId ? 'selected' : ''}>${this.escapeHTML(m.name)} (${m.type === 'neural' ? 'MLP' : (m.type === 'attention' ? 'Attention' : 'TF-IDF')})</option>
    `)  .join('');
    if (this.referenceMode && this.activeReferenceId) select.value = this.activeReferenceId;
    else if (this.activeModelId) select.value = this.activeModelId;
    else if (!this.models.length) select.value = '__ref_smollm2_360__';
    this.renderModelsHubCards();
  }

  renderModelsHubCards() {
    const container = document.getElementById('models-card-list');
    container.innerHTML = this.models.map(m => {
      const isActive = m.id === this.activeModelId;
      const vocab = m.engine.tokenizer ? m.engine.tokenizer.size() : 0;
      const qaCount = m.engine.memory ? m.engine.memory.qaPairs.length : 0;
      return `
        <div class="p-4 rounded-xl glass-panel border ${isActive ? 'border-brand-500 bg-brand-950/20' : 'border-slate-800'} space-y-3">
          <div class="flex items-center justify-between">
            <div>
              <span class="font-bold text-sm text-white block">${this.escapeHTML(m.name)}</span>
              <span class="text-[10px] text-slate-400 font-mono">${m.type === 'neural' ? 'Turn-Anchored Neural MLP' : (m.type === 'attention' ? 'Tiny Causal Attention LM' : 'Semantic TF-IDF + N-Gram')}</span>
            </div>
            ${isActive ? '<span class="px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 text-[10px] font-semibold border border-brand-500/40">Active</span>' : ''}
          </div>

          <div class="grid grid-cols-3 gap-2 text-[11px] font-mono text-slate-400 border-t border-b border-slate-800/60 py-2">
            <div>Vocab: <span class="text-slate-200">${vocab}</span></div>
            <div>Q&As: <span class="text-emerald-400">${qaCount}</span></div>
            <div>Epochs: <span class="text-slate-200">${m.engine.epochsTrained || (m.engine.isTrained ? 1 : 0)}</span></div>
          </div>

          <div class="flex items-center gap-2 pt-1">
            ${!isActive ? `<button onclick="app.selectModel('${m.id}')" class="px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-medium transition">Select</button>` : ''}
            <button onclick="app.duplicateModel('${m.id}')" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition" title="Duplicate"><i class="fa-solid fa-copy"></i></button>
            <button onclick="app.deleteModel('${m.id}')" class="px-2.5 py-1.5 rounded-lg bg-rose-950/40 hover:bg-rose-900 text-rose-300 text-xs transition ml-auto" title="Delete"><i class="fa-solid fa-trash-can"></i></button>
          </div>
        </div>
      `;
    }).join('');
  }

  getResearchPoints() {
    const m = this.getActiveModelObj();
    if (!m) return 0;
    const e = m.engine;
    const vocab = e.tokenizer?.size() || 0, qa = e.memory?.qaPairs?.length || 0, chars = (e.rawCorpus || '').length, epochs = e.epochsTrained || 0;
    return 100 + (e.isTrained ? 100 : 0) + (vocab >= 100 ? 100 : 0) + (qa >= 10 ? 100 : 0) + (chars >= 10000 ? 100 : 0) + (epochs >= 10 ? 100 : 0);
  }

  renderProgression() {
    const m = this.getActiveModelObj();
    if (!m) return;
    const e = m.engine;
    const vocab = e.tokenizer?.size() || 0, qa = e.memory?.qaPairs?.length || 0, chars = (e.rawCorpus || '').length, epochs = e.epochsTrained || 0;
    const tasks = [
      { name: 'Model founder · +100 RP', done: true, detail: 'Create your first model.' },
      { name: 'First training · +100 RP', done: !!e.isTrained, detail: 'Complete a training run.' },
      { name: 'Vocabulary builder · +100 RP', done: vocab >= 100, detail: `${vocab}/100 tokens` },
      { name: 'Knowledge collector · +100 RP', done: qa >= 10, detail: `${qa}/10 Q&A pairs` },
      { name: 'Corpus explorer · +100 RP', done: chars >= 10000, detail: `${chars.toLocaleString()}/10,000 characters` },
      { name: 'Training veteran · +100 RP', done: epochs >= 10, detail: `${epochs}/10 epochs` }
    ];
    const xp = this.getResearchPoints(), level = Math.floor(xp / 300) + 1, into = xp % 300, pct = into / 3;
    const names = ['Apprentice', 'Tinkerer', 'Trainer', 'Researcher', 'Architect', 'AI Engineer', 'Model Scientist'];
    const lvlName = names[Math.min(names.length - 1, level - 1)];
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('progress-level', `Level ${level} · ${lvlName}`);
    set('progress-xp', `${xp} research points · ${Math.max(0, 300 - into)} RP to next level`);
    const bar = document.getElementById('progress-bar');
    if (bar) bar.style.width = `${pct}%`;
    const box = document.getElementById('progress-achievements');
    if (box) box.innerHTML = tasks.map(t => `<div class="flex items-center gap-2 p-2 rounded-lg ${t.done ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-slate-950 border border-slate-800'}"><i class="fa-solid ${t.done ? 'fa-circle-check text-emerald-400' : 'fa-lock text-slate-600'}"></i><span class="min-w-0"><b class="block text-[10px] ${t.done ? 'text-emerald-200' : 'text-slate-300'}">${t.name}</b><small class="text-[9px] text-slate-500">${t.detail}</small></span></div>`).join('') +
      `<div class="sm:col-span-2 p-3 rounded-lg border ${xp >= 300 ? 'border-violet-500/30 bg-violet-500/10' : 'border-slate-800 bg-slate-950'}"><b class="text-[11px] text-violet-200">${xp >= 300 ? 'Unlocked: Tiny Attention LM' : 'Next unlock: Tiny Attention LM'}</b><p class="text-[10px] text-slate-400">${xp >= 300 ? 'You can now create an experimental attention-based model.' : 'Earn 300 research points through training, vocabulary growth, and knowledge collection to unlock the experimental attention architecture.'}</p></div>`;
  }

  updateStatsDisplay() {
    const activeObj = this.getActiveModelObj();
    const model = activeObj.engine;

    this.syncChatHeader();

    const vocabSize = model.tokenizer ? model.tokenizer.size() : 0;
    const paramsCount = model.getNumParameters ? model.getNumParameters() : 0;
    const qaCount = model.memory ? model.memory.qaPairs.length : 0;

    document.getElementById('side-vocab').innerText = vocabSize + ' tokens';
    document.getElementById('side-qa-count').innerText = qaCount + ' pairs';
    document.getElementById('side-params').innerText = paramsCount.toLocaleString();
    document.getElementById('side-epochs').innerText = model.epochsTrained || (model.isTrained ? 1 : 0);
    const contextSlider = document.getElementById('param-context');
    const contextLabel = document.getElementById('val-context');
    if (contextSlider && model.contextSize) { contextSlider.value = String(model.contextSize); if (contextLabel) contextLabel.textContent = String(model.contextSize); }
    const sideContext = document.getElementById('side-context');
    if (sideContext) sideContext.innerText = (model.contextSize || 'N/A') + (model.contextSize ? ' tokens' : '');
    this.renderModelsHubCards();
    this.renderProgression();
    this.refreshMemoryInspector();
  }

  openModelModal() {
    if (this.isRestoring) this.isRestoring = false;
    document.getElementById('modal-model-name').value = '';
    const att = document.querySelector('#modal-model-type option[value="attention"]');
    const unlocked = this.getResearchPoints() >= 300;
    if (att) {
      att.disabled = !unlocked;
      att.textContent = unlocked ? 'Tiny Attention LM · unlocked at Researcher' : 'Tiny Attention LM · locked (earn 300 research points)';
    }
    if (!unlocked) document.getElementById('modal-model-type').value = 'neural';
    document.getElementById('model-modal').classList.remove('hidden');
  }

  closeModelModal() {
    document.getElementById('model-modal').classList.add('hidden');
  }

  handleSaveNewModel() {
    const name = document.getElementById('modal-model-name').value.trim();
    const type = document.getElementById('modal-model-type').value;
    if (type === 'attention' && this.getResearchPoints() < 300) {
      this.showToast('Unlock Tiny Attention LM by earning 300 research points.', 'error');
      return;
    }
    if (!name) {
      this.showToast("Please enter a model name!", "error");
      return;
    }
    this.createNewModel(name, type);
    this.closeModelModal();
    this.showToast('Created new model: ' + name);
  }

  exportActiveModelJSON() {
    const activeObj = this.getActiveModelObj();
    const payload = {
      formatVersion: 2,
      appVersion: '0.2.0-beta',
      architecture: activeObj.type === 'neural' ? 'tiny-mlp-next-token' : (activeObj.type === 'attention' ? 'tiny-causal-self-attention-plus-ffn' : 'tfidf-ngram-retrieval'),
      exportedAt: new Date().toISOString(),
      trainedOnDevice: true,
      name: activeObj.name,
      type: activeObj.type,
      data: activeObj.engine.toJSON()
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model-' + activeObj.name.toLowerCase().replace(/[^a-z0-9_-]+/g, '-') + '.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  exportAllModelsBundle() {
    const bundle = this.models.map(m => ({
      name: m.name,
      type: m.type,
      data: m.engine.toJSON()
    }));
    const blob = new Blob([JSON.stringify({ isBundle: true, formatVersion: 2, appVersion: '0.2.0-beta', exportedAt: new Date().toISOString(), trainedOnDevice: true, models: bundle }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'teachable-ai-bundle-0.2.0-beta-' + Date.now() + '.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  importJSONFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const parsed = JSON.parse(evt.target.result);
        if (parsed.isBundle) {
          if (!Array.isArray(parsed.models)) throw new Error('Invalid model bundle');
          parsed.models.forEach(item => {
            const newM = this.createNewModel(item.name, item.type);
            newM.engine.fromJSON(item.data);
          });
          this.showToast("Imported model bundle!");
        } else if (parsed.type) {
          const newM = this.createNewModel(parsed.name || 'Imported Model', parsed.type);
          newM.engine.fromJSON(parsed.data || parsed);
          this.showToast("Imported model!");
        }
        this.saveModelsToStorage();
        this.updateStatsDisplay();
      } catch (err) {
        this.showToast("Invalid JSON file", "error");
      }
    };
    reader.readAsText(file);
  }
}
