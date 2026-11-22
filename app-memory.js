// Memory tab: preferences, vocabulary, knowledge facts with version history, memory inspector.
// Methods here are copied onto AIApp in main.js, so `this` is the app.

export class AppMemory {
  renderPreferences() {
    const p = this.preferences || {};
    const tone = document.getElementById('pref-tone'), length = document.getElementById('pref-length'), instruction = document.getElementById('pref-instruction');
    if (tone) tone.value = p.tone || 'natural';
    if (length) length.value = p.length || 'balanced';
    if (instruction) instruction.value = p.instruction || '';
  }

  savePreferencesFromUI() {
    this.preferences = {
      tone: document.getElementById('pref-tone').value,
      length: document.getElementById('pref-length').value,
      instruction: document.getElementById('pref-instruction').value.trim()
    };
    localStorage.setItem('teachable_ai_preferences_v6', JSON.stringify(this.preferences));
    this.showToast('Response preferences saved on this device.');
  }

  applyPreferences(text) {
    let out = String(text || '');
    const p = this.preferences || {};
    if (p.length === 'concise') out = out.split(/(?<=[.!?])\s+/).slice(0, 2).join(' ');
    if (p.tone === 'formal') out = out.replace(/\b(yeah|yep|nope)\b/gi, 'yes');
    if (p.tone === 'friendly' && out && !/[.!?]$/.test(out)) out += '!';
    return out;
  }

  analyzeKnowledgeCoverage(engine, prompt) {
    const memory = engine.memory, terms = memory?.extractTerms(prompt) || [];
    const known = terms.filter(t => memory?.idf?.has(t));
    const candidates = memory?.queryCandidates(prompt, 5) || [];
    const best = candidates[0] || null;
    const coverage = terms.length ? known.length / terms.length : 0;
    return { terms, known, coverage, best, candidates, unknownTerms: terms.filter(t => !memory?.idf?.has(t)) };
  }

  generatePreferred(engine, prompt, maxTokens, temperature, repetitionPenalty, topK) {
    const p = this.preferences || {}, analysis = this.analyzeKnowledgeCoverage(engine, prompt), exact = engine.memory?.queryBestMatch(prompt);
    if (exact) return this.applyPreferences(exact.text);
    if (!analysis.best || (analysis.best.score < 0.16 && analysis.coverage < 0.55) || (analysis.coverage < 0.22 && analysis.best.score < 0.34)) {
      this.lastUnknownAnalysis = analysis;
      return `I don't know that yet. My learned vocabulary and knowledge do not contain enough relevant information to answer confidently.\n\nTeach me this topic and I'll add it to my knowledge.[[[UNKNOWN_RESPONSE]]]`;
    }
    this.lastUnknownAnalysis = null;
    const conditioned = [prompt, p.instruction ? `Preference: ${p.instruction}` : '', `Use only learned evidence when possible. If evidence is insufficient, say you do not know.`].filter(Boolean).join('\n');
    return this.applyPreferences(engine.generate(conditioned, maxTokens, temperature, repetitionPenalty, topK));
  }

  addVocabularyEntry() {
    const term = document.getElementById('vocab-term').value.trim(), definition = document.getElementById('vocab-definition').value.trim();
    if (!term || !definition) { this.showToast('Enter a term and its definition.', 'error'); return; }
    const facts = this.activeFacts(), key = term.toLowerCase();
    const existing = facts.find(f => f.title.toLowerCase() === ('vocabulary: ' + key));
    const now = new Date().toISOString();
    if (existing) {
      existing.versions = existing.versions || [];
      existing.versions.push({ title: existing.title, body: existing.body, at: now });
      existing.body = `${term}: ${definition}`;
      existing.updatedAt = now;
    } else {
      facts.push({ id: 'vocab-' + Date.now() + '-' + Math.floor(Math.random() * 9999), title: 'Vocabulary: ' + term, body: `${term}: ${definition}`, createdAt: now, updatedAt: now, versions: [] });
    }
    this.syncFactsToCorpus();
    this.saveModelsToStorage();
    this.renderKnowledgeFacts();
    this.refreshMemoryInspector();
    document.getElementById('vocab-term').value = '';
    document.getElementById('vocab-definition').value = '';
    this.showToast(`Added “${term}” to learned knowledge. Retrain to update token weights.`);
  }

  activeFacts() {
    const m = this.getActiveModelObj();
    if (!m) return [];
    if (!Array.isArray(m.knowledgeItems)) m.knowledgeItems = [];
    return m.knowledgeItems;
  }

  resetFactEditor() {
    this.editingFactId = null;
    document.getElementById('fact-title').value = '';
    document.getElementById('fact-body').value = '';
    document.getElementById('save-fact-btn').textContent = 'Add fact';
  }

  saveKnowledgeFact() {
    const title = document.getElementById('fact-title').value.trim(), body = document.getElementById('fact-body').value.trim();
    if (!title || !body) { this.showToast('Enter a title and fact text.', 'error'); return; }
    const facts = this.activeFacts(), now = new Date().toISOString();
    let item = facts.find(x => x.id === this.editingFactId);
    if (item) {
      item.versions = item.versions || [{ title: item.title, body: item.body, at: item.updatedAt || item.createdAt }];
      item.versions.push({ title: item.title, body: item.body, at: now });
      item.title = title;
      item.body = body;
      item.updatedAt = now;
    } else {
      facts.push({ id: 'fact-' + Date.now() + '-' + Math.floor(Math.random() * 9999), title, body, createdAt: now, updatedAt: now, versions: [] });
    }
    this.syncFactsToCorpus();
    this.resetFactEditor();
    this.renderKnowledgeFacts();
    this.saveModelsToStorage();
    this.refreshMemoryInspector();
    this.showToast('Knowledge fact saved and indexed.');
  }

  syncFactsToCorpus() {
    const m = this.getActiveModelObj();
    if (!m) return;
    const facts = this.activeFacts();
    const factText = facts.map(f => `Knowledge: ${f.title}. ${f.body}`).join('\n\n');
    const prior = (m.engine.rawCorpus || '').split(/\n\n/).filter(b => !/^Knowledge: /i.test(b.trim())).join('\n\n');
    m.engine.rawCorpus = [prior, factText].filter(Boolean).join('\n\n');
    if (m.engine.memory) m.engine.memory.indexText(m.engine.rawCorpus);
  }

  renderKnowledgeFacts() {
    const box = document.getElementById('knowledge-facts-list');
    if (!box) return;
    const facts = this.activeFacts();
    box.innerHTML = facts.length ? facts.map(f => `<article class="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2"><div class="flex justify-between gap-2"><div class="text-xs font-semibold text-white">${this.escapeHTML(f.title)}</div><span class="text-[10px] text-slate-500">${(f.versions || []).length + 1} version(s)</span></div><p class="text-[11px] text-slate-400 whitespace-pre-wrap">${this.escapeHTML(f.body)}</p><div class="flex flex-wrap gap-2"><button data-fact-edit="${f.id}" class="text-[10px] px-2 py-1 rounded bg-slate-800 text-cyan-300">Edit</button><button data-fact-versions="${f.id}" class="text-[10px] px-2 py-1 rounded bg-slate-800 text-amber-300">Versions / restore</button><button data-fact-merge="${f.id}" class="text-[10px] px-2 py-1 rounded bg-slate-800 text-violet-300">Merge duplicates</button><button data-fact-delete="${f.id}" class="text-[10px] px-2 py-1 rounded bg-rose-950 text-rose-300">Delete</button></div><div id="fact-history-${f.id}" class="hidden text-[10px] text-slate-400 space-y-1"></div></article>`).join('') : '<p class="text-[11px] text-slate-500">No individual facts yet. Add facts above to manage them separately from raw training examples.</p>';
    box.querySelectorAll('[data-fact-edit]').forEach(b => b.onclick = () => {
      const f = facts.find(x => x.id === b.dataset.factEdit);
      if (!f) return;
      this.editingFactId = f.id;
      document.getElementById('fact-title').value = f.title;
      document.getElementById('fact-body').value = f.body;
      document.getElementById('save-fact-btn').textContent = 'Save revision';
    });
    box.querySelectorAll('[data-fact-delete]').forEach(b => b.onclick = () => {
      if (!confirm('Delete this fact and all its versions?')) return;
      const i = facts.findIndex(x => x.id === b.dataset.factDelete);
      if (i >= 0) facts.splice(i, 1);
      this.syncFactsToCorpus();
      this.renderKnowledgeFacts();
      this.saveModelsToStorage();
      this.refreshMemoryInspector();
    });
    box.querySelectorAll('[data-fact-versions]').forEach(b => b.onclick = () => {
      const f = facts.find(x => x.id === b.dataset.factVersions), h = document.getElementById('fact-history-' + f.id);
      const versions = [...(f.versions || []), { title: f.title, body: f.body, at: f.updatedAt || f.createdAt, current: true }];
      h.classList.toggle('hidden');
      h.innerHTML = versions.map((v, i) => `<div class="flex justify-between gap-2 border-t border-slate-800 pt-1"><span>${v.current ? 'Current' : new Date(v.at).toLocaleString()} · ${this.escapeHTML(v.title)}</span>${v.current ? '' : `<button data-restore-index="${i}" class="text-cyan-300">Restore</button>`}</div>`).join('');
      h.querySelectorAll('[data-restore-index]').forEach(r => r.onclick = () => {
        const v = versions[Number(r.dataset.restoreIndex)];
        f.versions.push({ title: f.title, body: f.body, at: new Date().toISOString() });
        f.title = v.title;
        f.body = v.body;
        f.updatedAt = new Date().toISOString();
        this.syncFactsToCorpus();
        this.renderKnowledgeFacts();
        this.saveModelsToStorage();
        this.refreshMemoryInspector();
      });
    });
    box.querySelectorAll('[data-fact-merge]').forEach(b => b.onclick = () => {
      const f = facts.find(x => x.id === b.dataset.factMerge);
      const dup = facts.filter(x => x.id !== f.id && x.title.toLowerCase() === f.title.toLowerCase());
      if (!dup.length) { this.showToast('No exact-title duplicates to merge.'); return; }
      dup.forEach(d => {
        f.versions.push({ title: d.title, body: d.body, at: d.updatedAt || d.createdAt, merged: true });
        f.body += '\n' + d.body;
      });
      for (let i = facts.length - 1; i >= 0; i--) if (dup.includes(facts[i])) facts.splice(i, 1);
      this.syncFactsToCorpus();
      this.renderKnowledgeFacts();
      this.saveModelsToStorage();
      this.refreshMemoryInspector();
      this.showToast('Merged ' + dup.length + ' duplicate fact(s).');
    });
  }

  refreshMemoryInspector() {
    const active = this.getActiveModelObj();
    if (!active) return;
    const engine = active.engine;
    const memory = engine.memory;
    const qa = memory?.qaPairs || [];
    const chunks = memory?.sentences || [];
    const corpus = engine.rawCorpus || '';
    const context = engine.contextSize || 0;
    const q = (document.getElementById('memory-search')?.value || '').trim().toLowerCase();
    const matches = [...qa.map(x => ({ kind: 'Q&A', title: x.prompt, body: x.response })), ...chunks.map(x => ({ kind: 'Knowledge', title: x.text, body: '' }))]
      .filter(x => !q || (x.title + ' ' + x.body).toLowerCase().includes(q)).slice(0, 100);
    const set = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
    set('memory-qa-count', qa.length.toLocaleString());
    set('memory-chunk-count', chunks.length.toLocaleString());
    set('memory-char-count', corpus.length.toLocaleString());
    set('memory-context-size', context ? context + ' tokens' : 'N/A');
    set('memory-capacity-label', active.name + ' · retrieval index');
    this.renderKnowledgeFacts();
    const box = document.getElementById('memory-results');
    if (box) box.innerHTML = matches.length ? matches.map((x, i) => `<article class="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1"><div class="flex justify-between gap-2"><span class="text-[10px] text-cyan-300 font-mono">${x.kind}</span><span class="text-[10px] text-slate-600">#${i + 1}</span></div><p class="text-xs font-semibold text-slate-200 whitespace-pre-wrap">${this.escapeHTML(x.title)}</p>${x.body ? `<p class="text-xs text-slate-400 whitespace-pre-wrap">${this.escapeHTML(x.body)}</p>` : ''}</article>`).join('') : '<p class="text-xs text-slate-500">No matching memory entries.</p>';
  }

  rebuildActiveMemory() {
    const engine = this.getActiveModelObj()?.engine;
    if (!engine) return;
    if (engine.memory) engine.memory.indexText(engine.rawCorpus || '');
    this.saveModelsToStorage();
    this.refreshMemoryInspector();
    this.showToast('Retrieval index rebuilt from the saved corpus.');
  }

  exportMemoryCorpus() {
    const engine = this.getActiveModelObj()?.engine;
    const corpus = engine?.rawCorpus || '';
    if (!corpus) { this.showToast('No learned corpus to export yet.', 'error'); return; }
    const blob = new Blob([corpus], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'learned-memory-' + Date.now() + '.txt'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
