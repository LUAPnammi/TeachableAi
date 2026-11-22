// Browser storage: IndexedDB, localStorage, Corpus Vault, saved chats, preferences and studio notes.
// Methods here are copied onto AIApp in main.js, so `this` is the app.
import { HybridNGramEngine, TinyAttentionLM, TinyNeuralLM } from './engines.js';

export class AppStorage {
  openStorageDB() {
    return new Promise((resolve, reject) => {
      if (!('indexedDB' in window)) { reject(new Error('IndexedDB unavailable')); return; }
      const request = indexedDB.open('teachable-ai-studio-v6', 3);
      let settled = false;
      const fail = (err) => {
        if (settled) return;
        settled = true;
        reject(err instanceof Error ? err : new Error(String(err || 'Could not open model database')));
      };
      const watchdog = setTimeout(() => fail(new Error('Model database open timed out. Close other Teachable AI tabs and reload.')), 4500);
      request.onupgradeneeded = () => {
        try {
          if (!request.result.objectStoreNames.contains('state')) request.result.createObjectStore('state');
          if (!request.result.objectStoreNames.contains('corpusDocs')) request.result.createObjectStore('corpusDocs', { keyPath: 'id' });
          if (!request.result.objectStoreNames.contains('taraAssets')) request.result.createObjectStore('taraAssets');
        } catch (err) { fail(err); }
      };
      request.onblocked = () => fail(new Error('Model database upgrade is blocked by another open tab. Close other Teachable AI tabs and reload.'));
      request.onsuccess = () => {
        const db = request.result;
        if (settled) { db.close(); return; }
        settled = true;
        clearTimeout(watchdog);
        db.onversionchange = () => db.close();
        resolve(db);
      };
      request.onerror = () => { clearTimeout(watchdog); fail(request.error || new Error('Could not open model database')); };
    });
  }

  saveModelsToStorage() {
    const serializable = this.models.map(m => ({ id: m.id, name: m.name, type: m.type, createdAt: m.createdAt, knowledgeItems: m.knowledgeItems || [], engineData: m.engine.toJSON() }));
    const payload = JSON.stringify(serializable);
    let localSaved = false;
    try { localStorage.setItem('teachable_ai_models_v6', payload); localSaved = true; }
    catch (e) { console.warn('localStorage quota reached; saving to IndexedDB instead.', e); }
    this.openStorageDB().then(db => new Promise((resolve, reject) => {
      const tx = db.transaction('state', 'readwrite');
      tx.objectStore('state').put({ savedAt: Date.now(), models: serializable }, 'models');
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error || new Error('IndexedDB write failed')); };
    })).catch(err => {
      console.warn('IndexedDB save failed:', err);
      if (!localSaved) this.showToast('Browser storage is full or unavailable. Export a JSON backup to keep this model.', 'error');
    });
    if (!localSaved) this.showToast('Saved using IndexedDB when available. Export a backup for safety.', 'warning');
  }

  async loadModelsFromStorage() {
    const restore = data => {
      if (!Array.isArray(data)) return false;
      this.models = data.map(m => {
        const engine = m.type === 'neural' ? new TinyNeuralLM() : (m.type === 'attention' ? new TinyAttentionLM() : new HybridNGramEngine());
        if (m.engineData) engine.fromJSON(m.engineData);
        return { id: m.id || ('model-' + Date.now()), name: m.name || 'Imported Model', type: m.type || 'neural', createdAt: m.createdAt || new Date().toLocaleDateString(), knowledgeItems: m.knowledgeItems || [], engine };
      });
      return true;
    };
    try {
      const raw = (localStorage.getItem('teachable_ai_models_v6') || localStorage.getItem('teachable_ai_models_v5'));
      if (raw) restore(JSON.parse(raw));
    } catch (e) { console.warn('Could not parse localStorage models:', e); }
    try {
      const db = await this.openStorageDB();
      const record = await new Promise((resolve, reject) => {
        const req = db.transaction('state', 'readonly').objectStore('state').get('models');
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      db.close();
      if (record && Array.isArray(record.models)) restore(record.models);
    } catch (e) {}
  }

  async putCorpusDoc(doc) {
    const db = await this.openStorageDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('corpusDocs', 'readwrite');
      tx.objectStore('corpusDocs').put(doc);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  async getCorpusDocs() {
    const db = await this.openStorageDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('corpusDocs', 'readonly'), req = tx.objectStore('corpusDocs').getAll();
      req.onsuccess = () => { db.close(); resolve(req.result || []); };
      req.onerror = () => { db.close(); reject(req.error); };
    });
  }

  async deleteCorpusDoc(id) {
    const db = await this.openStorageDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('corpusDocs', 'readwrite');
      tx.objectStore('corpusDocs').delete(id);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  async refreshCorpusVault() {
    const docs = await this.getCorpusDocs();
    this.vaultDocs = docs;
    this.ragIndex = null;
    if (this.updateRagInfo) this.updateRagInfo();
    const list = document.getElementById('corpus-vault-list');
    if (!list) return;
    document.getElementById('vault-summary').textContent = `${docs.length} files · ${docs.reduce((n, d) => n + (d.size || 0), 0).toLocaleString()} bytes stored in IndexedDB`;
    list.innerHTML = docs.length ? docs.map(d => `<label class="flex items-start gap-2 p-2 rounded-lg bg-slate-950 border border-slate-800"><input type="checkbox" class="vault-doc-check mt-1" value="${this.escapeHTML(d.id)}" checked><span class="min-w-0 flex-1"><span class="block text-xs text-slate-200 truncate">${this.escapeHTML(d.name)}</span><span class="block text-[10px] text-slate-500">${(d.size || 0).toLocaleString()} bytes · ${new Date(d.updatedAt).toLocaleDateString()}</span></span><button type="button" data-vault-delete="${this.escapeHTML(d.id)}" class="text-rose-400 text-xs px-1" title="Delete dataset"><i class="fa-solid fa-trash"></i></button></label>`).join('') : '<p class="text-xs text-slate-500">No external datasets yet. Import .txt, .md, .csv, .json, .jsonl, or .tsv files.</p>';
    list.querySelectorAll('[data-vault-delete]').forEach(b => b.addEventListener('click', async e => {
      e.preventDefault();
      await this.deleteCorpusDoc(b.dataset.vaultDelete);
      await this.refreshCorpusVault();
    }));
  }

  appendSelectedVaultDocs() {
    const ids = Array.from(document.querySelectorAll('.vault-doc-check:checked')).map(x => x.value);
    const docs = this.vaultDocs.filter(d => ids.includes(d.id));
    if (!docs.length) { this.showToast('Select at least one dataset.', 'error'); return; }
    const text = docs.map(d => `\n\n# Imported dataset: ${d.name}\n${d.text}`).join('');
    const area = document.getElementById('training-text');
    area.value = this.mergeCorpus(area.value, text, false);
    this.updateCorpusStats();
    this.switchTab('train-tab');
    this.showToast(`${docs.length} dataset(s) appended. Train to index and learn from them.`);
  }

  async importCorpusFiles(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    for (const file of files) {
      try {
        const text = await file.text();
        if (!text.trim()) continue;
        await this.putCorpusDoc({ id: 'doc-' + Date.now() + '-' + Math.floor(Math.random() * 99999), name: file.name, type: file.type || 'text/plain', size: file.size, updatedAt: Date.now(), text });
      } catch (err) {
        this.showToast('Could not import ' + file.name + ': ' + err.message, 'error');
      }
    }
    event.target.value = '';
    await this.refreshCorpusVault();
    this.showToast(`${files.length} dataset file(s) stored in Corpus Vault.`);
  }

  loadPreferences() { try { return JSON.parse(localStorage.getItem('teachable_ai_preferences_v6') || '{}'); } catch (e) { return {}; } }

  /* ============ Assistant side panel ============ */
  lsGet(k, d) { try { const v = localStorage.getItem('teachable_ai_panel_' + k); return v === null ? d : v; } catch (e) { return d; } }

  lsSet(k, v) { try { localStorage.setItem('teachable_ai_panel_' + k, v); } catch (e) {} }

  loadChatSessions() { try { return JSON.parse(localStorage.getItem('teachable_ai_chat_sessions_beta') || '[]'); } catch (e) { return []; } }

  persistChatSessions() { try { localStorage.setItem('teachable_ai_chat_sessions_beta', JSON.stringify(this.chatSessions.slice(0, 40))); } catch (e) { this.showToast('Chat history storage is full.', 'error'); } }

  loadStudioKnowledge() {
    const starter = [
      'MY_PROJECT: Add your own goals, roadmap and custom notes here. The built-in studio guide is always included automatically, so you only need to write what is unique to your project.'
    ].join('\n');
    try {
      const saved = localStorage.getItem('teachable_ai_studio_knowledge_v016') || localStorage.getItem('teachable_ai_studio_knowledge_v015') || localStorage.getItem('teachable_ai_studio_knowledge_v013');
      return saved || starter;
    } catch (e) {
      return starter;
    }
  }

  saveStudioKnowledgeFromUI() {
    const el = document.getElementById('studio-knowledge-input');
    if (!el) return;
    this.studioKnowledge = el.value.trim();
    try {
      localStorage.setItem('teachable_ai_studio_knowledge_v016', this.studioKnowledge);
    } catch (e) {
      this.showToast('Could not save studio notes in browser storage.', 'error');
      return;
    }
    const status = document.getElementById('studio-knowledge-status');
    if (status) status.textContent = 'Saved · shared with all 3 local assistant models.';
    this.showToast('Studio knowledge saved for the 3 local assistant models.');
  }

  renderStudioKnowledgeEditor() {
    const el = document.getElementById('studio-knowledge-input');
    if (el) el.value = this.studioKnowledge || '';
  }
}
