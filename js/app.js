// Core application class: constructor, UI wiring, tabs and toasts. Other features are mixed in from app-*.js (see main.js).

import { CORPUS_LIBRARY } from './corpus-library.js';

export class AIApp {
  constructor() {
    this.models = [];
    this.activeModelId = null;
    this.lastScreenText = '';
    this.lastTrainingSnapshot = null;
    this.isRestoring = true;
    this.isTraining = false;
    this.stopRequested = false;
    this.trainingWorker = null;
    this.lastEvaluation = [];
    this.lastMemoryQuery = '';
    this.preferences = this.loadPreferences();
    this.editingFactId = null;
    this.previewCache = null;
    this.chatSessions = this.loadChatSessions();
    this.activeChatId = null;
    this.vaultDocs = [];
    this.localReferencePipelines = {};
    this.localReferenceLoading = {};
    this.activeReferenceId = null;
    this.referenceMode = false;
    this.refHistory = [];
    this.studioKnowledge = this.loadStudioKnowledge();

    this.initUI();
    this.refreshModelSelect();
    const earlyModelSelect = document.getElementById('active-model-select');
    if (earlyModelSelect) earlyModelSelect.addEventListener('change', (e) => {
      this.refHistory = [];
      if (e.target.value.startsWith('__ref_')) {
        this.referenceMode = true;
        this.activeReferenceId = e.target.value;
        this.syncChatHeader();
        const ref = this.getLocalReferenceDefinition(this.activeReferenceId);
        this.showToast(ref.name + (this.localReferencePipelines[ref.key] ? ' selected.' : ' selected. Your first message downloads it (cached afterwards).'));
        return;
      }
      this.referenceMode = false;
      this.activeReferenceId = null;
      this.activeModelId = e.target.value;
      const active = this.getActiveModelObj();
      if (!active) return;
      document.getElementById('training-text').value = active.engine.rawCorpus || '';
      this.updateCorpusStats(); this.updateStatsDisplay();
      this.refreshMemoryInspector(); this.renderKnowledgeFacts();
      this.syncChatHeader();
      this.showToast('Switched to: ' + active.name);
    });
    this.renderStudioKnowledgeEditor();
    document.getElementById('save-studio-knowledge')?.addEventListener('click', () => this.saveStudioKnowledgeFromUI());
    document.querySelectorAll('[data-local-model-download]').forEach(btn => btn.addEventListener('click', () => this.prepareLocalReference(btn.dataset.localModelDownload)));
    this.renderCorpusLibrary('all');

    const restoreTask = this.loadModelsFromStorage();
    restoreTask.catch(err => console.warn('Model restore failed:', err)).then(() => {
      this.initializeModels();
      if (this.models.length) {
        if (!this.models.some(m => m.id === this.activeModelId)) this.activeModelId = this.models[0].id;
        this.refreshModelSelect();
        const active = this.getActiveModelObj();
        if (active) {
          document.getElementById('training-text').value = active.engine.rawCorpus || '';
          document.getElementById('chat-model-title').innerText = 'Chatting with: ' + active.name;
          document.getElementById('side-model-name').innerText = active.name;
          this.updateStatsDisplay();
        }
      }
      this.updateSaveIndicator();
    }).catch(err => console.error('Startup failed:', err));
  }

  initUI() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => this.switchTab(btn.getAttribute('tab-target')));
    });

    this.bindSlider('chat-temp', 'temp-val');
    this.bindSlider('chat-topk', 'topk-val');
    this.bindSlider('chat-rep-penalty', 'rep-val');
    this.bindSlider('chat-max-tokens', 'max-tokens-val');
    this.bindSlider('param-epochs', 'val-epochs');
    this.bindSlider('param-lr', 'val-lr');
    this.bindSlider('param-context', 'val-context');
    this.bindSlider('param-hidden', 'val-hidden');

    document.getElementById('param-epochs').addEventListener('input', (e) => {
      document.getElementById('chart-max-epoch').innerText = 'Epoch ' + e.target.value;
    });

    document.getElementById('new-model-modal-btn').addEventListener('click', () => this.openModelModal());
    document.getElementById('hub-create-btn').addEventListener('click', () => this.openModelModal());
    document.getElementById('close-model-modal').addEventListener('click', () => this.closeModelModal());
    document.getElementById('cancel-model-btn').addEventListener('click', () => this.closeModelModal());
    document.getElementById('save-model-btn').addEventListener('click', () => this.handleSaveNewModel());

    document.getElementById('send-msg-btn').addEventListener('click', () => this.handleSendMessage());
    document.getElementById('chat-input').addEventListener('input', () => { this.previewCache = null; });
    document.getElementById('chat-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage();
      }
    });

    document.querySelectorAll('.quick-prompt-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.getElementById('chat-input').value = chip.innerText;
        this.handleSendMessage();
      });
    });

    document.getElementById('clear-chat-btn').addEventListener('click', () => {
      document.getElementById('chat-messages').innerHTML = '';
      this.refHistory = [];
      const session = this.getActiveChatSession();
      if (session) { session.messages = []; this.persistChatSessions(); }
      this.showToast("Chat cleared.");
    });

    document.getElementById('export-chat-btn').addEventListener('click', () => this.exportChatTranscript());
    document.getElementById('new-chat-btn').addEventListener('click', () => this.newChatSession());
    document.getElementById('chat-session-select').addEventListener('change', e => this.switchChatSession(e.target.value));
    document.getElementById('corpus-vault-upload').addEventListener('change', e => this.importCorpusFiles(e));
    document.getElementById('refresh-vault-btn').addEventListener('click', () => this.refreshCorpusVault());
    document.getElementById('load-vault-corpus-btn').addEventListener('click', () => this.appendSelectedVaultDocs());

    document.getElementById('open-library-shortcut-btn').addEventListener('click', () => this.switchTab('library-tab'));
    document.getElementById('library-category-filter').addEventListener('change', (e) => this.renderCorpusLibrary(e.target.value));
    document.getElementById('auto-format-btn').addEventListener('click', () => this.autoFormatCorpus());
    document.getElementById('clear-corpus-btn').addEventListener('click', () => {
      document.getElementById('training-text').value = '';
      this.updateCorpusStats();
    });

    document.getElementById('start-train-btn').addEventListener('click', () => this.startTraining().catch(err => {
      console.error('Training error:', err);
      this.isTraining = false;
      document.getElementById('start-train-btn').disabled = false;
      document.getElementById('stop-train-btn').disabled = true;
      document.getElementById('train-status-dot').className = 'w-2 h-2 rounded-full bg-rose-500';
      document.getElementById('train-status-text').innerText = 'Training failed';
      this.showToast('Training failed: ' + (err.message || 'unexpected error'), 'error');
    }));
    document.getElementById('stop-train-btn').addEventListener('click', () => this.requestStopTraining());
    document.getElementById('training-profile').addEventListener('change', () => this.updateTrainingProfileNote());
    this.updateTrainingProfileNote();
    document.getElementById('file-upload').addEventListener('change', (e) => this.handleFileUpload(e));

    document.getElementById('export-active-btn').addEventListener('click', () => this.exportActiveModelJSON());
    document.getElementById('export-all-btn').addEventListener('click', () => this.exportAllModelsBundle());
    document.getElementById('import-brain-file').addEventListener('change', (e) => this.importJSONFile(e));

    document.getElementById('close-teach-modal').addEventListener('click', () => this.closeTeachModal());
    document.getElementById('cancel-teach-btn').addEventListener('click', () => this.closeTeachModal());
    document.getElementById('submit-teach-btn').addEventListener('click', () => this.submitCorrection());

    document.getElementById('training-text').addEventListener('input', () => this.updateCorpusStats());
    document.getElementById('clean-duplicates-btn').addEventListener('click', () => this.cleanCorpusDuplicates());
    document.getElementById('preflight-btn')?.addEventListener('click', () => this.switchTab('train-tab') || this.runCorpusPreflight());
    document.getElementById('auto-fix-btn')?.addEventListener('click', () => this.switchTab('train-tab') || this.autoFixCorpus());
    document.getElementById('export-corpus-csv-btn')?.addEventListener('click', () => this.exportCorpusCSV());
    document.getElementById('undo-train-btn')?.addEventListener('click', () => this.undoLastTraining());
    document.getElementById('run-eval-btn').addEventListener('click', () => this.runEvaluation());
    document.getElementById('export-eval-btn').addEventListener('click', () => this.exportEvaluation());
    document.getElementById('compare-eval-btn').addEventListener('click', () => this.compareEvaluation());
    document.getElementById('compare-reference-btn')?.addEventListener('click', () => this.compareAgainstReference());
    document.getElementById('export-corpus-btn').addEventListener('click', () => this.exportCorpus());
    document.getElementById('refresh-memory-btn').addEventListener('click', () => this.refreshMemoryInspector());
    document.getElementById('memory-search').addEventListener('input', () => this.refreshMemoryInspector());
    document.getElementById('reindex-memory-btn').addEventListener('click', () => this.rebuildActiveMemory());
    document.getElementById('export-memory-btn').addEventListener('click', () => this.exportMemoryCorpus());
    document.getElementById('calculate-prompt-btn').addEventListener('click', () => this.calculatePrompt());
    document.getElementById('save-preferences-btn').addEventListener('click', () => this.savePreferencesFromUI());
    document.getElementById('save-fact-btn').addEventListener('click', () => this.saveKnowledgeFact());
    document.getElementById('add-vocab-btn').addEventListener('click', () => this.addVocabularyEntry());
    document.getElementById('cancel-fact-edit-btn').addEventListener('click', () => this.resetFactEditor());
    this.renderPreferences();
    this.renderKnowledgeFacts();
    this.initBackupUI();
    this.initAssistantPanel();
    this.initQAGenerator();
    this.initBenchmark();
  }

  initializeModels() {
    if (this.modelsInitialized) return;
    this.modelsInitialized = true;
    this.isRestoring = false;
    if (this.models.length === 0) {
      const starter = this.createNewModel('Studio Assistant', 'neural');
      const defaultCorpus = this.mergeCorpus(CORPUS_LIBRARY[0].text, CORPUS_LIBRARY[1].text);
      document.getElementById('training-text').value = defaultCorpus;
      this.updateCorpusStats();
      this.quickTrainModel(starter, defaultCorpus, 12).catch(err => {
        console.error('Starter training failed:', err);
        this.showToast('Starter model training failed. You can retrain it in Train Studio.', 'error');
      });
    } else {
      this.activeModelId = this.models[0].id;
      const active = this.getActiveModelObj();
      document.getElementById('training-text').value = active.engine.rawCorpus || '';
      this.updateCorpusStats();
    }
    this.refreshModelSelect();
    this.updateStatsDisplay();
    this.refreshMemoryInspector();
    this.initChatSessions();
    this.refreshCorpusVault().catch(err => console.warn('Corpus Vault unavailable', err));
  }

  switchTab(targetId) {
    this.currentTabId = targetId;
    document.querySelectorAll('.tab-btn').forEach(btn => {
      if (btn.getAttribute('tab-target') === targetId) {
        btn.classList.add('active', 'bg-brand-600/30', 'border-brand-500/40', 'text-white');
        btn.classList.remove('text-slate-400', 'border-transparent');
      } else {
        btn.classList.remove('active', 'bg-brand-600/30', 'border-brand-500/40', 'text-white');
        btn.classList.add('text-slate-400', 'border-transparent');
      }
    });

    document.querySelectorAll('.tab-content').forEach(sec => {
      sec.classList.add('hidden');
      sec.classList.remove('flex');
    });

    const activeSec = document.getElementById(targetId);
    if (!activeSec) return;
    activeSec.classList.remove('hidden');
    activeSec.classList.add('flex');
    if (targetId === 'eval-tab') this.renderBenchTargets();
    if (this.panel && !this.panel.messages.length) this.panelRender();
  }

  bindSlider(sliderId, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    slider.addEventListener('input', (e) => { display.innerText = e.target.value; });
  }

  showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'p-3 rounded-xl shadow-xl text-xs font-medium text-white transition-all transform duration-300 pointer-events-auto flex items-center justify-between border ' + (
      type === 'error' ? 'bg-rose-900 border-rose-700' : 'bg-slate-900 border-slate-700'
    );
    toast.innerHTML = '<span>' + this.escapeHTML(message) + '</span><button class="ml-2 text-slate-400 hover:text-white"><i class="fa-solid fa-xmark"></i></button>';
    toast.querySelector('button').addEventListener('click', () => toast.remove());
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  escapeHTML(str) {
    if (str === null || str === undefined || str === '') return '';
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' };
    return String(str).replace(/[&<>'"]/g, ch => map[ch] || ch);
  }

  syncChatHeader() {
    const title = document.getElementById('chat-model-title'), side = document.getElementById('side-model-name'), badge = document.getElementById('side-engine-badge');
    if (this.referenceMode && this.activeReferenceId) {
      const ref = this.getLocalReferenceDefinition(this.activeReferenceId);
      if (title) title.innerText = 'Chatting with: ' + ref.name;
      if (side) side.innerText = ref.name;
      if (badge) badge.innerText = 'Assistant · Local';
    } else {
      const a = this.getActiveModelObj();
      if (a) {
        if (title) title.innerText = 'Chatting with: ' + a.name;
        if (side) side.innerText = a.name;
        if (badge) badge.innerText = a.type === 'neural' ? 'Neural MLP' : (a.type === 'attention' ? 'Tiny Attention LM' : 'TF-IDF');
      }
    }
    const labels = this.referenceMode ? ['how do I train a model', 'what is Live Memory', 'how do I import data'] : ['hi', 'who are you', 'tell me a joke'];
    document.querySelectorAll('.quick-prompt-chip').forEach((chip, i) => { if (labels[i]) chip.innerText = labels[i]; });
  }

  /* ============ Assistant Actions: the app does things on request ============ */
  static ENGINE_LABELS = { neural: 'Turn-Anchored Neural MLP', ngram: 'Semantic TF-IDF + N-Gram', attention: 'Tiny Attention LM' };
}
