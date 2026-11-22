// Improve Lab: held-out test scoring, review queue for assistant-generated pairs,
// and progressive training rounds with automatic keep-or-revert.
// Methods here are copied onto AIApp in main.js, so `this` is the app.

const TEST_SET_LS = 'teachable_ai_testsets_v1';
const REVIEW_QUEUE_LS = 'teachable_ai_review_queue_v1';

// A "correct" answer needs this much word overlap (F1) with the held-out answer.
const PASS_F1 = 0.5;
const MAX_TEST_PAIRS = 200;
const REVIEW_CAP = 30; // hard safety cap on how many pairs one round can add

// Stable string hash, so the held-out split is the same every time it is rebuilt.
function stableHash(s) { let h = 0; for (let i = 0; i < s.length; i++) { h = (h * 31 + s.charCodeAt(i)) | 0; } return Math.abs(h); }
function normText(s) { return String(s || '').toLowerCase().replace(/[^\w' ]+/g, ' ').replace(/\s+/g, ' ').trim(); }
function wordList(s) { return normText(s).split(' ').filter(Boolean); }
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

export class AppImprove {

  /* ============ setup ============ */

  initImproveLab() {
    const $ = id => document.getElementById(id);
    if (!$('test-build-btn')) return;
    $('test-build-btn').addEventListener('click', () => { this.buildTestSetFromCorpus(); });
    $('test-score-btn').addEventListener('click', () => {
      this.scoreActiveModel().then(r => { if (r) this.showToast(`Test score: ${r.correct}/${r.total} (${r.percent}%) answered correctly.`); });
    });
    $('test-clear-btn').addEventListener('click', () => this.clearTestSet());
    $('review-generate-btn').addEventListener('click', () => this.generateReviewPairs());
    $('review-approve-all').addEventListener('click', () => this.approveAllClean());
    $('review-reject-all').addEventListener('click', () => this.rejectAllStaged());
    $('review-commit-btn').addEventListener('click', () => this.commitApprovedPairs());
    $('prog-start-btn').addEventListener('click', () => this.startProgressiveRounds());
    $('prog-stop-btn').addEventListener('click', () => this.stopProgressiveRounds());
    $('prog-continue-btn')?.addEventListener('click', () => this.approveAndContinueRound());
    $('review-list').addEventListener('input', (e) => {
      const i = Number(e.target.dataset.item);
      if (Number.isNaN(i)) return;
      const item = (this.reviewQueue || [])[i];
      if (!item) return;
      if (e.target.matches('[data-field="q"]')) item.q = e.target.value;
      if (e.target.matches('[data-field="a"]')) item.a = e.target.value;
    });
    $('review-list').addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const item = (this.reviewQueue || [])[Number(btn.dataset.item)];
      if (!item) return;
      if (btn.dataset.act === 'approve') item.approved = true;
      if (btn.dataset.act === 'unapprove') item.approved = false;
      if (btn.dataset.act === 'reject') { item.rejected = true; item.approved = false; }
      if (btn.dataset.act === 'restore') item.rejected = false;
      this.persistReviewQueue();
      this.renderReviewQueue();
    });
    this._testStore = null;
    this.reviewQueue = [];
    this.loadReviewQueueFromStorage();
    this.progressRunning = false;
    this.progressStopRequested = false;
    this.refreshImproveLabUI();
  }

  // Called by app.js when the Improve Lab tab opens.
  refreshImproveLabUI() {
    this.renderTestSetUI();
    this.renderReviewQueue();
  }

  /* ============ feature 1: held-out test set with a score ============ */

  _loadTestStore() {
    if (!this._testStore) {
      try { this._testStore = JSON.parse(localStorage.getItem(TEST_SET_LS) || '{}') || {}; }
      catch (e) { this._testStore = {}; }
    }
    return this._testStore;
  }

  _persistTestStore() {
    try { localStorage.setItem(TEST_SET_LS, JSON.stringify(this._testStore || {})); }
    catch (e) { this.showToast('Could not save the test set (browser storage is full).', 'error'); }
  }

  getTestSetForModel(modelId) {
    const id = modelId || this.activeModelId;
    if (!id) return null;
    const store = this._loadTestStore();
    const ts = store[id];
    return ts && Array.isArray(ts.pairs) && ts.pairs.length ? ts : null;
  }

  clearTestSet() {
    const store = this._loadTestStore();
    delete store[this.activeModelId];
    this._persistTestStore();
    this.renderTestSetUI();
    this.showToast('Held-out test set removed for this model.');
  }

  // Builds the test set by holding out 20% of the corpus's user:/ai: pairs (at least 1,
  // always leaving 80% for training) and REMOVING them from the training corpus, so the
  // model never trains on them. Same corpus → same held-out questions every time.
  buildTestSetFromCorpus() {
    const store = this._loadTestStore();
    const pairs = this.parseDialoguePairs(document.getElementById('training-text').value);
    if (pairs.length < 5) {
      this.showToast('The corpus needs at least 5 user:/ai: pairs to build a test set. Convert paragraphs into pairs with the Q&A Generator first.', 'error');
      return null;
    }
    const held = [], train = [];
    pairs.forEach(p => { (stableHash(normText(p.q)) % 5 === 0 && held.length < MAX_TEST_PAIRS ? held : train).push(p); });
    if (!held.length) { held.push(train.pop()); }
    if (!train.length) { train.push(held.pop()); }

    const keys = new Set(held.map(p => normText(p.q)));
    const filterText = text => String(text || '').replace(/\r\n/g, '\n')
      .split(/\n\s*\n/)
      .filter(block => {
        const m = block.match(/^user\s*:\s*(.+?)(\n|$)/im);
        return !m || !keys.has(normText(m[1]));
      })
      .join('\n\n').trim();

    // Remove the held-out pairs from the corpus text AND from what the model will merge in,
    // otherwise the next Merge & Train would sneak them back into training.
    const active = this.getActiveModelObj();
    const area = document.getElementById('training-text');
    area.value = filterText(area.value);
    if (active && active.engine) {
      active.engine.rawCorpus = filterText(active.engine.rawCorpus);
      if (active.engine.memory) active.engine.memory.indexText(active.engine.rawCorpus);
    }
    this.updateCorpusStats();
    this.saveModelsToStorage();

    store[this.activeModelId] = { pairs: held.slice(0, MAX_TEST_PAIRS), builtAt: Date.now(), history: [], lastResult: null };
    this._persistTestStore();
    this.renderTestSetUI();
    this.showToast(`Test set built: ${held.length} question(s) are now held out and were removed from the training corpus. They are only used for scoring.`);
    this.progLog && this.progLog(`Held-out test set: ${held.length} question(s) reserved (not used for training).`);
    return store[this.activeModelId];
  }

  // Scores the active model against its held-out test set.
  async scoreActiveModel() {
    const active = this.getActiveModelObj();
    const ts = this.getTestSetForModel();
    if (!ts) { this.renderTestSetUI('<b>Build a test set first:</b> it holds out some Q&A pairs so more training can be judged honestly.'); return null; }
    if (!active || !active.engine || !active.engine.isTrained) { this.renderTestSetUI(); return null; }

    const failures = [];
    let correct = 0;
    const status = document.getElementById('prog-status');
    for (let i = 0; i < ts.pairs.length; i++) {
      const p = ts.pairs[i];
      if (i % 5 === 0) {
        if (status) status.textContent = `Scoring ${i + 1}/${ts.pairs.length}…`;
        await sleep(0); // keep the page responsive during scoring
      }
      let got = '';
      try { got = this.generatePreferred(active.engine, p.q, 60, 0.25, 1.3, 8).replace('[[[UNKNOWN_RESPONSE]]]', '').trim(); } catch (e) { got = ''; }
      const f1 = this.scoreBenchmark(got, p.a, 0).f1;
      if (f1 >= PASS_F1) correct++;
      else failures.push({ q: p.q, expected: p.a, got: got || '(no answer)', f1 });
    }

    const total = ts.pairs.length;
    const percent = Math.round((correct / Math.max(1, total)) * 100);
    ts.lastResult = { correct, total, percent, failures, at: Date.now() };
    ts.history = (ts.history || []).concat([{ at: Date.now(), correct, total, percent }]).slice(-20);
    this._persistTestStore();
    this.renderTestSetUI();
    return { correct, total, percent, failures };
  }

  // Called by app-training.js after a training run finishes: re-score automatically.
  async rescoreAfterTraining(modelObj) {
    const ts = this.getTestSetForModel(modelObj && modelObj.id);
    if (!ts) return null;
    const r = await this.scoreActiveModel();
    if (r) this.showToast(`Held-out test score after this run: ${r.correct}/${r.total} (${r.percent}%).`);
    return r;
  }

  renderTestSetUI(notice) {
    const status = document.getElementById('test-status');
    const summary = document.getElementById('test-summary');
    const history = document.getElementById('test-history');
    const fails = document.getElementById('test-failures');
    if (!status) return;
    const ts = this.getTestSetForModel();
    if (!ts) {
      status.innerHTML = notice || 'No held-out test set for this model yet.';
      if (summary) summary.textContent = '';
      if (history) history.innerHTML = '';
      if (fails) fails.innerHTML = '';
      return;
    }
    status.innerHTML = `Held-out test set: <b>${ts.pairs.length}</b> question(s) · built ${new Date(ts.builtAt).toLocaleString()}. These are never trained on.`;
    const r = ts.lastResult;
    if (summary) {
      summary.innerHTML = r
        ? `Latest score: <b class="text-emerald-300">${r.correct}/${r.total}</b> (${r.percent}% answered correctly) · scored ${new Date(r.at).toLocaleTimeString()}`
        : 'Not scored yet — press "Score model now".';
    }
    if (history) {
      history.innerHTML = (ts.history || []).slice(-10).map(h =>
        `<span class="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 ${h.percent >= 50 ? 'text-emerald-300' : 'text-slate-400'}" title="${this.escapeHTML(new Date(h.at).toLocaleString())}">${h.correct}/${h.total} · ${h.percent}%</span>`
      ).join('') || '';
    }
    if (fails) {
      const list = (r && r.failures) || [];
      fails.innerHTML = list.length
        ? `<p class="text-[11px] text-amber-300 font-semibold pt-1">Failed questions (${list.length}) — these are what Progressive Rounds teaches:</p>` +
          list.slice(0, 12).map((f, i) =>
            `<div class="p-2 rounded-lg bg-slate-950 border border-slate-800"><div class="text-[11px] text-slate-200">Q: ${this.escapeHTML(f.q)}</div><div class="text-[10px] text-rose-300 mt-0.5">Model: ${this.escapeHTML(f.got)}</div><div class="text-[10px] text-emerald-300">Expected: ${this.escapeHTML(f.expected)}</div></div>`
          ).join('') + (list.length > 12 ? `<p class="text-[10px] text-slate-500">…and ${list.length - 12} more.</p>` : '')
        : (r ? '<p class="text-[11px] text-emerald-300 pt-1">Every held-out question passed.</p>' : '');
    }
  }

  /* ============ feature 2: review queue for generated pairs ============ */

  loadReviewQueueFromStorage() {
    try { this.reviewQueue = JSON.parse(localStorage.getItem(REVIEW_QUEUE_LS) || '[]') || []; }
    catch (e) { this.reviewQueue = []; }
    if (!Array.isArray(this.reviewQueue)) this.reviewQueue = [];
  }

  persistReviewQueue() {
    try { localStorage.setItem(REVIEW_QUEUE_LS, JSON.stringify((this.reviewQueue || []).slice(0, 120))); }
    catch (e) { /* small list; ignore storage errors */ }
  }

  // Excerpts from the user's own documents (Corpus Vault + model memory) used to seed generation.
  async gatherSeedExcerpts(topic) {
    const out = { excerpts: [], source: 'your documents' };
    try {
      const idx = await this.getRagIndex();
      if (idx && idx.chunks.length && idx.search) {
        idx.search.search(topic || 'overview facts', 4).forEach(h => {
          out.excerpts.push(String(h.chunk.text).slice(0, 500));
        });
        if (out.excerpts.length) return out;
      }
    } catch (e) { /* vault empty or unavailable */ }
    const raw = (this.getActiveModelObj()?.engine?.rawCorpus) || document.getElementById('training-text')?.value || '';
    const paras = this.parseParagraphs(raw);
    (paras.length ? paras : raw.split(/\n\s*\n/)).slice(0, 4).forEach(p => out.excerpts.push(p.replace(/\s+/g, ' ').slice(0, 500)));
    if (out.excerpts.length) out.source = 'the model corpus';
    return out;
  }

  // Generation core shared by the Review Queue button and Progressive Rounds.
  // topics: array of weak questions to target (progressive), or empty for free generation about `topic`.
  async generateCandidatePairs({ count = 10, topic = '', topics = [], seedDocs = true, onStatus = () => {} } = {}) {
    const wanted = Math.min(REVIEW_CAP, Math.max(1, count | 0));
    let seeds = { excerpts: [], source: 'general knowledge' };
    if (seedDocs) {
      onStatus('Loading seed excerpts from your documents…');
      seeds = await this.gatherSeedExcerpts(topic || topics.join('; '));
    }
    const targetList = topics.length
      ? `Answer these questions the user's model got wrong (one pair per question, best answer found in the SOURCE):\n${topics.slice(0, 10).map((q, i) => `${i + 1}. ${q}`).join('\n')}`
      : `Write pairs about this topic: ${topic || 'the SOURCE text in general'}`;

    const system = [
      `You write practice training pairs (study flashcards) for a small model.`,
      targetList,
      `Rules: exactly ${wanted} pairs. Use this exact format and nothing else:`,
      `user: <short question, phrased the way a person would ask>`,
      `ai: <one-sentence answer, max 25 words>`,
      seedDocs && seeds.excerpts.length ? 'Base every answer strictly on the SOURCE text below. Never invent facts that are not in it.' : 'Keep answers short and factual.',
      seeds.excerpts.length ? `\nSOURCE (from ${seeds.source}):\n${seeds.excerpts.join('\n---\n')}` : ''
    ].filter(Boolean).join('\n');

    let raw = [];
    const refId = document.getElementById('panel-model-select')?.value || '__ref_smollm2_360__';
    try {
      onStatus('Assistant writing candidate pairs… (first use downloads the assistant model)');
      const text = await this.runAssistantPlain(refId, system, 'Write the pairs now.', 350);
      raw = this.parseDialoguePairs(text.replace(/^user/gim, '\nuser'));
    } catch (err) {
      onStatus('Assistant unavailable (' + (err.message || err) + '); using rule-based generation from seed text instead.');
      await sleep(10);
    }
    if (!raw.length && seeds.excerpts.length) {
      seeds.excerpts.slice(0, 4).forEach(p => raw.push(...this.qaFromParagraph(p, 2)));
    }
    return raw.slice(0, REVIEW_CAP);
  }

  // Automatic safety filters. Returns { kept, dropped } where kept = [{q, a}],
  // dropped = [{q, reason}].
  classifyCandidatePairs(rawPairs, cap = REVIEW_CAP) {
    const area = document.getElementById('training-text');
    const corpusByQ = new Map();
    this.parseDialoguePairs(area.value).forEach(p => corpusByQ.set(normText(p.q), normText(p.a)));
    this.parseDialoguePairs(this.getActiveModelObj()?.engine?.rawCorpus || '').forEach(p => {
      if (!corpusByQ.has(normText(p.q))) corpusByQ.set(normText(p.q), normText(p.a));
    });

    const answerCounts = new Map();
    rawPairs.forEach(p => { const a = normText(p.a); if (a) answerCounts.set(a, (answerCounts.get(a) || 0) + 1); });

    const seenQ = new Set();
    const ansUsed = new Map();
    const kept = [], dropped = [];
    for (const p of rawPairs) {
      if (kept.length >= cap) { dropped.push({ q: p.q, reason: `cap of ${cap} pairs per round reached` }); continue; }
      const qKey = normText(p.q), aKey = normText(p.a);
      if (wordList(p.q).length < 2 || !qKey) { dropped.push({ q: p.q, reason: 'question too short' }); continue; }
      if (seenQ.has(qKey)) { dropped.push({ q: p.q, reason: 'duplicate question in this batch' }); continue; }
      if (corpusByQ.has(qKey)) {
        dropped.push({ q: p.q, reason: corpusByQ.get(qKey) === aKey ? 'already in the corpus (duplicate)' : 'conflicts with an existing answer' });
        continue;
      }
      if (wordList(p.a).length < 3) { dropped.push({ q: p.q, reason: 'answer too short' }); continue; }
      const used = ansUsed.get(aKey) || 0;
      if (used >= 2) { dropped.push({ q: p.q, reason: 'same answer repeated too many times' }); continue; }
      if (answerCounts.get(aKey) >= 3 && used >= 1) { dropped.push({ q: p.q, reason: 'repeated generic answer' }); continue; }
      seenQ.add(qKey);
      ansUsed.set(aKey, used + 1);
      kept.push({ q: p.q.trim(), a: p.a.trim() });
    }
    return { kept, dropped };
  }

  async generateReviewPairs() {
    const $ = id => document.getElementById(id);
    const btn = $('review-generate-btn'), status = $('review-status');
    if (!btn) return;
    if (this.progressRunning && this.progressStaging) { this.showToast('A Progressive Round is staging pairs right now.', 'error'); return; }
    const topic = $('review-topic').value.trim();
    const count = parseInt($('review-count').value);
    const seedDocs = $('review-seed-docs').checked;
    btn.disabled = true;
    try {
      const raw = await this.generateCandidatePairs({ count, topic, seedDocs, onStatus: s => { status.textContent = s; } });
      if (!raw.length) {
        status.textContent = 'Generation produced no usable pairs. Try importing documents into the Corpus Vault and seeding from them.';
        this.showToast('The assistant produced no usable pairs this time.', 'error');
        return;
      }
      const { kept, dropped } = this.classifyCandidatePairs(raw);
      // Staged pairs never touch the corpus directly: they go to the pending list.
      kept.forEach(p => this.reviewQueue.push({ q: p.q, a: p.a, approved: false, rejected: false, seeded: seedDocs, at: Date.now() }));
      this.persistReviewQueue();
      this.renderReviewQueue();
      const dropLine = dropped.length ? ' · auto-filtered ' + dropped.length + ': ' + [...new Set(dropped.map(d => d.reason))].slice(0, 3).join(', ') : '';
      status.textContent = `${kept.length} pair(s) staged for review${dropLine}.`;
      this.showToast(`${kept.length} candidate pair(s) are waiting for your review. Nothing was added to the corpus yet.`);
    } finally {
      btn.disabled = false;
    }
  }

  renderReviewQueue() {
    const list = document.getElementById('review-list');
    if (!list) return;
    const queue = this.reviewQueue || [];
    const pending = queue.filter(i => !i.approved && !i.rejected).length;
    const approved = queue.filter(i => i.approved && !i.rejected).length;
    const counts = document.getElementById('review-counts');
    if (counts) counts.textContent = `${queue.length} staged · ${pending} waiting · ${approved} approved`;
    const commitBtn = document.getElementById('review-commit-btn');
    if (commitBtn) {
      commitBtn.disabled = approved === 0;
      commitBtn.textContent = `Add ${approved} approved pair(s) to the corpus`;
    }
    if (!queue.length) {
      list.innerHTML = '<p class="text-xs text-slate-500">Nothing staged yet. Generate candidate pairs about a topic — they land here instead of straight into your corpus.</p>';
      return;
    }
    list.innerHTML = queue.map((it, i) => `
      <div class="p-2.5 rounded-xl border ${it.rejected ? 'border-rose-900 bg-rose-950/20 opacity-60' : (it.approved ? 'border-emerald-600/50 bg-emerald-950/20' : 'border-slate-800 bg-slate-950')} space-y-1.5">
        <label class="block text-[10px] text-slate-500">Question${it.seeded ? ' · seeded from your documents' : ''}</label>
        <input data-field="q" data-item="${i}" value="${this.escapeHTML(it.q)}" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-brand-500">
        <label class="block text-[10px] text-slate-500">Answer</label>
        <textarea data-field="a" data-item="${i}" rows="2" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-brand-500">${this.escapeHTML(it.a)}</textarea>
        <div class="flex items-center gap-1.5 pt-0.5">
          ${it.approved ? '<span class="text-[10px] text-emerald-300 font-semibold mr-1">Approved</span>' : ''}
          ${it.rejected ? '<span class="text-[10px] text-rose-300 font-semibold mr-1">Rejected</span>' : ''}
          ${!it.approved && !it.rejected ? `<button data-act="approve" data-item="${i}" class="px-2 py-1 rounded-md bg-emerald-700 hover:bg-emerald-600 text-white text-[10px] font-semibold"><i class="fa-solid fa-check"></i> Approve</button>` : ''}
          ${it.approved && !it.rejected ? `<button data-act="unapprove" data-item="${i}" class="px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px]"><i class="fa-solid fa-rotate-left"></i> Unapprove</button>` : ''}
          ${!it.rejected ? `<button data-act="reject" data-item="${i}" class="px-2 py-1 rounded-md bg-rose-950 hover:bg-rose-900 text-rose-300 text-[10px]" title="Reject"><i class="fa-solid fa-xmark"></i> Reject</button>` : ''}
          ${it.rejected ? `<button data-act="restore" data-item="${i}" class="px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px]"><i class="fa-solid fa-rotate-left"></i> Restore</button>` : ''}
        </div>
      </div>
    `).join('');
  }

  approveAllClean() {
    let n = 0;
    (this.reviewQueue || []).forEach(it => { if (!it.rejected) { it.approved = true; n++; } });
    this.persistReviewQueue();
    this.renderReviewQueue();
    this.showToast(n ? `Approved ${n} pairs. Nothing is in your corpus until you press "Add approved pairs".` : 'Nothing waiting to approve.');
  }

  rejectAllStaged() {
    let n = 0;
    (this.reviewQueue || []).forEach(it => { if (!it.rejected && !it.approved) { it.rejected = true; n++; } });
    this.persistReviewQueue();
    this.renderReviewQueue();
    this.showToast(n ? `Rejected ${n} staged pairs.` : 'No unapproved pairs to reject.');
  }

  // Approved pairs become real corpus text (this is the only door into the corpus).
  commitApprovedPairs() {
    const approved = (this.reviewQueue || []).filter(i => i.approved && !i.rejected);
    if (!approved.length) { this.showToast('Approve at least one pair first.', 'error'); return; }
    const area = document.getElementById('training-text');
    area.value = this.mergeCorpus(area.value, this.formatPairs(approved), false);
    this.reviewQueue = (this.reviewQueue || []).filter(i => !(i.approved && !i.rejected));
    this.persistReviewQueue();
    this.updateCorpusStats();
    this.renderReviewQueue();
    this.showToast(`${approved.length} pair(s) added to the training corpus. Press Merge & Train (or run a Progressive Round) to learn them.`);
  }

  /* ============ feature 3: progressive rounds with keep-or-revert ============ */

  profileLimitsFor(profile) {
    return {
      phone: { vocabLimit: 240, maxExamples: 450, epochCap: 20, maxLr: 0.09, label: 'Phone / low-memory CPU' },
      balanced: { vocabLimit: 520, maxExamples: 1100, epochCap: 45, maxLr: 0.16, label: 'Balanced CPU' },
      large: { vocabLimit: 900, maxExamples: 2200, epochCap: 80, maxLr: 0.16, label: 'Large CPU' }
    }[profile || 'balanced'];
  }

  progLog(message, tone = 'info') {
    const log = document.getElementById('prog-log');
    if (!log) return;
    const color = { info: 'text-slate-300', good: 'text-emerald-300', warn: 'text-amber-300', error: 'text-rose-300' }[tone] || 'text-slate-300';
    const row = document.createElement('div');
    row.className = color + ' text-[11px]';
    row.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  waitForRoundApproval() {
    return new Promise(resolve => {
      this._roundApprovalResolver = resolve;
      const btn = document.getElementById('prog-continue-btn');
      if (btn) btn.classList.remove('hidden');
    });
  }

  approveAndContinueRound() {
    const btn = document.getElementById('prog-continue-btn');
    if (btn) btn.classList.add('hidden');
    const r = this._roundApprovalResolver;
    this._roundApprovalResolver = null;
    if (r) r();
  }

  stopProgressiveRounds() {
    if (!this.progressRunning) { this.showToast('No progressive round loop is running.'); return; }
    this.progressStopRequested = true;
    this.stopRequested = true; // also reaches the training job this.roundWorker…
    const w = this.trainingWorker;
    if (w) w.postMessage({ type: 'stop' });
    const r = this._roundApprovalResolver;
    if (r) { this._roundApprovalResolver = null; document.getElementById('prog-continue-btn')?.classList.add('hidden'); r(); }
    const status = document.getElementById('prog-status');
    if (status) status.textContent = 'Stopping…';
    this.progLog('Stop requested — finishing the current step, then cancelling remaining rounds.', 'warn');
  }

  async startProgressiveRounds() {
    const $ = id => document.getElementById(id);
    if (this.progressRunning) { this.showToast('A progressive loop is already running.', 'error'); return; }
    if (this.isTraining) { this.showToast('Regular training is running. Wait for it to finish first.', 'error'); return; }
    const active = this.getActiveModelObj();
    if (!active || !active.engine || !active.engine.isTrained) { this.showToast('Train the model once normally before running Progressive Rounds.', 'error'); return; }

    let ts = this.getTestSetForModel();
    if (!ts) {
      ts = this.buildTestSetFromCorpus();
      if (!ts) return;
    }

    const maxRounds = Math.min(10, Math.max(1, parseInt($('prog-rounds').value) || 3));
    const corpusCap = Math.max(2000, parseInt($('prog-cap').value) || 20000);
    const approvalMode = $('prog-approval').checked;
    const profile = document.getElementById('training-profile').value || 'balanced';
    const lim = this.profileLimitsFor(profile);

    this.progressRunning = true;
    this.progressStopRequested = false;
    this.isTraining = true; // block ordinary training while rounds run, so weights never race
    $('prog-start-btn').disabled = true;
    $('prog-stop-btn').disabled = false;
    $('prog-log').innerHTML = '';
    $('prog-progress-bar').style.width = '0%';
    this.progLog(`Progressive loop: up to ${maxRounds} round(s) · ${lim.label} profile · corpus cap ${corpusCap.toLocaleString()} chars · ${approvalMode ? 'pausing for your approval each round' : 'fully automatic'}.`);

    try {
      const baseline = await this.scoreActiveModel();
      if (!baseline) { this.progLog('No held-out test set could be built — stopping.', 'error'); return; }
      let lastScore = baseline;
      this.progLog(`Baseline score: ${baseline.correct}/${baseline.total} (${baseline.percent}%) answered correctly.`, 'good');
      let noImprovement = 0;

      for (let round = 1; round <= maxRounds; round++) {
        if (this.progressStopRequested) { this.progLog('Stopped by you before round ' + round + '.', 'warn'); break; }
        if (noImprovement >= 2) { this.progLog('Score has not improved for 2 rounds in a row — loop stopped.', 'warn'); break; }

        const failures = (ts.lastResult && ts.lastResult.failures) || [];
        if (!failures.length) { this.progLog('Every held-out question already passes — nothing to teach.', 'good'); break; }
        if ((active.engine.rawCorpus || '').length > corpusCap) { this.progLog(`Corpus cap of ${corpusCap.toLocaleString()} characters reached — stopping before round ${round}.`, 'warn'); break; }

        const status = $('prog-status');
        if (status) status.textContent = `Round ${round}: generating targeted pairs for ${failures.length} weak question(s)…`;
        this.progLog(`Round ${round}: generating targeted pairs for ${failures.length} weak question(s)…`);

        const raw = await this.generateCandidatePairs({
          count: Math.min(REVIEW_CAP, failures.length * 2),
          topic: 'correct answers to weak questions',
          topics: failures.map(f => f.q),
          seedDocs: true,
          onStatus: s => { if (status) status.textContent = `Round ${round}: ${s}`; }
        });
        if (this.progressStopRequested) { this.progLog('Stopped by you during generation.', 'warn'); break; }
        if (!raw.length) { this.progLog(`Round ${round}: generation produced no usable pairs.`, 'warn'); noImprovement++; continue; }

        const { kept, dropped } = this.classifyCandidatePairs(raw);
        this.progLog(`Round ${round}: ${kept.length} candidate(s) passed the filters` + (dropped.length ? ` (${dropped.length} auto-filtered: ${[...new Set(dropped.map(d => d.reason))].slice(0, 2).join(', ')})` : '') + '.');

        let approved = kept;
        if (approvalMode) {
          this.progressStaging = true;
          kept.forEach(p => this.reviewQueue.push({ q: p.q, a: p.a, approved: false, rejected: false, seeded: true, at: Date.now() }));
          this.persistReviewQueue();
          this.renderReviewQueue();
          this.progressStaging = false;
          this.progLog(`Round ${round}: ${kept.length} pair(s) staged in the Review Queue below. Approve, edit or reject them, then press "Approve & continue round". Point is: nothing trains until you say so.`, 'warn');
          if (status) status.textContent = `Round ${round}: waiting for your review…`;
          await this.waitForRoundApproval();
          if (this.progressStopRequested) { this.progLog('Stopped by you while waiting for approval.', 'warn'); break; }
          approved = (this.reviewQueue || []).filter(i => i.approved && !i.rejected).map(i => ({ q: i.q, a: i.a }));
          this.reviewQueue = (this.reviewQueue || []).filter(i => !approved.some(a => a.q === i.q && a.a === i.a));
          this.persistReviewQueue();
          this.renderReviewQueue();
        }
        if (!approved.length) { this.progLog(`Round ${round}: nothing approved to train on — skipping training.`, 'warn'); continue; }

        // Snapshot everything, then add the pairs and train.
        const snapshotJson = active.engine.toJSON({ weights: 'f32' });
        const area = document.getElementById('training-text');
        const corpusBefore = area.value;
        area.value = this.mergeCorpus(corpusBefore, this.formatPairs(approved), false);
        this.updateCorpusStats();
        const trainingText = this.mergeCorpus(active.engine.rawCorpus, area.value);

        const cfg = {
          epochs: Math.min(parseInt($('prog-epochs').value) || 20, lim.epochCap),
          learningRate: Math.min(0.08, lim.maxLr),
          contextSize: active.engine.contextSize,
          hiddenDim: active.engine.hiddenDim,
          embDim: active.engine.embDim,
          augment: true,
          validate: true,
          earlyStop: true,
          lrFloor: 0.2,
          vocabLimit: lim.vocabLimit,
          maxExamples: lim.maxExamples
        };
        this.progLog(`Round ${round}: training on ${approved.length} new pair(s) · epoch budget ${cfg.epochs}…`);
        if (status) status.textContent = `Round ${round}: training…`;
        const result = await this.runTrainingJob({ engineType: active.type, text: trainingText, config: cfg }, info => {
          $('prog-progress-bar').style.width = (info.fraction * 100) + '%';
        }, { primary: true });

        const applied = this.applyTrainingResult(active, result, trainingText, {});
        if (!applied) {
          area.value = corpusBefore;
          this.progLog(`Round ${round}: training produced no finished epochs (stopped or diverged); model unchanged.`, 'warn');
          noImprovement++;
          continue;
        }
        this.saveModelsToStorage();
        this.updateStatsDisplay();

        const after = await this.scoreActiveModel();
        // Keep only when strictly better on the held-out score.
        if (after.correct > lastScore.correct) {
          this.progLog(`Round ${round} KEPT: score went up ${lastScore.correct}/${lastScore.total} → ${after.correct}/${after.total} (${after.percent}%).`, 'good');
          lastScore = after;
          noImprovement = 0;
          if (status) status.textContent = `Round ${round}: kept (score now ${after.percent}%)`;
        } else {
          active.engine.fromJSON(snapshotJson);
          area.value = corpusBefore;
          this.updateCorpusStats();
          await this.saveModelsToStorage();
          this.updateStatsDisplay();
          noImprovement++;
          this.progLog(`Round ${round} REVERTED: score ${lastScore.correct}/${lastScore.total} → ${after.correct}/${after.total}. The round's pairs and weights were rolled back.`, 'warn');
          if (status) status.textContent = `Round ${round}: reverted (no improvement)`;
        }
        $('prog-progress-bar').style.width = '0%';
        await sleep(0);
        if (this.progressStopRequested) { this.progLog('Stopped by you after round ' + round + '.', 'warn'); break; }
      }

      const finalStatus = $('prog-status');
      if (!this.progressStopRequested) {
        this.progLog(`Loop finished. Final held-out score: ${lastScore.correct}/${lastScore.total} (${lastScore.percent}%), started at ${baseline.percent}%.`, 'good');
        if (finalStatus) finalStatus.textContent = `Done · score ${lastScore.correct}/${lastScore.total} (${lastScore.percent}%)`;
        this.showToast(`Progressive training finished. Held-out score: ${lastScore.percent}%.`);
      }
    } catch (err) {
      this.progLog('Progressive loop failed: ' + (err.message || err), 'error');
      this.showToast('Progressive training failed: ' + (err.message || 'unexpected error'), 'error');
    } finally {
      this.progressRunning = false;
      this.progressStaging = false;
      this.isTraining = false;
      this.progressStopRequested = false;
      this.stopRequested = false;
      $('prog-start-btn').disabled = false;
      $('prog-stop-btn').disabled = true;
      document.getElementById('prog-continue-btn')?.classList.add('hidden');
      this.saveModelsToStorage();
      this.refreshImproveLabUI();
    }
  }
}
