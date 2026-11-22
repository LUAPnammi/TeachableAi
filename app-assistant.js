// Studio Assistant side panel, local pretrained models (Transformers.js), document search and actions.
// Methods here are copied onto AIApp in main.js, so `this` is the app.
import { DocSearch, ragChunkText } from './doc-search.js';
import { STUDIO_BASE_NOTES } from './studio-notes.js';

export class AppAssistant {
  getLocalReferenceDefinition(id) {
    const defs = {
      '__ref_smollm2_360__': {
        key: 'smollm2-360',
        name: 'SmolLM2 360M Instruct',
        repos: ['onnx-community/SmolLM2-360M-Instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct'],
        params: '360M'
      },
      '__ref_qwen05__': {
        key: 'qwen05',
        name: 'Qwen2.5 0.5B Instruct',
        repos: ['onnx-community/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'],
        params: '494M'
      },
      '__ref_smollm2_135__': {
        key: 'smollm2-135',
        name: 'SmolLM2 135M Instruct',
        repos: ['onnx-community/SmolLM2-135M-Instruct', 'HuggingFaceTB/SmolLM2-135M-Instruct'],
        params: '135M'
      }
    };
    if (id.startsWith('__ref_')) return defs[id] || defs['__ref_smollm2_360__'];
    return Object.values(defs).find(x => x.key === id) || defs['__ref_smollm2_360__'];
  }

  buildStudioContext(question) {
    const splitChunks = txt => String(txt || '').split(/\n(?=[A-Z][A-Z_ -]{2,}:)/).map(x => x.trim()).filter(Boolean);
    const stop = new Set(['the','and','what','how','can','you','this','that','with','for','are','does','your','studio','about','tell','please']);
    const stem = w => w.slice(0, 5);
    const q = new Set((String(question || '').toLowerCase().match(/[a-z0-9]{3,}/g) || []).filter(w => !stop.has(w)).map(stem));
    const all = [...splitChunks(STUDIO_BASE_NOTES).map(t => ({ t, base: true })), ...splitChunks(this.studioKnowledge).map(t => ({ t, base: false }))];
    const ranked = all.map((c, i) => {
      let score = 0;
      for (const w of (c.t.toLowerCase().match(/[a-z0-9]{3,}/g) || [])) if (q.has(stem(w))) score++;
      if (c.t.startsWith('OVERVIEW:')) score += 1000;
      return { ...c, score, i };
    }).sort((a, b) => b.score - a.score || a.i - b.i);
    const picked = []; let used = 0;
    for (const c of ranked) {
      if (picked.length >= 4 || (picked.length && used + c.t.length > 2600)) continue;
      if (c.score === 0 && picked.length >= 2) break;
      picked.push(c.t); used += c.t.length;
    }
    return picked.join('\n\n');
  }

  tidyAssistantAnswer(text, maxChars = 300) {
    let out = String(text || '').replace(/\*\*|__|`|^#+\s*/gm, '').replace(/\s+/g, ' ').trim();
    const sentences = out.match(/[^.!?]+[.!?]+(?=\s|$)/g);
    if (sentences && sentences.length) out = sentences.slice(0, 2).join(' ').trim();
    if (out.length > maxChars) {
      const cut = out.slice(0, maxChars);
      const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '));
      out = end > 60 ? cut.slice(0, end + 1) : cut.replace(/\s+\S*$/, '') + '…';
    }
    return out;
  }

  async loadTransformersLibrary() {
    if (this.transformersModule) return this.transformersModule;
    if (!this.transformersModulePromise) {
      const cdns = [
        'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2',
        'https://esm.sh/@huggingface/transformers@3.7.2',
        'https://unpkg.com/@huggingface/transformers@3.7.2'
      ];
      this.transformersModulePromise = (async () => {
        let lastErr = null;
        for (const url of cdns) {
          try {
            const mod = await import(url);
            this.transformersModule = mod;
            mod.env.allowLocalModels = false;
            mod.env.useBrowserCache = true;
            return mod;
          } catch (err) {
            lastErr = err;
          }
        }
        throw new Error('Could not load browser inference library from CDN: ' + (lastErr?.message || 'network error'));
      })().catch(err => {
        this.transformersModulePromise = null;
        throw err;
      });
    }
    return this.transformersModulePromise;
  }

  async prepareLocalReference(key) {
    const ref = this.getLocalReferenceDefinition(key);
    const status = document.getElementById('status-' + ref.key);
    const bar = document.getElementById('progress-' + ref.key);
    const btn = document.querySelector(`[data-local-model-download="${ref.key}"]`);

    if (this.localReferencePipelines[ref.key]) {
      if (status) status.textContent = 'Ready · loaded in this tab';
      this.showToast(ref.name + ' is ready. Open the Assistant panel to chat.');
      return this.localReferencePipelines[ref.key];
    }
    if (this.localReferenceLoading[ref.key]) return this.localReferenceLoading[ref.key];

    if (btn) { btn.disabled = true; btn.textContent = 'Downloading / loading…'; }
    if (status) status.textContent = 'Loading Transformers.js and checking device support…';
    if (bar) bar.style.width = '8%';

    this.localReferenceLoading[ref.key] = (async () => {
      try {
        const { pipeline, env } = await this.loadTransformersLibrary();
        env.allowLocalModels = false;
        env.useBrowserCache = true;

        const hasWebGPU = Boolean(navigator.gpu && window.isSecureContext);
        const attempts = [];
        if (hasWebGPU) {
          attempts.push({ device: 'webgpu', dtype: 'q4f16' });
          attempts.push({ device: 'webgpu', dtype: 'q4' });
        }
        attempts.push({ device: 'wasm', dtype: 'q4' });
        attempts.push({ device: 'wasm', dtype: 'q8' });
        attempts.push({ device: 'wasm', dtype: undefined });

        let pipe = null;
        let usedAttempt = null;
        let lastError = null;

        for (const repo of ref.repos) {
          for (const att of attempts) {
            try {
              if (status) status.textContent = `Fetching ${repo} (${att.device}, ${att.dtype || 'default'})…`;
              const opts = {
                device: att.device,
                progress_callback: (info) => {
                  if (!info) return;
                  let pct = 0;
                  if (typeof info.progress === 'number') pct = Math.max(5, Math.min(99, info.progress));
                  if (bar && pct) bar.style.width = pct + '%';
                  if (status && info.file) {
                    status.textContent = `Downloading ${info.file}${pct ? ' · ' + Math.round(pct) + '%' : ''}`;
                  }
                }
              };
              if (att.dtype) opts.dtype = att.dtype;
              pipe = await pipeline('text-generation', repo, opts);
              usedAttempt = { repo, ...att };
              break;
            } catch (err) {
              lastError = err;
            }
          }
          if (pipe) break;
        }

        if (!pipe) {
          throw lastError || new Error('All download configurations failed.');
        }

        this.localReferencePipelines[ref.key] = pipe;
        if (bar) bar.style.width = '100%';
        if (status) {
          status.textContent = `Ready · cached by browser · ${usedAttempt.device.toUpperCase()}${usedAttempt.dtype ? ' (' + usedAttempt.dtype + ')' : ''}`;
        }
        if (btn) btn.textContent = 'Ready · use in Assistant panel';
        this.showToast(ref.name + ' is downloaded and ready in the Assistant panel!');
        return pipe;
      } catch (err) {
        if (bar) bar.style.width = '0%';
        if (status) status.textContent = 'Download failed: ' + err.message;
        if (btn) btn.textContent = 'Retry download';
        this.showToast(ref.name + ' download failed: ' + err.message, 'error');
        throw err;
      } finally {
        delete this.localReferenceLoading[ref.key];
        if (btn) btn.disabled = false;
      }
    })();

    return this.localReferenceLoading[ref.key];
  }

  async requestLocalReference(id, prompt, history = [], opts = {}) {
    const ref = this.getLocalReferenceDefinition(id);
    const pipe = this.localReferencePipelines[ref.key] || await this.prepareLocalReference(ref.key);
    const mod = this.transformersModule || {};
    const context = this.buildStudioContext(prompt);
    const systemContent = [
      `You are the Teachable AI Studio assistant (${ref.name}), a friendly helper that lives inside the app.`,
      `RULES: Reply in 1-3 short sentences. Be direct and practical, name the exact tab or button, and never ramble, list, or repeat yourself. Never invent features that are not in the STUDIO NOTES.`,
      `You also have ASSISTANT ACTIONS: the app itself can create a model, start training, run evaluation, clean or check the corpus, export the active model, read a screen with OCR, and switch tabs when the user asks in plain words (examples: "create a model called Tutor and train it", "clean my corpus", "read my screen"). If the user asks you to change or build something, tell them to phrase it as one of those plain requests and the app will do it for them.`,
      `For studio questions use ONLY the STUDIO NOTES below. If the notes do not cover it, say you are not sure and point to the closest tab. For general questions answer briefly. Never claim you can change your own weights.`,
      ``,
      `STUDIO NOTES:`,
      context,
      ...(opts.docContext ? ['', `DOCUMENT EXCERPTS from the user's own files and model memory (use them first when they answer the question, and cite like [1]):`, opts.docContext] : [])
    ].join('\n');
    const shots = [
      { role: 'user', content: 'how do I teach my model new facts?' },
      { role: 'assistant', content: 'Open Train Studio, paste your facts or user:/ai: pairs, then press Merge & Train. You can also add single facts in the Memory tab.' },
      { role: 'user', content: 'what does temperature do?' },
      { role: 'assistant', content: 'Temperature sets creativity: low values give steady, predictable replies and high values give more varied ones.' },
      { role: 'user', content: 'make me a space bot and teach it space facts' },
      { role: 'assistant', content: "Just say: create a model called Space Bot and train it. The app builds it, trains it on whatever is in Train Studio, and you can load the space pack from Corpus Library first." }
    ];
    const genOpts = { max_new_tokens: 60, do_sample: true, temperature: 0.25, top_p: 0.85, repetition_penalty: 1.1 };

    let acc = '', tFirst = null, stopper = null;
    const t0 = performance.now();
    if (opts.onToken && mod.TextStreamer && pipe.tokenizer) {
      try {
        if (mod.InterruptableStoppingCriteria) {
          stopper = new mod.InterruptableStoppingCriteria();
          genOpts.stopping_criteria = stopper;
          if (opts.controller) opts.controller.interrupt = () => stopper.interrupt();
        }
        genOpts.streamer = new mod.TextStreamer(pipe.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: chunk => {
            if (!chunk) return;
            if (tFirst === null) tFirst = performance.now();
            acc += chunk;
            opts.onToken(acc, performance.now() - t0);
            if (stopper && /^\s*[^.!?]+[.!?]+\s+[^.!?]+[.!?]+\s/.test(acc.replace(/\s+/g, ' '))) stopper.interrupt();
          }
        });
      } catch (streamErr) { delete genOpts.streamer; delete genOpts.stopping_criteria; stopper = null; }
    }

    let answer = '';
    try {
      const messages = [{ role: 'system', content: systemContent }, ...shots, ...history, { role: 'user', content: prompt }];
      const result = await pipe(messages, genOpts);
      const generated = result?.[0]?.generated_text;
      if (Array.isArray(generated)) {
        const last = generated[generated.length - 1];
        answer = typeof last === 'string' ? last : (last?.content || '');
      } else if (typeof generated === 'string') {
        answer = generated;
      }
    } catch (chatTemplateErr) {
      acc = ''; tFirst = null;
      if (stopper && stopper.reset) stopper.reset();
      const plainPrompt = `${systemContent}\n\nUser: ${shots[0].content}\nAssistant: ${shots[1].content}\n\nUser: ${prompt}\nAssistant:`;
      const fallbackResult = await pipe(plainPrompt, genOpts);
      const genText = fallbackResult?.[0]?.generated_text || '';
      answer = typeof genText === 'string'
        ? genText.slice(plainPrompt.length).split(/\nUser:/)[0].trim()
        : '';
    }

    const ms = performance.now() - t0;
    const rawText = acc || answer;
    answer = this.tidyAssistantAnswer(answer || acc);
    if (!answer) throw new Error('The assistant returned an empty response. Try a shorter prompt.');
    let tokens = 0;
    try { tokens = pipe.tokenizer.encode(rawText).length; } catch (e) { tokens = Math.max(1, Math.round(rawText.split(/\s+/).filter(Boolean).length * 1.3)); }
    const ttftMs = tFirst !== null ? tFirst - t0 : null;
    const decodeMs = ttftMs !== null ? ms - ttftMs : ms;
    const tps = (ttftMs !== null && tokens > 1 && decodeMs > 0) ? (tokens - 1) / (decodeMs / 1000) : tokens / Math.max(ms, 1) * 1000;
    Object.assign(opts.stats || {}, { tokens, ms, ttftMs, tps, source: 'generated' });
    return answer;
  }

  initAssistantPanel() {
    const P = this.panel = { messages: [], busy: false, controller: null, live: null };
    const panel = document.getElementById('assistant-panel');
    if (!panel) return;
    try { P.messages = JSON.parse(this.lsGet('chat', '[]')) || []; } catch (e) { P.messages = []; }
    const sel = document.getElementById('panel-model-select');
    sel.value = this.lsGet('model', '__ref_smollm2_360__');
    if (!sel.value) sel.selectedIndex = 0;
    document.getElementById('panel-rag-toggle').checked = this.lsGet('rag', '1') === '1';
    const w = Math.min(720, Math.max(300, parseInt(this.lsGet('width', '400')) || 400));
    panel.style.setProperty('--panel-w', w + 'px');
    const stored = this.lsGet('open', null);
    this.setPanelOpen(stored === null ? window.innerWidth >= 1200 : stored === '1', false);

    document.getElementById('assistant-toggle-btn').addEventListener('click', () => this.setPanelOpen(panel.classList.contains('hidden')));
    document.getElementById('panel-close-btn').addEventListener('click', () => this.setPanelOpen(false));
    document.getElementById('panel-new-btn').addEventListener('click', () => {
      if (P.busy) { this.showToast('Stop the current reply first.', 'error'); return; }
      P.messages = [];
      this.persistPanel();
      this.panelRender();
    });
    sel.addEventListener('change', () => { this.lsSet('model', sel.value); this.updatePanelStatus(); });
    document.getElementById('panel-rag-toggle').addEventListener('change', e => { this.lsSet('rag', e.target.checked ? '1' : '0'); this.updateRagInfo(); });
    document.getElementById('panel-download-btn').addEventListener('click', () => {
      const ref = this.getLocalReferenceDefinition(sel.value);
      this.watchPanelDownload(ref.key);
      this.prepareLocalReference(ref.key).catch(() => {}).finally(() => this.updatePanelStatus());
    });
    document.getElementById('panel-send-btn').addEventListener('click', () => this.panelSend());
    document.getElementById('panel-screen-btn')?.addEventListener('click', () => this.runScreenRead());
    document.getElementById('panel-shot-btn')?.addEventListener('click', () => document.getElementById('panel-screenshot-input')?.click());
    document.getElementById('panel-screenshot-input')?.addEventListener('change', e => this.ocrScreenshotFile(e));
    const input = document.getElementById('panel-input');
    input.addEventListener('paste', e => {
      const items = Array.from((e.clipboardData && e.clipboardData.items) || []);
      const imgItem = items.find(it => it.type && it.type.startsWith('image/'));
      if (imgItem) {
        const file = imgItem.getAsFile();
        if (file) { e.preventDefault(); this.ocrScreenshotFile({ target: { files: [file] } }); }
      }
    });
    input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!P.busy) this.panelSend(); } });
    input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 120) + 'px'; });
    document.getElementById('panel-resize').addEventListener('mousedown', e => {
      e.preventDefault();
      const right = panel.getBoundingClientRect().right;
      const move = ev => panel.style.setProperty('--panel-w', Math.min(720, Math.max(300, right - ev.clientX)) + 'px');
      const up = () => {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        document.body.style.userSelect = '';
        this.lsSet('width', String(parseInt(panel.style.getPropertyValue('--panel-w')) || 400));
      };
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
    document.addEventListener('keydown', e => {
      if (e.altKey && !e.ctrlKey && !e.metaKey && e.code === 'KeyA') { e.preventDefault(); this.setPanelOpen(panel.classList.contains('hidden')); }
    });
    this.panelRender();
    this.updatePanelStatus();
    this.updateRagInfo();
  }

  setPanelOpen(open, persist = true) {
    const panel = document.getElementById('assistant-panel'), btn = document.getElementById('assistant-toggle-btn');
    if (!panel) return;
    panel.classList.toggle('hidden', !open);
    panel.classList.toggle('flex', open);
    if (btn) {
      btn.classList.toggle('border-cyan-500/60', open);
      btn.classList.toggle('bg-cyan-900/30', open);
    }
    if (persist) this.lsSet('open', open ? '1' : '0');
    if (open) {
      this.updatePanelStatus();
      this.updateRagInfo();
      const box = document.getElementById('panel-messages');
      if (box) box.scrollTop = box.scrollHeight;
      if (window.innerWidth >= 768) setTimeout(() => document.getElementById('panel-input')?.focus(), 50);
    }
  }

  updatePanelStatus() {
    const sel = document.getElementById('panel-model-select');
    if (!sel) return;
    const ref = this.getLocalReferenceDefinition(sel.value);
    const ready = !!this.localReferencePipelines[ref.key], loading = !!this.localReferenceLoading[ref.key];
    const hub = document.getElementById('status-' + ref.key)?.textContent || '';
    const status = document.getElementById('panel-status'), btn = document.getElementById('panel-download-btn');
    if (status) status.textContent = ready ? ref.name + ' · ready, running locally' : (loading ? hub : (hub.startsWith('Download failed') ? hub : 'Not loaded yet · first message downloads it (cached afterwards)'));
    if (btn) { btn.classList.toggle('hidden', ready); btn.disabled = loading; btn.textContent = loading ? 'Loading…' : 'Download'; }
  }

  watchPanelDownload(key) {
    if (this._panelWatch) clearInterval(this._panelWatch);
    this._panelWatch = setInterval(() => {
      this.updatePanelStatus();
      const loading = !!this.localReferenceLoading[key];
      if (loading && this.panel?.live) this.panel.live.setText(document.getElementById('status-' + key)?.textContent || 'Loading model…');
      if (!loading) { clearInterval(this._panelWatch); this._panelWatch = null; this.updatePanelStatus(); }
    }, 500);
  }

  async getRagIndex() {
    let docs = [];
    try { docs = await this.getCorpusDocs(); } catch (e) {}
    const m = this.getActiveModelObj();
    const key = docs.map(d => d.id + ':' + (d.updatedAt || 0)).join('|') + '#' + (m ? m.id + ':' + (m.engine.rawCorpus || '').length : '');
    if (this.ragIndex && this.ragIndex.key === key) return this.ragIndex;
    const chunks = [];
    docs.forEach(d => ragChunkText(d.text).forEach(t => chunks.push({ title: d.name, text: t, kind: 'file' })));
    if (m && m.engine.rawCorpus) ragChunkText(m.engine.rawCorpus).forEach(t => chunks.push({ title: 'Model memory: ' + m.name, text: t, kind: 'memory' }));
    this.ragIndex = { key, chunks, docCount: docs.length, search: new DocSearch(chunks) };
    return this.ragIndex;
  }

  async updateRagInfo() {
    const el = document.getElementById('panel-rag-info');
    if (!el) return;
    let n = 0;
    try { n = (await this.getCorpusDocs()).length; } catch (e) {}
    el.textContent = `${n} file${n === 1 ? '' : 's'} + model memory`;
  }

  async searchDocs(prompt) {
    const out = { context: '', sources: [], noDocsAsked: false };
    const idx = await this.getRagIndex();
    const asksDocs = /\b(my|the|these|those|our)\s+(docs?|documents?|files?|datasets?|notes)\b/i.test(prompt);
    if (!idx.docCount && asksDocs) { out.noDocsAsked = true; return out; }
    if (!idx.chunks.length) return out;
    let hits = [];
    if (asksDocs && /\b(summar|overview|what('?s| is) in)/i.test(prompt)) {
      const seen = new Set();
      for (const c of idx.chunks) if (c.kind === 'file' && !seen.has(c.title)) { seen.add(c.title); hits.push({ chunk: c }); if (hits.length >= 3) break; }
    } else hits = idx.search.search(prompt, 3);
    let total = 0;
    hits.forEach(h => {
      const text = h.chunk.text.length > 450 ? h.chunk.text.slice(0, 450) + '…' : h.chunk.text;
      if (total + text.length > 1400) return;
      total += text.length;
      out.sources.push({ n: out.sources.length + 1, title: h.chunk.title, excerpt: text });
    });
    out.context = out.sources.map(x => `[${x.n}] (${x.title}) ${x.excerpt}`).join('\n');
    return out;
  }

  persistPanel() { this.lsSet('chat', JSON.stringify((this.panel?.messages || []).slice(-60))); }

  panelRender() {
    const box = document.getElementById('panel-messages');
    if (!box || !this.panel) return;
    const msgs = this.panel.messages;
    if (!msgs.length) {
      const tabNames = { 'chat-tab': 'Chat Studio', 'train-tab': 'Train Studio', 'library-tab': 'Corpus Library', 'models-tab': 'Models Hub', 'eval-tab': 'Evaluation', 'memory-tab': 'Memory', 'guide-tab': 'How To Guide' };
      const chips = ['What can you do?', 'create a model called Tutor and train it', 'How do I train a model?', 'Summarize my documents'];
      box.innerHTML = `<div class="text-center py-6 space-y-3"><div class="w-10 h-10 mx-auto rounded-xl bg-gradient-to-tr from-brand-600 to-accent-500 flex items-center justify-center text-white"><i class="fa-solid fa-wand-magic-sparkles"></i></div><div class="text-sm font-bold text-white">Studio Assistant</div><p class="text-[11px] text-slate-400 leading-relaxed px-2">Ask how anything in the studio works, or turn on document search to ask about your imported files. Replies are short and stream live.</p><div class="flex flex-col gap-1.5 pt-1">${chips.map(c => `<button data-panel-chip="${this.escapeHTML(c)}" class="text-left px-3 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[11px] text-slate-200 transition">${this.escapeHTML(c)}</button>`).join('')}</div></div>`;
      box.querySelectorAll('[data-panel-chip]').forEach(b => b.addEventListener('click', () => this.panelSend(b.dataset.panelChip)));
      return;
    }
    const last = msgs.length - 1;
    box.innerHTML = msgs.map((m, i) => {
      if (m.role === 'user') return `<div class="flex justify-end"><div class="max-w-[88%] p-2.5 rounded-2xl rounded-tr-sm bg-accent-600 text-white leading-relaxed break-words whitespace-pre-wrap">${this.escapeHTML(m.text)}</div></div>`;
      const sources = (m.sources || []).map(x => `<span title="${this.escapeHTML(x.excerpt)}" class="px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/25 text-cyan-300 text-[10px] max-w-full truncate">[${x.n}] ${this.escapeHTML(x.title)}</span>`).join('');
      return `<div class="flex gap-2"><div class="w-6 h-6 rounded-md bg-brand-600/30 border border-brand-500/40 text-brand-400 flex items-center justify-center shrink-0 text-[9px] font-mono font-bold">AI</div><div class="flex-1 min-w-0 space-y-1"><div class="p-2.5 rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800 text-slate-200 leading-relaxed whitespace-pre-wrap break-words">${this.escapeHTML(String(m.text).trim())}</div>${sources ? `<div class="flex flex-wrap gap-1"><span class="text-[10px] text-slate-500 self-center">Sources</span>${sources}</div>` : ''}<div class="flex items-center gap-2.5 px-1 text-[10px] text-slate-500 flex-wrap"><button data-panel-copy="${i}" class="hover:text-cyan-300 flex items-center gap-1"><i class="fa-regular fa-copy"></i> Copy</button>${i === last ? '<button data-panel-regen="1" class="hover:text-emerald-300 flex items-center gap-1"><i class="fa-solid fa-rotate"></i> Regenerate</button>' : ''}${m.stats ? `<span class="font-mono"><i class="fa-solid fa-bolt text-amber-400"></i> ${this.escapeHTML(this.formatStats(m.stats))}</span>` : ''}</div></div></div>`;
    }).join('');
    box.querySelectorAll('[data-panel-copy]').forEach(b => b.addEventListener('click', () => {
      const m = msgs[Number(b.dataset.panelCopy)];
      navigator.clipboard.writeText(m ? m.text : '').then(() => this.showToast('Reply copied.')).catch(() => this.showToast('Could not copy.', 'error'));
    }));
    box.querySelectorAll('[data-panel-regen]').forEach(b => b.addEventListener('click', () => this.panelRegenerate()));
    box.scrollTop = box.scrollHeight;
  }

  panelStreamBubble() {
    const box = document.getElementById('panel-messages');
    const empty = box.querySelector('[data-panel-chip]');
    if (empty) box.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'flex gap-2';
    wrap.innerHTML = `<div class="w-6 h-6 rounded-md bg-brand-600/30 border border-brand-500/40 text-brand-400 flex items-center justify-center shrink-0 text-[9px] font-mono font-bold">AI</div><div class="flex-1 min-w-0 space-y-1"><div class="ptxt p-2.5 rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800 text-slate-200 leading-relaxed whitespace-pre-wrap break-words"><span class="inline-flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce"></span><span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce [animation-delay:0.2s]"></span><span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce [animation-delay:0.4s]"></span></span></div><div class="plive px-1 text-[10px] font-mono text-slate-500">waiting…</div></div>`;
    box.appendChild(wrap);
    box.scrollTop = box.scrollHeight;
    const txt = wrap.querySelector('.ptxt'), live = wrap.querySelector('.plive');
    return {
      setText: t => { if (t) txt.textContent = t; box.scrollTop = box.scrollHeight; },
      update: (t, ms) => { if (t) txt.textContent = t; live.textContent = (ms / 1000).toFixed(1) + ' s'; box.scrollTop = box.scrollHeight; }
    };
  }

  setPanelBusy(b) {
    const btn = document.getElementById('panel-send-btn');
    if (!btn) return;
    btn.innerHTML = b ? '<i class="fa-solid fa-stop"></i>' : '<i class="fa-solid fa-paper-plane"></i>';
    btn.title = b ? 'Stop' : 'Send';
    btn.classList.toggle('bg-rose-600', b);
    btn.classList.toggle('hover:bg-rose-500', b);
    btn.classList.toggle('bg-brand-600', !b);
    btn.classList.toggle('hover:bg-brand-500', !b);
  }

  panelSend(text) {
    const P = this.panel;
    if (P.busy) { if (P.controller && P.controller.interrupt) P.controller.interrupt(); return; }
    const input = document.getElementById('panel-input');
    const prompt = String(text !== undefined ? text : input.value).trim();
    if (!prompt) return;
    input.value = '';
    input.style.height = 'auto';
    P.messages.push({ role: 'user', text: prompt });
    this.persistPanel();
    this.panelRender();
    this.panelGenerate();
  }

  panelRegenerate() {
    const P = this.panel;
    if (P.busy) return;
    if (P.messages.length && P.messages[P.messages.length - 1].role === 'ai') P.messages.pop();
    this.persistPanel();
    this.panelRender();
    this.panelGenerate();
  }

  async panelGenerate() {
    const P = this.panel, msgs = P.messages;
    if (P.busy) return;
    const lastUser = msgs[msgs.length - 1];
    if (!lastUser || lastUser.role !== 'user') return;
    const prompt = lastUser.text;
    const refId = document.getElementById('panel-model-select').value;
    const ref = this.getLocalReferenceDefinition(refId);
    P.busy = true;
    this.setPanelBusy(true);
    const finish = aiMsg => {
      P.busy = false;
      P.live = null;
      P.controller = null;
      this.setPanelBusy(false);
      if (aiMsg) msgs.push(aiMsg);
      this.persistPanel();
      this.panelRender();
      this.updatePanelStatus();
      const sp = document.getElementById('panel-speed');
      if (sp && aiMsg && aiMsg.stats) sp.textContent = 'Last reply: ' + this.formatStats(aiMsg.stats);
      if (aiMsg && aiMsg.stats) this.updateSpeedBadge(aiMsg.stats);
    };
    try {
      const quick = this.quickAssistantReply(prompt, ref.name);
      if (quick) { finish({ role: 'ai', text: quick, stats: { source: 'instant', ms: 0, tokens: 0, tps: 0 } }); return; }
      const acted = await this.handleAssistantCommand(prompt);
      if (acted) { finish({ role: 'ai', text: acted, stats: { source: 'action', ms: 0, tokens: 0, tps: 0 } }); return; }
      let hits = { context: '', sources: [], noDocsAsked: false };
      if (document.getElementById('panel-rag-toggle').checked) {
        hits = await this.searchDocs(prompt);
        if (hits.noDocsAsked) {
          finish({ role: 'ai', text: "I don't see any imported files yet. Add some in Corpus Library > Corpus Vault, then ask again.", stats: { source: 'instant', ms: 0, tokens: 0, tps: 0 } });
          return;
        }
      }
      const history = msgs.slice(0, -1).slice(-4).map(m => ({ role: m.role === 'ai' ? 'assistant' : 'user', content: m.text }));
      const live = this.panelStreamBubble();
      P.live = live;
      if (!this.localReferencePipelines[ref.key]) {
        live.setText('Loading ' + ref.name + '… (first use downloads it once)');
        this.watchPanelDownload(ref.key);
      }
      const controller = {}, stats = {};
      P.controller = controller;
      const answer = await this.requestLocalReference(refId, prompt, history, {
        stats, controller, docContext: hits.context,
        onToken: (acc, ms) => live.update(this.streamDisplay(acc), ms)
      });
      finish({ role: 'ai', text: answer, stats, sources: hits.sources });
    } catch (err) {
      finish({ role: 'ai', text: 'Assistant unavailable: ' + err.message });
    }
  }

  async handleAssistantCommand(prompt) {
    const p = String(prompt || '').toLowerCase();
    const isQuestion = /\b(how|what|why|when|which|where|should|explain|tell me about the|what does)\b/.test(p);
    // Screen text follow-ups (after a screen/screenshot read)
    if (/(turn|convert|make|use)\s+(this|it|that|the screen|the text)\b.*\b(training|pairs|qa|q&a|dataset|data)\b/.test(p)) return this.materialToTraining(this.lastScreenText, 'Screen text');
    if (/(add|store|save|keep)\s+(this|it|that|the screen|the text)\b.*\b(corpus|training|dataset)\b/.test(p)) return this.appendRawToCorpus(this.lastScreenText, 'Screen text');
    // Create a model
    const createMatch = p.match(/\b(create|make|build|new|spawn)\b[^.?!]*\b(model|ai|assistant|bot|brain|character)\b/);
    let created = null;
    if (createMatch && !isQuestion) {
      let name = '';
      const nm = prompt.match(/(?:called|named|name\s+it|call\s+it)\s+["']?([A-Za-z0-9][\w\s'-]{0,40}?)(?=["']?\s*(?:,|and\b|then\b|with\b|please|now\b|\.|$))/i);
      if (nm) name = nm[1].trim();
      let type = 'neural';
      if (/\b(ngram|n-gram|tf-?idf|semantic|retrieval)\b/.test(p)) type = 'ngram';
      else if (/\battention\b/.test(p)) type = 'attention';
      created = this.createNewModel(name || 'Assistant Model', type);
      this.switchTab('models-tab');
    }
    // Train the model
    const trainMatch = p.match(/\b(train|teach|fit)\b/);
    if (trainMatch && !isQuestion) {
      if (this.isTraining) return 'Training is already running; let it finish first.';
      const corpusEl = document.getElementById('training-text');
      const hasCorpus = Boolean(corpusEl && corpusEl.value.trim());
      if (created && !hasCorpus) {
        this.switchTab('train-tab');
        return `Created "${created.name}" (${this.constructor.ENGINE_LABELS[created.type] || created.type}). Train Studio is open, but the corpus is empty: paste facts, load a pack from Corpus Library, or say "read my screen" to capture text, then say "train it".`;
      }
      if (!hasCorpus) return 'The training corpus is empty. Add facts in Train Studio or load a pack from Corpus Library first, then say "train it".';
      this.switchTab('train-tab');
      const tName = created ? created.name : (this.getActiveModelObj()?.name || 'the active model');
      this.startTraining().catch(err => this.showToast('Training failed: ' + (err.message || 'unexpected error'), 'error'));
      return (created ? `Created "${created.name}" and ` : '') + `started training "${tName}" on the current corpus. Watch the loss curve in Train Studio; the app saves automatically when it finishes.`;
    }
    if (created) return `Created "${created.name}" (${this.constructor.ENGINE_LABELS[created.type] || created.type}) and made it the active model. Say "train it" to train it on the current corpus, or paste new data in Train Studio.`;
    // Evaluate
    if (/\b(evaluate|run (a |the )?(tests?|eval|benchmark)|test (my|the) model)\b/.test(p) && !isQuestion) {
      this.switchTab('eval-tab');
      setTimeout(() => document.getElementById('run-eval-btn')?.click(), 60);
      return 'Opened Evaluation and started a test run on the active model.';
    }
    // Corpus pre-flight / clean
    if (/\b(clean|dedupe|deduplicate|remove duplicates|fix)\b.*\b(corpus|dataset|data|pairs)\b/.test(p) && !isQuestion) {
      this.switchTab('train-tab');
      return this.autoFixCorpus();
    }
    if (/\b(preflight|pre-flight|check|scan|audit|review)\b.*\b(corpus|dataset|data|pairs|training)\b/.test(p) && !isQuestion) {
      this.switchTab('train-tab');
      return this.runCorpusPreflight();
    }
    // Export model
    if (/\bexport\b.*\b(model|brain)\b|\bsave\b.*\bbrain\b/.test(p) && !isQuestion) {
      this.exportActiveModelJSON();
      return 'Exported the active model as a .json brain file via your browser download.';
    }
    // Switch tabs
    const tabMap = [['chat', 'chat-tab', 'Chat Studio'], ['training', 'train-tab', 'Train Studio'], ['train', 'train-tab', 'Train Studio'], ['library', 'library-tab', 'Corpus Library'], ['corpus', 'library-tab', 'Corpus Library'], ['model', 'models-tab', 'Models Hub'], ['eval', 'eval-tab', 'Evaluation'], ['memory', 'memory-tab', 'Memory'], ['guide', 'guide-tab', 'How To Guide']];
    if (!isQuestion && /\b(open|go to|switch to|show|jump to|take me to)\b/.test(p)) {
      for (const [word, tab, label] of tabMap) {
        if (new RegExp('\\b' + word + '\\b').test(p)) { this.switchTab(tab); return 'Opened ' + label + '.'; }
      }
    }
    return '';
  }
}
