// Chat Studio: sending messages, chat history, streaming bubbles, the Teach/Correct modal.
// Methods here are copied onto AIApp in main.js, so `this` is the app.

export class AppChat {
  handleSendMessage() {
    const inputEl = document.getElementById('chat-input');
    const prompt = inputEl.value.trim();
    if (!prompt) return;
    if (!this.referenceMode && !this.getActiveModelObj()) { this.showToast('Create or select a model first.', 'error'); return; }

    inputEl.value = '';
    this.appendChatMessage('user', prompt);

    const selectedReference = document.getElementById('active-model-select').value;
    if (selectedReference.startsWith('__ref_')) {
      this.referenceMode = true;
      this.activeReferenceId = selectedReference;
      this.askAssistant(selectedReference, prompt);
      return;
    }
    const activeObj = this.getActiveModelObj();
    const model = activeObj.engine;

    const temp = parseFloat(document.getElementById('chat-temp').value);
    const topK = parseInt(document.getElementById('chat-topk').value);
    const repPenalty = parseFloat(document.getElementById('chat-rep-penalty').value);
    const maxTokens = parseInt(document.getElementById('chat-max-tokens').value);

    const typingBubble = this.appendTypingIndicator();

    setTimeout(() => {
      const cached = this.previewCache && this.previewCache.prompt === prompt ? this.previewCache : null;
      const hitsMemory = Boolean(model.memory && model.memory.queryBestMatch(prompt));
      const tGen = performance.now();
      const response = cached ? cached.response : this.generatePreferred(model, prompt, maxTokens, temp, repPenalty, topK);
      const genMs = cached ? cached.ms : performance.now() - tGen;
      this.previewCache = null;
      typingBubble.remove();
      const isUnknown = response.includes('[[[UNKNOWN_RESPONSE]]]');
      const shownText = response.replace('[[[UNKNOWN_RESPONSE]]]', '');
      let tokenCount = 0;
      try { tokenCount = model.tokenizer ? model.tokenizer.tokenize(shownText).length : shownText.split(/\s+/).filter(Boolean).length; } catch (e) { tokenCount = shownText.split(/\s+/).filter(Boolean).length; }
      const stats = { tokens: tokenCount, ms: genMs, ttftMs: null, tps: tokenCount / Math.max(genMs, 0.05) * 1000, source: hitsMemory ? 'retrieval' : 'generated' };
      this.appendChatMessage('ai', shownText, prompt, isUnknown, false, stats);
      this.updateSpeedBadge(stats);

      if (!isUnknown && document.getElementById('live-learn-toggle').checked && prompt.split(/\s+/).length >= 3) {
        const learnedTurn = 'user: ' + prompt + '\nai: ' + response;
        const updatedCorpus = this.mergeCorpus(model.rawCorpus, learnedTurn, false);
        model.rawCorpus = updatedCorpus;
        if (model.memory) model.memory.indexText(updatedCorpus);
        this.saveModelsToStorage();
      }

      this.updateStatsDisplay();
      this.refreshMemoryInspector();
    }, 160);
  }

  formatStats(st) {
    if (!st) return '';
    if (st.source === 'instant') return 'instant reply';
    const fmt = ms => ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(2) + ' s';
    const tps = st.tps >= 1000 ? '1000+' : (st.tps || 0).toFixed(1);
    const parts = [`${tps} t/s`, `${st.tokens} tok`, fmt(st.ms)];
    if (st.ttftMs !== null && st.ttftMs !== undefined) parts.push('first token ' + fmt(st.ttftMs));
    if (st.source === 'retrieval') parts.push('retrieved from memory');
    return parts.join(' · ');
  }

  updateSpeedBadge(st) {
    const el = document.getElementById('side-speed');
    if (!el) return;
    if (!st || st.source === 'instant') { el.textContent = st ? 'instant' : '—'; return; }
    el.textContent = (st.tps >= 1000 ? '1000+' : (st.tps || 0).toFixed(1)) + ' t/s · ' + (st.ms < 1000 ? Math.round(st.ms) + ' ms' : (st.ms / 1000).toFixed(1) + ' s');
  }

  streamDisplay(acc) {
    const clean = String(acc || '').replace(/\*\*|__|`/g, '').replace(/\s+/g, ' ').trimStart();
    const m = clean.match(/^[^.!?]+[.!?]+\s+[^.!?]+[.!?]+(?=\s|$)/);
    return m ? m[0] : clean;
  }

  appendStreamingBubble(onStop) {
    const container = document.getElementById('chat-messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'flex gap-3 max-w-2xl';
    wrapper.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-brand-600/30 border border-brand-500/40 text-brand-400 flex items-center justify-center shrink-0 text-xs font-mono font-bold">AI</div>
      <div class="flex flex-col gap-1.5 flex-1 min-w-0">
        <div class="stream-text p-3.5 rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800 text-slate-200 leading-relaxed shadow-sm whitespace-pre-wrap break-words min-w-0"><span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-brand-400 animate-bounce"></span><span class="w-2 h-2 rounded-full bg-brand-400 animate-bounce [animation-delay:0.2s]"></span><span class="w-2 h-2 rounded-full bg-brand-400 animate-bounce [animation-delay:0.4s]"></span></span></div>
        <div class="flex items-center gap-3 px-1 text-[11px] text-slate-400">
          <span class="stream-live font-mono text-[10px] text-slate-500">waiting…</span>
          <button class="stream-stop hover:text-rose-300 transition flex items-center gap-1"><i class="fa-solid fa-stop"></i> Stop</button>
        </div>
      </div>`;
    const textEl = wrapper.querySelector('.stream-text'), liveEl = wrapper.querySelector('.stream-live'), stopBtn = wrapper.querySelector('.stream-stop');
    stopBtn.addEventListener('click', () => { stopBtn.disabled = true; stopBtn.textContent = 'Stopping…'; if (onStop) onStop(); });
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return {
      wrapper,
      update: (text, ms) => {
        if (text) textEl.textContent = text;
        liveEl.textContent = (ms / 1000).toFixed(1) + ' s';
        container.scrollTop = container.scrollHeight;
      }
    };
  }

  appendChatMessage(role, text, originalPrompt = '', isUnknown = false, noTeach = false, stats = null) {
    const container = document.getElementById('chat-messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'flex gap-3 max-w-2xl ' + (role === 'user' ? 'ml-auto flex-row-reverse' : '');

    if (role === 'user') {
      wrapper.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-accent-600/30 border border-accent-500/40 text-accent-300 flex items-center justify-center shrink-0 text-xs font-mono font-bold">
          YOU
        </div>
        <div class="p-3 rounded-2xl rounded-tr-sm bg-accent-600 text-white leading-relaxed shadow-sm break-words min-w-0">
          ${this.escapeHTML(text)}
        </div>
      `;
    } else {
      wrapper.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-brand-600/30 border border-brand-500/40 text-brand-400 flex items-center justify-center shrink-0 text-xs font-mono font-bold">
          AI
        </div>
        <div class="flex flex-col gap-1.5 flex-1">
          <div class="p-3.5 rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800 text-slate-200 leading-relaxed shadow-sm whitespace-pre-wrap break-words min-w-0">${this.escapeHTML(String(text).trim())}</div>
          <div class="flex items-center gap-3 px-1 text-[11px] text-slate-400">
            ${noTeach ? '' : `<button class="teach-btn hover:text-amber-300 transition flex items-center gap-1" data-prompt="${this.escapeHTML(originalPrompt)}" data-response="${this.escapeHTML(text)}"><i class="fa-solid fa-graduation-cap text-amber-400"></i> ${isUnknown ? 'Teach missing vocabulary / topic' : 'That answer is wrong · Correct it'}</button>`}
            <button class="copy-btn hover:text-cyan-300 transition flex items-center gap-1"><i class="fa-regular fa-copy"></i> Copy</button>
            ${noTeach ? `<button class="regen-btn hover:text-emerald-300 transition flex items-center gap-1"><i class="fa-solid fa-rotate"></i> Regenerate</button>` : ''}
            ${stats ? `<span class="font-mono text-[10px] text-slate-500" title="Speed of this reply"><i class="fa-solid fa-bolt text-amber-400"></i> ${this.escapeHTML(this.formatStats(stats))}</span>` : ''}
          </div>
        </div>
      `;

      const copyBtn = wrapper.querySelector('.copy-btn');
      if (copyBtn) copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(text).then(() => this.showToast('Reply copied.')).catch(() => this.showToast('Could not copy.', 'error'));
      });
      const regenBtn = wrapper.querySelector('.regen-btn');
      if (regenBtn) regenBtn.addEventListener('click', () => {
        if (!this.referenceMode || !this.activeReferenceId) { this.showToast('Select an assistant model to regenerate.', 'error'); return; }
        const s2 = this.getActiveChatSession();
        if (s2) {
          for (let k = s2.messages.length - 1; k >= 0; k--) if (s2.messages[k].role === 'ai' && s2.messages[k].text === text) { s2.messages.splice(k, 1); break; }
          this.persistChatSessions();
        }
        if (this.refHistory.length >= 2 && this.refHistory[this.refHistory.length - 1].content === text) this.refHistory.splice(-2);
        wrapper.remove();
        this.askAssistant(this.activeReferenceId, originalPrompt, true);
      });
      const teachBtn = wrapper.querySelector('.teach-btn');
      if (teachBtn) teachBtn.addEventListener('click', (e) => {
        const btn = e.currentTarget;
        this.openTeachModal(btn.getAttribute('data-prompt'), btn.getAttribute('data-response'), isUnknown);
      });
    }

    container.appendChild(wrapper);
    const session = this.getActiveChatSession();
    if (session && !this.isRenderingChatHistory) {
      session.messages.push({ role, text, prompt: originalPrompt, unknown: isUnknown, noTeach, stats });
      if (role === 'user' && session.title === 'New conversation') session.title = text.slice(0, 32);
      this.renderChatSessionOptions();
      this.persistChatSessions();
    }
    container.scrollTop = container.scrollHeight;
  }

  appendTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'flex gap-3 max-w-2xl';
    wrapper.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-brand-600/30 border border-brand-500/40 text-brand-400 flex items-center justify-center shrink-0 text-xs font-mono font-bold">
        AI
      </div>
      <div class="p-3.5 rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800 text-slate-400 flex items-center gap-1">
        <span class="w-2 h-2 rounded-full bg-brand-400 animate-bounce"></span>
        <span class="w-2 h-2 rounded-full bg-brand-400 animate-bounce [animation-delay:0.2s]"></span>
        <span class="w-2 h-2 rounded-full bg-brand-400 animate-bounce [animation-delay:0.4s]"></span>
      </div>
    `;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return wrapper;
  }

  getActiveChatSession() { return this.chatSessions.find(s => s.id === this.activeChatId) || null; }

  initChatSessions() {
    if (!this.chatSessions.length) { this.newChatSession(false); return; }
    this.activeChatId = this.chatSessions[0].id;
    this.renderChatSessionOptions();
    this.renderChatSession(this.getActiveChatSession());
  }

  renderChatSessionOptions() {
    const select = document.getElementById('chat-session-select');
    if (!select) return;
    select.innerHTML = this.chatSessions.map(s => `<option value="${this.escapeHTML(s.id)}">${this.escapeHTML(s.title || 'Conversation')}</option>`).join('');
    select.value = this.activeChatId || '';
  }

  newChatSession(showToast = true) {
    const id = 'chat-' + Date.now() + '-' + Math.floor(Math.random() * 9999);
    this.chatSessions.unshift({ id, title: 'New conversation', createdAt: Date.now(), messages: [] });
    this.activeChatId = id;
    this.refHistory = [];
    document.getElementById('chat-messages').innerHTML = '';
    this.renderChatSessionOptions();
    this.persistChatSessions();
    if (showToast) this.showToast('New conversation started.');
  }

  switchChatSession(id) {
    const s = this.chatSessions.find(x => x.id === id);
    if (!s) return;
    this.activeChatId = id;
    this.refHistory = [];
    this.renderChatSession(s);
  }

  renderChatSession(session) {
    const box = document.getElementById('chat-messages');
    box.innerHTML = '';
    this.isRenderingChatHistory = true;
    (session?.messages || []).forEach(m => this.appendChatMessage(m.role, m.text, m.prompt || '', !!m.unknown, !!m.noTeach, m.stats || null));
    this.isRenderingChatHistory = false;
    if (session) session.messages = (session.messages || []).slice(-500);
    box.scrollTop = box.scrollHeight;
  }

  exportChatTranscript() {
    const msgs = Array.from(document.getElementById('chat-messages').children).map(el => el.innerText.trim()).join('\n\n---\n\n');
    const blob = new Blob([msgs], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chat-transcript-' + Date.now() + '.txt';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  openTeachModal(prompt, currentResponse, isUnknown = false) {
    document.getElementById('teach-prompt-display').innerText = prompt || "(No user prompt)";
    const currentEl = document.getElementById('teach-current-display');
    if (currentEl) currentEl.innerText = currentResponse || '(No previous answer)';
    document.getElementById('teach-ideal-input').value = '';
    let prefillTerm = '';
    if (isUnknown) {
      const m = this.getActiveModelObj();
      const a = this.analyzeKnowledgeCoverage(m.engine, prompt || '');
      prefillTerm = a.unknownTerms?.[0] || '';
    }
    document.getElementById('teach-vocab-term').value = prefillTerm;
    document.getElementById('teach-vocab-definition').value = '';
    document.getElementById('teach-modal').classList.remove('hidden');
  }

  closeTeachModal() {
    document.getElementById('teach-modal').classList.add('hidden');
  }

  async submitCorrection() {
    const prompt = document.getElementById('teach-prompt-display').innerText;
    const idealResponse = document.getElementById('teach-ideal-input').value.trim();
    const term = document.getElementById('teach-vocab-term').value.trim(), definition = document.getElementById('teach-vocab-definition').value.trim();
    if (!idealResponse && !(term && definition)) {
      this.showToast('Enter a corrected answer or provide both a vocabulary term and definition.', 'error');
      return;
    }
    const activeObj = this.getActiveModelObj();
    let updatedCorpus = activeObj.engine.rawCorpus || document.getElementById('training-text').value || '';
    if (idealResponse) {
      const newEntry = 'user: ' + prompt + '\nai: ' + idealResponse;
      updatedCorpus = this.mergeCorpus(updatedCorpus, newEntry, true);
    }
    if (term && definition) {
      const facts = this.activeFacts(), key = term.toLowerCase(), now = new Date().toISOString();
      const existing = facts.find(f => f.title.toLowerCase() === ('vocabulary: ' + key));
      if (existing) {
        existing.versions = existing.versions || [];
        existing.versions.push({ title: existing.title, body: existing.body, at: now });
        existing.body = `${term}: ${definition}`;
        existing.updatedAt = now;
      } else {
        facts.push({ id: 'vocab-' + Date.now() + '-' + Math.floor(Math.random() * 9999), title: 'Vocabulary: ' + term, body: `${term}: ${definition}`, createdAt: now, updatedAt: now, versions: [] });
      }
      updatedCorpus = this.mergeCorpus(updatedCorpus, `Knowledge: Vocabulary ${term}. ${definition}`, false);
    }
    document.getElementById('training-text').value = updatedCorpus;
    this.updateCorpusStats();
    this.closeTeachModal();
    this.showToast('Your correction and vocabulary were saved. Training the active model…');
    try {
      await this.quickTrainModel(activeObj, updatedCorpus, 40);
      this.renderKnowledgeFacts();
      this.refreshMemoryInspector();
      this.showToast('Correction learned and model saved.');
    } catch (err) {
      console.error('Correction training failed:', err);
      activeObj.engine.rawCorpus = updatedCorpus;
      if (activeObj.engine.memory) activeObj.engine.memory.indexText(updatedCorpus);
      this.saveModelsToStorage();
      this.showToast('Saved the correction, but training failed. Try Train Studio.', 'error');
    }
  }

  calculatePrompt() {
    const prompt = document.getElementById('chat-input').value.trim();
    if (!prompt) return;
    const m = this.getActiveModelObj(), e = m.engine, tokens = e.tokenizer?.tokenize(prompt) || [], ids = e.tokenizer?.encode(tokens) || [], candidates = e.memory?.queryCandidates(prompt, 5) || [], hit = e.memory?.queryBestMatch(prompt) || null;
    const tCalc = performance.now();
    const response = this.generatePreferred(e, prompt, parseInt(document.getElementById('chat-max-tokens').value), parseFloat(document.getElementById('chat-temp').value), parseFloat(document.getElementById('chat-rep-penalty').value), parseInt(document.getElementById('chat-topk').value));
    this.previewCache = { prompt, response, ms: performance.now() - tCalc };
    const vocab = e.tokenizer?.vocab;
    const known = tokens.filter(t => vocab?.has(t)).length;
    const trace = document.getElementById('prompt-trace');
    trace.classList.remove('hidden');
    trace.innerHTML = `<div class="font-semibold text-cyan-300">Prompt computation · ${this.escapeHTML(m.name)}</div><div><b>1. Tokenize:</b> ${tokens.length} tokens · ${known} known vocabulary tokens · ${tokens.length - known} unknown (fallback/subword handling applies).</div><div class="break-words"><b>2. Token IDs:</b> ${this.escapeHTML(ids.join(', ') || '—')}</div><div><b>3. Retrieval candidates:</b> ${candidates.length ? candidates.map((c, i) => `<div class="ml-2 mt-1">${i + 1}. [${c.source}] ${Math.round(c.score * 100)}% · ${this.escapeHTML(c.prompt || c.text.slice(0, 90))}</div>`).join('') : 'No candidate passed retrieval ranking.'}</div><div><b>4. Selection:</b> ${hit ? `Retrieved ${hit.source} answer (similarity ${hit.score.toFixed(3)}).` : 'No retrieval match; next-token model sampling used.'}</div><div class="border-t border-slate-800 pt-2"><b>5. Predicted output:</b><p class="mt-1 text-emerald-300 whitespace-pre-wrap">${this.escapeHTML(response || '(empty)')}</p></div>`;
  }

  async askAssistant(refId, prompt) {
    const ref = this.getLocalReferenceDefinition(refId);
    const quick = this.quickAssistantReply(prompt, ref.name);
    if (quick) {
      const st = { source: 'instant', ms: 0, tokens: 0, tps: 0 };
      this.appendChatMessage('ai', quick, prompt, false, true, st);
      this.updateSpeedBadge(st);
      return;
    }
    if (!this.localReferencePipelines[ref.key]) this.showToast('First use: downloading ' + ref.name + ' (cached afterwards)…');
    const controller = {};
    const stream = this.appendStreamingBubble(() => { if (controller.interrupt) controller.interrupt(); });
    const stats = {};
    try {
      const answer = await this.requestLocalReference(refId, prompt, this.refHistory.slice(-4), {
        stats, controller,
        onToken: (acc, ms) => stream.update(this.streamDisplay(acc), ms)
      });
      stream.wrapper.remove();
      this.refHistory.push({ role: 'user', content: prompt }, { role: 'assistant', content: answer });
      this.refHistory = this.refHistory.slice(-6);
      this.appendChatMessage('ai', answer, prompt, false, true, stats);
      this.updateSpeedBadge(stats);
    } catch (err) {
      stream.wrapper.remove();
      this.appendChatMessage('ai', 'Local assistant unavailable: ' + err.message + '\n\nOpen Models Hub and click Download & prepare on this assistant model.', prompt, false, true);
    }
  }

  quickAssistantReply(prompt, name) {
    const p = String(prompt || '').toLowerCase().trim().replace(/[!.?]+$/g, '');
    if (/^(hi|hello|hey|yo|hiya|good (morning|afternoon|evening))$/.test(p)) return "Hi! I'm your Teachable AI Studio assistant. I can explain any tab, create or train a model for you (try: create a model called Tutor and train it), check your dataset, or read text from your screen with the Read my screen button below.";
    if (/^(thanks|thank you|ty|thx)$/.test(p)) return "You're welcome! Ask me anything else about the studio.";
    if (/^(bye|goodbye|see you|see you later)$/.test(p)) return 'Goodbye! Come back anytime.';
    if (/^(who are you|what are you|what is your name|what's your name|whats your name)$/.test(p)) return `I'm ${name}, a small local assistant that runs inside Teachable AI Studio. I can explain the app, run actions like creating or training a model, and OCR text from your screen.`;
    if (/^what can you do|^(help|capabilities|features|commands)/.test(p)) return "Four things: answer studio questions, run app actions from plain requests (create a model, train it, evaluate, clean the corpus, switch tabs), check your training data with a pre-flight scan, and read text from your screen or a screenshot with OCR.";
    if (/(how (do|to|can))\s*(i|you)?\s*(train|teach)/.test(p)) return "Open Train Studio, paste user:/ai: pairs or knowledge paragraphs (or load a pack from Corpus Library), then press Merge & Train. Or just tell me: create a model called Tutor and train it.";
    if (/(what is|what's|whats)\s*(an?|the)?\s*epoch/.test(p)) return "An epoch is one full pass over your training examples. More epochs remember better but overfit and take longer; the app early-stops when validation loss stops improving.";
    if (/(learning rate|what is lr|what does lr)/.test(p)) return "Learning rate is the step size of weight updates: too high and answers get unstable, too low and training is slow. The built-in profiles cap it safely for your device.";
    if (/(what does|what is)\s*temperature/.test(p)) return "Temperature sets creativity: low values give steady, predictable replies and high values give more varied ones.";
    if (/(what is|how does).*live memory/.test(p)) return "Live Memory adds new chat turns to the model's retrieval memory instantly, without changing neural weights. Retrain only when you want the neural side to learn permanently.";
    if (/(import|upload).*(file|data|corpus)|(corpus vault)/.test(p)) return "Corpus Library > Corpus Vault imports .txt, .md, .csv, .json, .jsonl or .tsv files into browser storage. Then select them and press Append selected datasets to Train Studio.";
    if (/(read my screen|screen (read|ocr)|what does read my screen)/.test(p)) return "Press Read my screen, choose a window or tab, and I extract the visible text with OCR. Then say turn this into training data or add this to my corpus. Screen capture needs https or localhost; otherwise use the image button with a screenshot.";
    if (/(tips|best practices).*(train|corpus|data)|how (do i|to) (get )?better (results|answers)/.test(p)) return "Use short questions with one clear answer each, run the Pre-flight check to catch duplicates and conflicts, and prefer more varied pairs over repeating the same fact. Then retrain and compare in Evaluation.";
    return '';
  }
}
