// Q&A generator and paraphrase augmenter used by Train Studio.
// Methods here are copied onto AIApp in main.js, so `this` is the app.

export class AppQATools {
  /* ============ Q&A generator ============ */
  initQAGenerator() {
    const $ = id => document.getElementById(id);
    if (!$('qa-gen-btn')) return;
    $('qa-gen-from-corpus').addEventListener('click', () => {
      const paras = this.parseParagraphs($('training-text').value);
      if (!paras.length) { this.showToast('No plain paragraphs found in the corpus.', 'error'); return; }
      $('qa-gen-source').value = paras.join('\n\n');
      this.showToast(paras.length + ' paragraph(s) loaded.');
    });
    $('qa-gen-btn').addEventListener('click', () => this.runQAGeneration());
    $('qa-para-btn').addEventListener('click', () => this.runParaphrase());
    $('qa-gen-clear').addEventListener('click', () => { $('qa-gen-output').value = ''; $('qa-gen-status').textContent = ''; });
    $('qa-gen-append').addEventListener('click', () => {
      const out = $('qa-gen-output').value.trim();
      if (!out) { this.showToast('Generate some pairs first.', 'error'); return; }
      const area = $('training-text');
      area.value = this.mergeCorpus(area.value, out, false);
      this.updateCorpusStats();
      this.showToast('Appended to the training corpus. Press Merge & Train to learn it.');
    });
  }

  parseParagraphs(text) {
    return String(text || '').split(/\n\s*\n/).map(x => x.trim()).filter(x => x && !/^(user|human|q|ai|bot|assistant|a)\s*:/im.test(x) && !/^#/.test(x) && x.split(/\s+/).length >= 8);
  }

  parseDialoguePairs(text) {
    const pairs = [];
    let q = null;
    for (const line of String(text || '').split(/\r?\n/)) {
      const u = line.match(/^\s*(?:user|human|q)\s*:\s*(.+)$/i), a = line.match(/^\s*(?:ai|bot|assistant|a)\s*:\s*(.+)$/i);
      if (u) q = u[1].trim();
      else if (a && q) { pairs.push({ q, a: a[1].trim() }); q = null; }
    }
    return pairs;
  }

  formatPairs(pairs) { return pairs.map(p => `user: ${p.q}\nai: ${p.a}`).join('\n\n'); }

  qaFromParagraph(paragraph, per) {
    const clean = paragraph.replace(/\s+/g, ' ').trim();
    const sentences = (clean.match(/[^.!?]+[.!?]+(?=\s|$)/g) || [clean]).map(x => x.trim()).filter(x => x.split(' ').length >= 4);
    if (!sentences.length) return [];
    const trunc = t => {
      if (t.length <= 320) return t;
      const cut = t.slice(0, 320), e = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '));
      return e > 80 ? cut.slice(0, e + 1) : cut.replace(/\s+\S*$/, '') + '…';
    };
    const defs = [];
    let topic = '';
    for (const sn of sentences) {
      const m = sn.match(/^(?:the |a |an )?([A-Za-z0-9][^,;:()]{1,60}?)\s+(is|are|was|were)\s+(?:a |an |the )?(.{8,})$/i);
      if (m) {
        const subj = m[1].trim();
        if (!topic) topic = subj;
        if (subj.split(/\s+/).length <= 6) defs.push({ q: `what ${m[2].toLowerCase()} ${subj.toLowerCase()}`, a: trunc(sn) });
      }
    }
    if (!topic) topic = sentences[0].split(/\s+/).slice(0, 4).join(' ').replace(/[,;:.]+$/, '');
    const t = topic.toLowerCase();
    const extra = [
      { q: `tell me about ${t}`, a: trunc(sentences.slice(0, 2).join(' ')) },
      { q: `explain ${t}`, a: trunc(sentences.slice(0, 3).join(' ')) },
      { q: `summarize ${t}`, a: trunc(sentences[0]) }
    ];
    const seen = new Set(), out = [];
    for (const pr of [...defs, ...extra]) {
      if (seen.has(pr.q)) continue;
      seen.add(pr.q);
      out.push(pr);
    }
    return out.slice(0, per);
  }

  paraphraseQuestion(q, n) {
    const base = String(q || '').trim().replace(/[?!.]+$/, '');
    if (!base) return [];
    let options = [], m;
    if ((m = base.match(/^(?:what is|what are|what's|whats)\s+(.+)$/i))) {
      const x = m[1];
      options = [`can you explain ${x}`, `define ${x}`, `tell me what ${x} is`, `explain ${x}`, `what does ${x} mean`, `i want to know about ${x}`];
    } else if ((m = base.match(/^(?:who is|who are)\s+(.+)$/i))) {
      options = [`tell me about ${m[1]}`, `can you tell me who ${m[1]} is`, `give me info on ${m[1]}`];
    } else if ((m = base.match(/^(?:how do i|how do you|how can i|how to)\s+(.+)$/i))) {
      options = [`how to ${m[1]}`, `can you show me how to ${m[1]}`, `what is the way to ${m[1]}`, `help me ${m[1]}`];
    } else if ((m = base.match(/^(tell me about|explain|summarize|describe)\s+(.+)$/i))) {
      const v = m[1].toLowerCase(), x = m[2];
      options = [`what is ${x}`, `can you ${v} ${x}`, `please ${v} ${x}`, `i would like you to ${v} ${x}`];
    }
    options.push(`can you tell me ${base}`, `please ${base}`, `i want to know: ${base}`, `quick question: ${base}`);
    const seen = new Set([base.toLowerCase()]), out = [];
    for (const o of options) {
      const k = o.toLowerCase().trim();
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(o.trim());
      if (out.length >= n) break;
    }
    return out;
  }

  async runAssistantPlain(refId, system, user, maxTokens) {
    const ref = this.getLocalReferenceDefinition(refId);
    const pipe = this.localReferencePipelines[ref.key] || await this.prepareLocalReference(ref.key);
    const opts = { max_new_tokens: maxTokens, do_sample: false, repetition_penalty: 1.1 };
    let out = '';
    try {
      const r = await pipe([{ role: 'system', content: system }, { role: 'user', content: user }], opts);
      const g = r?.[0]?.generated_text;
      if (Array.isArray(g)) { const last = g[g.length - 1]; out = typeof last === 'string' ? last : (last?.content || ''); }
      else if (typeof g === 'string') out = g;
    } catch (e) {
      const plain = `${system}\n\n${user}\nAssistant:`;
      const r = await pipe(plain, opts);
      const g = r?.[0]?.generated_text || '';
      out = typeof g === 'string' ? g.slice(plain.length) : '';
    }
    return String(out || '').trim();
  }

  async generateQAWithAssistant(refId, paragraph, n) {
    const text = await this.runAssistantPlain(refId,
      `You write study flashcards. Write exactly ${n} short question and answer pairs about the text. Use this exact format and nothing else:\nuser: <question>\nai: <one-sentence answer>`,
      'Text: ' + paragraph.slice(0, 900), 220);
    const pairs = [];
    let q = null;
    for (const line of text.split(/\r?\n/)) {
      const u = line.match(/^\s*(?:user|q|question)\s*:\s*(.+)$/i), a = line.match(/^\s*(?:ai|a|answer)\s*:\s*(.+)$/i);
      if (u) q = u[1].trim().replace(/\?+$/, '');
      else if (a && q) {
        const ans = a[1].trim();
        if (q.split(/\s+/).length >= 2 && ans.split(/\s+/).length >= 3) pairs.push({ q: q.toLowerCase(), a: ans });
        q = null;
      }
    }
    return pairs.slice(0, n);
  }

  async runQAGeneration() {
    const $ = id => document.getElementById(id);
    const paras = this.parseParagraphs($('qa-gen-source').value);
    if (!paras.length) { this.showToast('Paste at least one paragraph (8+ words).', 'error'); return; }
    const per = parseInt($('qa-gen-count').value), mode = $('qa-gen-mode').value, variants = parseInt($('qa-gen-variants').value);
    const status = $('qa-gen-status'), btn = $('qa-gen-btn');
    btn.disabled = true;
    let pairs = [], note = '';
    try {
      if (mode === 'assistant') {
        const refId = $('panel-model-select')?.value || '__ref_smollm2_360__';
        const count = Math.min(paras.length, 8);
        for (let i = 0; i < count; i++) {
          status.textContent = `Assistant writing pairs… paragraph ${i + 1}/${count}`;
          try { pairs.push(...await this.generateQAWithAssistant(refId, paras[i], per)); }
          catch (err) { this.showToast('Assistant failed: ' + err.message, 'error'); break; }
        }
        if (paras.length > 8) note = ' (assistant mode handles 8 paragraphs per run)';
        if (!pairs.length) { note = ' Assistant produced no usable pairs, so rules mode was used.'; for (const pg of paras) pairs.push(...this.qaFromParagraph(pg, per)); }
      } else {
        paras.forEach(pg => pairs.push(...this.qaFromParagraph(pg, per)));
      }
      const seen = new Set();
      pairs = pairs.filter(pr => { const k = pr.q.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });
      const expanded = variants > 0 ? pairs.flatMap(pr => [pr, ...this.paraphraseQuestion(pr.q, variants).map(v => ({ q: v, a: pr.a }))]) : pairs;
      $('qa-gen-output').value = this.formatPairs(expanded);
      status.textContent = `${pairs.length} generated + ${expanded.length - pairs.length} paraphrased = ${expanded.length} pairs. Review, then append.${note}`;
    } finally { btn.disabled = false; }
  }

  runParaphrase() {
    const $ = id => document.getElementById(id);
    const pairs = this.parseDialoguePairs($('training-text').value);
    if (!pairs.length) { this.showToast('No user:/ai: pairs found in the training corpus.', 'error'); return; }
    const n = parseInt($('qa-para-count').value);
    const out = [];
    for (const pr of pairs) {
      for (const v of this.paraphraseQuestion(pr.q, n)) { out.push({ q: v, a: pr.a }); if (out.length >= 2000) break; }
      if (out.length >= 2000) break;
    }
    $('qa-gen-output').value = this.formatPairs(out);
    $('qa-gen-status').textContent = `${out.length} reworded pairs from ${pairs.length} corpus pairs. Review, then append.`;
  }
}
