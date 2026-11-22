// Evaluation tab: test bench, model comparison and the speed benchmark.
// Methods here are copied onto AIApp in main.js, so `this` is the app.

export class AppEval {
  scoreBenchmark(response, expected, latencyMs) {
    const words = text => (String(text || '').toLowerCase().match(/[a-z0-9]+/g) || []);
    const out = words(response);
    const gold = words(expected);
    const counts = new Map();
    out.forEach(w => counts.set(w, (counts.get(w) || 0) + 1));
    let overlap = 0;
    gold.forEach(w => { const n = counts.get(w) || 0; if (n > 0) { overlap++; counts.set(w, n - 1); } });
    const precision = out.length ? overlap / out.length : 0;
    const recall = gold.length ? overlap / gold.length : 0;
    const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
    const repeated = out.length ? 1 - new Set(out).size / out.length : 0;
    const instructionFollowing = !!String(response || '').trim();
    const factualRecall = recall;
    const consistency = 1 - repeated;
    const reasoning = gold.length ? f1 : (out.length ? Math.min(1, out.length / 12) : 0);
    return { latencyMs, precision, recall, f1, reasoning, factualRecall, instructionFollowing, consistency, repeated, empty: !String(response || '').trim(), exact: !!gold.length && String(response || '').trim().toLowerCase() === String(expected || '').trim().toLowerCase() };
  }

  runEvaluation() {
    if (this.isRestoring) this.isRestoring = false;
    const active = this.getActiveModelObj();
    const prompts = document.getElementById('eval-prompts').value.split(/\r?\n/).map(x => x.trim()).filter(Boolean).slice(0, 100);
    const gold = document.getElementById('eval-gold').value.split(/\r?\n/).map(x => x.trim());
    if (!active || !prompts.length) { this.showToast('Add at least one test prompt.', 'error'); return; }
    let baseline = {};
    try { baseline = JSON.parse(localStorage.getItem('teachable_ai_eval_baseline_v6') || '{}'); } catch (e) {}
    const prior = baseline[active.id] || {};
    const overlap = (a, b) => {
      const A = new Set((String(a || '').toLowerCase().match(/[a-z0-9]+/g) || []));
      const B = new Set((String(b || '').toLowerCase().match(/[a-z0-9]+/g) || []));
      if (!A.size && !B.size) return 1;
      let n = 0;
      A.forEach(x => { if (B.has(x)) n++; });
      return n / Math.max(1, new Set([...A, ...B]).size);
    };
    const results = prompts.map((prompt, i) => {
      const t = performance.now();
      const response = this.generatePreferred(active.engine, prompt, 80, 0.25, 1.3, 8);
      const latencyMs = performance.now() - t;
      const second = this.generatePreferred(active.engine, prompt, 80, 0.25, 1.3, 8);
      const metrics = this.scoreBenchmark(response, gold[i] || '', latencyMs);
      metrics.consistency = overlap(response, second);
      const key = prompt.toLowerCase().replace(/\s+/g, ' '), previous = prior[key];
      metrics.regression = previous ? overlap(response, previous) : null;
      metrics.regressionStatus = previous ? (metrics.regression < 0.35 ? 'possible-regression' : 'stable') : 'new-baseline';
      const p = prompt.toLowerCase();
      metrics.testType = /why|explain|reason|compare|solve|therefore|because/.test(p) ? 'Reasoning' : (/who|what|when|where|define|name|capital|fact/.test(p) ? 'Factual recall' : (/exactly|list|format|only|json|bullet|word/.test(p) ? 'Instruction following' : 'General / regression'));
      return { prompt, response, repeatResponse: second, expected: gold[i] || '', metrics, model: active.name, modelType: active.type, testType: metrics.testType, createdAt: new Date().toISOString() };
    });
    baseline[active.id] = Object.fromEntries(results.map(r => [r.prompt.toLowerCase().replace(/\s+/g, ' '), r.response]));
    try { localStorage.setItem('teachable_ai_eval_baseline_v6', JSON.stringify(baseline)); } catch (e) {}
    this.lastEvaluation = results;
    const avg = key => results.reduce((n, r) => n + (r.metrics[key] || 0), 0) / results.length;
    document.getElementById('eval-results').innerHTML = results.map((r, i) => `<article class="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-2"><div class="text-[11px] text-brand-300 font-semibold">${i + 1}. ${this.escapeHTML(r.prompt)} <span class="text-slate-500">· ${r.testType}</span></div><p class="text-xs text-slate-200 whitespace-pre-wrap">${this.escapeHTML(r.response || '(empty response)')}</p>${r.expected ? `<p class="text-[11px] text-slate-500">Expected: ${this.escapeHTML(r.expected)}</p>` : ''}<div class="flex flex-wrap gap-3 text-[10px] font-mono text-slate-400"><span>Gold F1 ${r.expected ? r.metrics.f1.toFixed(2) : '—'}</span><span>Repeat consistency ${(r.metrics.consistency * 100).toFixed(0)}%</span><span>Regression ${r.metrics.regression === null ? 'baseline' : r.metrics.regressionStatus + ' ' + r.metrics.regression.toFixed(2)}</span><span>${r.metrics.latencyMs.toFixed(1)} ms</span></div></article>`).join('');
    document.getElementById('eval-summary').textContent = `${results.length} prompts · avg latency ${avg('latencyMs').toFixed(1)} ms · repeat consistency ${(avg('consistency') * 100).toFixed(0)}% · possible regressions ${results.filter(r => r.metrics.regressionStatus === 'possible-regression').length}.`;
  }

  async compareAgainstReference() {
    const active = this.getActiveModelObj();
    const prompts = document.getElementById('eval-prompts').value.split(/\r?\n/).map(x => x.trim()).filter(Boolean).slice(0, 20);
    const gold = document.getElementById('eval-gold').value.split(/\r?\n/).map(x => x.trim());
    if (!prompts.length) { this.showToast('Add evaluation prompts first.', 'error'); return; }
    const refId = document.getElementById('panel-model-select')?.value || '__ref_smollm2_360__';
    const ref = this.getLocalReferenceDefinition(refId);
    const btn = document.getElementById('compare-reference-btn');
    btn.disabled = true;
    btn.textContent = 'Preparing assistant…';
    const results = [];
    try {
      await this.prepareLocalReference(ref.key);
      for (let i = 0; i < prompts.length; i++) {
        const prompt = prompts[i], expected = gold[i] || '';
        const localStart = performance.now();
        const local = this.generatePreferred(active.engine, prompt, 80, 0.25, 1.3, 8);
        const localMs = performance.now() - localStart;
        const refStart = performance.now();
        const reference = await this.requestLocalReference(refId, prompt);
        const refMs = performance.now() - refStart;
        results.push({ prompt, expected, model: active.name, response: local, metrics: this.scoreBenchmark(local, expected, localMs) });
        results.push({ prompt, expected, model: ref.name, response: reference, metrics: this.scoreBenchmark(reference, expected, refMs) });
      }
      this.lastEvaluation = results;
      const groups = [active.name, ref.name].map(name => {
        const rows = results.filter(r => r.model === name), avg = k => rows.reduce((n, r) => n + (r.metrics[k] || 0), 0) / Math.max(1, rows.length);
        return `<div class="p-3 rounded-xl bg-slate-900 border border-slate-800"><div class="text-xs font-semibold text-white">${this.escapeHTML(name)}</div><div class="text-[11px] text-slate-400">${rows.length} prompts · ${avg('latencyMs').toFixed(0)} ms avg · ${gold.some(Boolean) ? 'Gold F1 ' + avg('f1').toFixed(3) : 'Add expected answers for lexical F1'}</div></div>`;
      }).join('');
      const details = results.map(r => `<article class="p-3 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1"><div class="text-[11px] text-amber-300 font-semibold">${this.escapeHTML(r.model)} · ${this.escapeHTML(r.prompt)}</div><p class="text-xs text-slate-200 whitespace-pre-wrap">${this.escapeHTML(r.response || '(empty response)')}</p><div class="text-[10px] text-slate-500">${r.expected ? 'Gold F1 ' + r.metrics.f1.toFixed(2) + ' · ' : ''}${r.metrics.latencyMs.toFixed(0)} ms</div></article>`).join('');
      document.getElementById('eval-results').innerHTML = groups + details;
      document.getElementById('eval-summary').textContent = `Compared ${active.name} against ${ref.name} on ${prompts.length} identical prompts.`;
    } catch (err) {
      this.showToast('Assistant comparison failed: ' + err.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Compare active vs selected assistant';
    }
  }

  compareEvaluation() {
    const prompts = document.getElementById('eval-prompts').value.split(/\r?\n/).map(x => x.trim()).filter(Boolean).slice(0, 30);
    const gold = document.getElementById('eval-gold').value.split(/\r?\n/).map(x => x.trim());
    if (!prompts.length || !this.models.length) { this.showToast('Add prompts and at least one model.', 'error'); return; }
    const results = [];
    for (const model of this.models) {
      for (let i = 0; i < prompts.length; i++) {
        const t = performance.now();
        const response = model.engine.generate(prompts[i], 80, 0.25, 1.3, 8);
        const latencyMs = performance.now() - t;
        results.push({ prompt: prompts[i], response, expected: gold[i] || '', metrics: this.scoreBenchmark(response, gold[i] || '', latencyMs), model: model.name, modelType: model.type, createdAt: new Date().toISOString() });
      }
    }
    this.lastEvaluation = results;
    const avg = (arr, key) => arr.reduce((n, r) => n + (r.metrics[key] || 0), 0) / Math.max(1, arr.length);
    const groups = this.models.map(m => {
      const rows = results.filter(r => r.model === m.name);
      return `<div class="p-3 rounded-xl bg-slate-900 border border-slate-800"><div class="text-xs font-semibold text-white">${this.escapeHTML(m.name)}</div><div class="text-[11px] text-slate-400">${rows.length} prompts · ${avg(rows, 'latencyMs').toFixed(1)} ms avg · ${gold.some(Boolean) ? 'F1 ' + avg(rows, 'f1').toFixed(3) : 'No gold score'}</div></div>`;
    }).join('');
    const details = results.map(r => `<article class="p-3 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1"><div class="text-[11px] text-amber-300 font-semibold">${this.escapeHTML(r.model)} · ${this.escapeHTML(r.prompt)}</div><p class="text-xs text-slate-200 whitespace-pre-wrap">${this.escapeHTML(r.response || '(empty response)')}</p><div class="text-[10px] text-slate-500">${r.expected ? 'F1 ' + r.metrics.f1.toFixed(2) + ' · ' : ''}${r.metrics.latencyMs.toFixed(1)} ms · repeat ${(r.metrics.repeated * 100).toFixed(0)}%</div></article>`).join('');
    document.getElementById('eval-results').innerHTML = groups + details;
    document.getElementById('eval-summary').textContent = `Compared ${this.models.length} model(s) on ${prompts.length} identical prompts.`;
  }

  exportEvaluation() {
    if (!this.lastEvaluation.length) { this.showToast('Run an evaluation first.', 'error'); return; }
    const blob = new Blob([JSON.stringify({ appVersion: '0.2.2', results: this.lastEvaluation }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = 'teachable-ai-evaluation-0.2.2.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ============ Speed benchmark ============ */
  initBenchmark() {
    document.getElementById('run-bench-btn')?.addEventListener('click', () => this.runBenchmark());
    document.getElementById('export-bench-btn')?.addEventListener('click', () => this.exportBenchmark());
  }

  renderBenchTargets() {
    const box = document.getElementById('bench-targets');
    if (!box) return;
    const prev = new Set(Array.from(box.querySelectorAll('input:checked')).map(i => i.value));
    const first = box.children.length === 0;
    const items = this.models.map(m => ({ id: 'm:' + m.id, label: m.name + ' · ' + (m.type === 'neural' ? 'MLP' : (m.type === 'attention' ? 'Attention' : 'TF-IDF')), ok: true }));
    ['__ref_smollm2_360__', '__ref_qwen05__', '__ref_smollm2_135__'].forEach(id => {
      const ref = this.getLocalReferenceDefinition(id), ready = !!this.localReferencePipelines[ref.key];
      items.push({ id: 'r:' + id, label: ref.name + (ready ? ' · assistant' : ' · download first'), ok: ready });
    });
    box.innerHTML = items.map(it => `<label class="flex items-center gap-2 p-2 rounded-lg bg-slate-950 border border-slate-800 text-[11px] ${it.ok ? 'text-slate-200 cursor-pointer' : 'text-slate-600'}"><input type="checkbox" value="${this.escapeHTML(it.id)}" ${it.ok ? '' : 'disabled'} ${(first ? it.ok : prev.has(it.id)) ? 'checked' : ''} class="accent-amber-400"><span class="truncate">${this.escapeHTML(it.label)}</span></label>`).join('');
  }

  async runBenchmark() {
    const $ = id => document.getElementById(id);
    const prompts = $('eval-prompts').value.split(/\r?\n/).map(x => x.trim()).filter(Boolean).slice(0, 20);
    const targets = Array.from($('bench-targets').querySelectorAll('input:checked')).map(i => i.value);
    if (!prompts.length) { this.showToast('Add test prompts in the box above.', 'error'); return; }
    if (!targets.length) { this.showToast('Select at least one model.', 'error'); return; }
    const runs = parseInt($('bench-runs').value) || 1;
    const btn = $('run-bench-btn'), status = $('bench-status');
    btn.disabled = true;
    const results = [];
    const tick = () => new Promise(r => setTimeout(r, 0));
    try {
      for (const t of targets) {
        const rows = [];
        let name = '', kind = '';
        if (t.startsWith('m:')) {
          const m = this.models.find(x => 'm:' + x.id === t);
          if (!m) continue;
          name = m.name; kind = m.type === 'neural' ? 'Neural MLP' : (m.type === 'attention' ? 'Attention' : 'TF-IDF');
          for (const pr of prompts) for (let r = 0; r < runs; r++) {
            status.textContent = `${name}: "${pr.slice(0, 30)}" run ${r + 1}/${runs}`;
            await tick();
            const hit = Boolean(m.engine.memory && m.engine.memory.queryBestMatch(pr));
            const t0 = performance.now();
            const resp = this.generatePreferred(m.engine, pr, 45, 0.25, 1.3, 8).replace('[[[UNKNOWN_RESPONSE]]]', '');
            const ms = performance.now() - t0;
            let tokens = 0;
            try { tokens = m.engine.tokenizer.tokenize(resp).length; } catch (e) { tokens = resp.split(/\s+/).filter(Boolean).length; }
            rows.push({ ms, tokens, ttft: null, hit });
          }
        } else {
          const id = t.slice(2), ref = this.getLocalReferenceDefinition(id);
          name = ref.name; kind = 'Assistant';
          for (const pr of prompts) for (let r = 0; r < runs; r++) {
            status.textContent = `${name}: "${pr.slice(0, 30)}" run ${r + 1}/${runs}`;
            await tick();
            const st = {};
            try { await this.requestLocalReference(id, pr, [], { stats: st, onToken: () => {} }); }
            catch (err) { this.showToast(name + ' failed: ' + err.message, 'error'); continue; }
            rows.push({ ms: st.ms, tokens: st.tokens, ttft: st.ttftMs, hit: false });
          }
        }
        if (!rows.length) continue;
        const totMs = rows.reduce((n, r) => n + r.ms, 0), totTok = rows.reduce((n, r) => n + r.tokens, 0);
        const ttfts = rows.filter(r => r.ttft !== null);
        results.push({
          name, kind, samples: rows.length,
          avgLatencyMs: totMs / rows.length,
          avgTokens: totTok / rows.length,
          tokensPerSec: totTok / Math.max(totMs / 1000, 0.0001),
          avgFirstTokenMs: ttfts.length ? ttfts.reduce((n, r) => n + r.ttft, 0) / ttfts.length : null,
          memoryHitPct: rows.filter(r => r.hit).length / rows.length * 100
        });
      }
      this.lastBench = { appVersion: '0.2.2', ranAt: new Date().toISOString(), prompts, runsPerPrompt: runs, results };
      this.renderBenchResults(results);
      status.textContent = `Done · ${results.length} model(s) · ${prompts.length} prompt(s) × ${runs} run(s).`;
    } finally { btn.disabled = false; }
  }

  renderBenchResults(results) {
    const box = document.getElementById('bench-results');
    if (!results.length) { box.innerHTML = '<p class="text-xs text-slate-500">No results.</p>'; return; }
    const sorted = [...results].sort((a, b) => b.tokensPerSec - a.tokensPerSec);
    const maxLog = Math.log10(sorted[0].tokensPerSec + 1) || 1;
    const fmt = ms => ms === null ? '—' : (ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(2) + ' s');
    const bars = sorted.map(r => `<div class="space-y-0.5"><div class="flex justify-between text-[10px] text-slate-400"><span class="truncate">${this.escapeHTML(r.name)}</span><span class="font-mono text-amber-300">${r.tokensPerSec >= 1000 ? '1000+' : r.tokensPerSec.toFixed(1)} t/s</span></div><div class="h-2 rounded bg-slate-800 overflow-hidden"><div class="h-full bg-gradient-to-r from-amber-500 to-brand-400" style="width:${Math.max(3, Math.log10(r.tokensPerSec + 1) / maxLog * 100)}%"></div></div></div>`).join('');
    const rowsHtml = sorted.map(r => `<tr class="border-t border-slate-800"><td class="py-1.5 pr-2 text-slate-200">${this.escapeHTML(r.name)}</td><td class="pr-2 text-slate-400">${this.escapeHTML(r.kind)}</td><td class="pr-2 text-right font-mono text-amber-300">${r.tokensPerSec >= 1000 ? '1000+' : r.tokensPerSec.toFixed(1)}</td><td class="pr-2 text-right font-mono">${fmt(r.avgLatencyMs)}</td><td class="pr-2 text-right font-mono">${fmt(r.avgFirstTokenMs)}</td><td class="pr-2 text-right font-mono">${r.avgTokens.toFixed(1)}</td><td class="text-right font-mono">${r.memoryHitPct.toFixed(0)}%</td></tr>`).join('');
    box.innerHTML = `<div class="space-y-2">${bars}</div><p class="text-[10px] text-slate-500">Bars use a log scale so instant retrieval lookups and neural generation can share one chart.</p><div class="overflow-x-auto"><table class="w-full text-[11px] text-slate-300"><thead><tr class="text-slate-500 text-left"><th class="pb-1 font-medium">Model</th><th class="font-medium">Type</th><th class="text-right font-medium pr-2">t/s</th><th class="text-right font-medium pr-2">Latency</th><th class="text-right font-medium pr-2">First token</th><th class="text-right font-medium pr-2">Avg tokens</th><th class="text-right font-medium">From memory</th></tr></thead><tbody>${rowsHtml}</tbody></table></div><p class="text-[10px] text-slate-500">"From memory" counts answers returned straight from retrieval memory. Those are instant lookups, so their t/s is not comparable to neural generation.</p>`;
  }

  exportBenchmark() {
    if (!this.lastBench) { this.showToast('Run the speed benchmark first.', 'error'); return; }
    const blob = new Blob([JSON.stringify(this.lastBench, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = 'teachable-ai-speed-benchmark-' + Date.now() + '.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
