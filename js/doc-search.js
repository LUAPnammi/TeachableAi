// Document search (BM25-style) used by the assistant panel to answer from your files.

export function ragChunkText(text, max = 700) {
  const out = [];
  let buf = '';
  const push = () => { const t = buf.trim(); if (t.length >= 20) out.push(t); buf = ''; };
  for (const raw of String(text || '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) { if (buf.length >= 250) push(); continue; }
    if (line.length > max) {
      push();
      line.split(/(?<=[.!?])\s+/).forEach(sn => { if (buf.length + sn.length > max) push(); buf += (buf ? ' ' : '') + sn; });
      push();
      continue;
    }
    if (buf.length + line.length > max) push();
    buf += (buf ? '\n' : '') + line;
  }
  push();
  return out.slice(0, 20000);
}

export class DocSearch {
  static terms(str) {
    const stop = DocSearch.STOP;
    return (String(str || '').toLowerCase().match(/[a-z0-9]{3,}/g) || []).filter(w => !stop.has(w)).map(w => w.length > 5 ? w.slice(0, 5) : w);
  }
  constructor(chunks) {
    this.chunks = chunks;
    this.df = new Map();
    chunks.forEach(c => {
      c.tf = new Map();
      const terms = DocSearch.terms(c.text);
      c.len = terms.length || 1;
      terms.forEach(t => c.tf.set(t, (c.tf.get(t) || 0) + 1));
      c.tf.forEach((_, t) => this.df.set(t, (this.df.get(t) || 0) + 1));
    });
    this.avgLen = chunks.reduce((n, c) => n + c.len, 0) / Math.max(1, chunks.length);
  }
  search(query, k = 3) {
    const q = [...new Set(DocSearch.terms(query))];
    if (!q.length || !this.chunks.length) return [];
    const N = this.chunks.length, k1 = 1.5, b = 0.75;
    const need = q.length <= 2 ? 1 : Math.ceil(q.length * 0.4);
    const scored = [];
    for (const c of this.chunks) {
      let score = 0, matched = 0;
      for (const t of q) {
        const tf = c.tf.get(t);
        if (!tf) continue;
        matched++;
        const df = this.df.get(t) || 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * c.len / this.avgLen));
      }
      if (matched >= need && score >= 1.2) scored.push({ chunk: c, score });
    }
    scored.sort((a, b2) => b2.score - a.score);
    const seen = new Set(), out = [];
    for (const h of scored) {
      const key = h.chunk.text.slice(0, 80);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(h);
      if (out.length >= k) break;
    }
    return out;
  }
}
DocSearch.STOP = new Set(['the','and','for','are','with','that','this','what','how','can','you','your','does','from','have','about','tell','please','explain','which','where','when','who','why','into','there','their','they','them','then','than','will','would','could','should','was','were','has','had','not','but','all','any','our','out','use','using','used','its','also','doc','docs','document','documents','file','files','dataset','datasets','summarize','summary','mine','give','show','find','search']);
