// TF-IDF retrieval memory: indexes user:/ai: pairs and paragraphs.

export class SemanticMemory {
  constructor() {
    this.qaPairs = [];
    this.sentences = [];
    this.idf = new Map();
  }

  indexText(text) {
    this.qaPairs = [];
    this.sentences = [];

    const lines = String(text || '').split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    let pendingUser = null;

    for (const line of lines) {
      const userMatch = line.match(/^(?:user|human|q)\s*:\s*(.+)$/i);
      const aiMatch = line.match(/^(?:ai|bot|assistant|a)\s*:\s*(.+)$/i);

      if (userMatch) {
        pendingUser = userMatch[1].trim();
      } else if (aiMatch && pendingUser) {
        this.qaPairs.push({
          prompt: pendingUser,
          response: aiMatch[1].trim(),
          terms: this.extractTerms(pendingUser)
        });
        pendingUser = null;
      } else if (!userMatch && !aiMatch && line.length > 20) {
        const rawSentences = line.match(/[^.!?]+[.!?]+/g) || [line];
        rawSentences.forEach(s => {
          const clean = s.trim();
          if (clean.split(/\s+/).length >= 4) {
            this.sentences.push({
              text: clean,
              terms: this.extractTerms(clean)
            });
          }
        });
      }
    }

    const docCount = Math.max(1, this.qaPairs.length + this.sentences.length);
    const df = new Map();
    [...this.qaPairs, ...this.sentences].forEach(item => {
      const unique = new Set(item.terms);
      unique.forEach(t => df.set(t, (df.get(t) || 0) + 1));
    });

    this.idf.clear();
    df.forEach((count, term) => {
      this.idf.set(term, Math.log((docCount + 1) / (count + 0.5)) + 1);
    });
  }

  extractTerms(str) {
    const stop = new Set(['the', 'is', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'it', 'that', 'for', 'on', 'with', 'as', 'are', 'this', 'by', 'be', 'at']);
    const words = (String(str || '').toLowerCase().match(/\b[\w']+\b/g) || []);
    const filtered = words.filter(w => !stop.has(w));
    return filtered.length > 0 ? filtered : words;
  }

  similarity(termsA, termsB) {
    if (!termsA.length || !termsB.length) return 0;
    const vecA = new Map();
    const vecB = new Map();
    termsA.forEach(t => vecA.set(t, (vecA.get(t) || 0) + (this.idf.get(t) || 1)));
    termsB.forEach(t => vecB.set(t, (vecB.get(t) || 0) + (this.idf.get(t) || 1)));

    let dot = 0, normA = 0, normB = 0;
    vecA.forEach((val, k) => {
      normA += val * val;
      if (vecB.has(k)) dot += val * vecB.get(k);
    });
    vecB.forEach(val => { normB += val * val; });
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
  }

  queryCandidates(prompt, limit = 5) {
    const clean = String(prompt || '').toLowerCase().trim().replace(/[?!.]+$/g, '');
    const qTerms = this.extractTerms(clean);
    const candidates = [];
    for (const qa of this.qaPairs) {
      const pClean = qa.prompt.toLowerCase().trim().replace(/[?!.]+$/g, '');
      const score = pClean === clean ? 1 : this.similarity(qTerms, qa.terms);
      if (score > 0) candidates.push({ text: qa.response, prompt: qa.prompt, score, source: 'qa' });
    }
    for (const sent of this.sentences) {
      const score = this.similarity(qTerms, sent.terms);
      if (score > 0) candidates.push({ text: sent.text, prompt: '', score, source: 'paragraph' });
    }
    return candidates.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  queryBestMatch(prompt) {
    const clean = String(prompt || '').toLowerCase().trim().replace(/[?!.]+$/g, '');
    const qTerms = this.extractTerms(clean);
    const ranked = this.queryCandidates(prompt, 1)[0];
    if (ranked && ((ranked.source === 'qa' && ranked.score >= 0.48) || (ranked.source === 'paragraph' && ranked.score >= 0.42))) return ranked;

    for (const qa of this.qaPairs) {
      const pClean = qa.prompt.toLowerCase().trim().replace(/[?!.]+$/g, '');
      if (pClean === clean) return { text: qa.response, score: 1.0, source: 'qa' };
    }
    return null;
  }
}
