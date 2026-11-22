// Word-level tokenizer with character fallback.

export class Tokenizer {
  constructor(includeCharacterFallback = true) {
    this.PAD = '<pad>';
    this.UNK = '<unk>';
    this.BOS = '<bos>';
    this.EOS = '<eos>';
    this.SEP = '<sep>';
    this.SUBWORD_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789'_-áéíóúñü";

    this.vocab = new Map();
    this.idToToken = [];
    [this.PAD, this.UNK, this.BOS, this.EOS, this.SEP].forEach(t => this.addToken(t));
    if (includeCharacterFallback) Array.from(this.SUBWORD_CHARS).forEach(ch => this.addToken('~c~' + ch));
  }

  addToken(token) {
    if (!this.vocab.has(token)) {
      const id = this.idToToken.length;
      this.vocab.set(token, id);
      this.idToToken.push(token);
      return id;
    }
    return this.vocab.get(token);
  }

  tokenize(text) {
    if (!text) return [];
    const normalized = text
      .replace(/\r\n/g, '\n')
      .replace(/\n{2,}/g, ' <eos> ')
      .replace(/\n/g, ' <sep> ');
    const matches = normalized.toLowerCase().match(/<eos>|<sep>|<bos>|<pad>|<unk>|\b[\w']+\b|[^\w\s]/g);
    return matches || [];
  }

  encode(tokens) {
    const encoded = [];
    for (const token of tokens) {
      if (this.vocab.has(token)) { encoded.push(this.vocab.get(token)); continue; }
      const chars = Array.from(String(token).toLowerCase());
      if (/^[\w'_-]+$/u.test(token) && chars.length && chars.every(ch => this.vocab.has('~c~' + ch))) {
        chars.forEach(ch => encoded.push(this.vocab.get('~c~' + ch)));
      } else encoded.push(this.vocab.get(this.UNK));
    }
    return encoded;
  }

  decode(ids) {
    const special = new Set([this.PAD, this.BOS, this.EOS, this.SEP, this.UNK]);
    const output = []; let charBuffer = '';
    for (const id of ids) {
      const token = this.idToToken[id] || this.UNK;
      if (token.startsWith('~c~')) { charBuffer += token.slice(3); continue; }
      if (charBuffer) { output.push(charBuffer); charBuffer = ''; }
      if (!special.has(token)) output.push(token);
    }
    if (charBuffer) output.push(charBuffer);
    return output;
  }

  buildVocabFromText(text, maxVocab = Infinity) {
    const tokens = this.tokenize(text);
    if (Number.isFinite(maxVocab) && maxVocab > this.idToToken.length) {
      const counts = new Map();
      tokens.forEach(t => counts.set(t, (counts.get(t) || 0) + 1));
      const reserved = new Set(this.idToToken);
      [...counts.entries()]
        .filter(([t]) => !reserved.has(t))
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .slice(0, Math.max(0, maxVocab - this.idToToken.length))
        .forEach(([t]) => this.addToken(t));
    } else if (!Number.isFinite(maxVocab)) {
      tokens.forEach(t => this.addToken(t));
    }
    return tokens;
  }

  size() {
    return this.idToToken.length;
  }
}
