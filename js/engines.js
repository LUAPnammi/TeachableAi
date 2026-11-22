// The three trainable engines: Neural MLP, Tiny Attention LM and TF-IDF + N-Gram.

import { SemanticMemory } from './semantic-memory.js';
import { Tokenizer } from './tokenizer.js';

export class TinyNeuralLM {
  constructor(contextSize = 4, embDim = 20, hiddenDim = 48) {
    this.contextSize = contextSize;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;
    this.tokenizer = new Tokenizer();
    this.memory = new SemanticMemory();

    this.E = null;
    this.W1 = null;
    this.b1 = null;
    this.W2 = null;
    this.b2 = null;

    this.lossHistory = [];
    this.isTrained = false;
    this.epochsTrained = 0;
    this.rawCorpus = '';
  }

  initWeights() {
    const V = this.tokenizer.size();
    const inDim = this.contextSize * this.embDim;

    this.E = new Float32Array(V * this.embDim);
    for (let i = 0; i < this.E.length; i++) {
      this.E[i] = (Math.random() - 0.5) * (2 / Math.sqrt(Math.max(1, this.embDim)));
    }

    this.W1 = new Float32Array(inDim * this.hiddenDim);
    for (let i = 0; i < this.W1.length; i++) {
      this.W1[i] = (Math.random() - 0.5) * (2 / Math.sqrt(Math.max(1, inDim)));
    }

    this.b1 = new Float32Array(this.hiddenDim);

    this.W2 = new Float32Array(this.hiddenDim * V);
    for (let i = 0; i < this.W2.length; i++) {
      this.W2[i] = (Math.random() - 0.5) * (2 / Math.sqrt(Math.max(1, this.hiddenDim)));
    }

    this.b2 = new Float32Array(V);
  }

  getNumParameters() {
    if (!this.E) return 0;
    return this.E.length + this.W1.length + this.b1.length + this.W2.length + this.b2.length;
  }

  prepareDataset(text, augmentShort = true, maxExamples = Infinity, maxVocab = Infinity) {
    this.rawCorpus = text;
    this.memory.indexText(text);
    this.tokenizer = new Tokenizer();

    let processedText = text;
    if (augmentShort && text.length < 2500) {
      processedText = text + '\n\n' + text + '\n\n' + text;
    }

    const tokens = this.tokenizer.buildVocabFromText(processedText, maxVocab);
    this.initWeights();

    const encoded = this.tokenizer.encode(tokens);
    const X = [];
    const Y = [];
    const padId = this.tokenizer.vocab.get(this.tokenizer.PAD);

    const sampleStride = Number.isFinite(maxExamples) && maxExamples > 0 ? Math.max(1, Math.ceil(encoded.length / maxExamples)) : 1;
    for (let i = 0; i < encoded.length; i += sampleStride) {
      const context = [];
      for (let c = this.contextSize; c > 0; c--) {
        const idx = i - c;
        context.push(idx >= 0 ? encoded[idx] : padId);
      }
      X.push(context);
      Y.push(encoded[i]);
    }

    return { X, Y, totalTokens: tokens.length };
  }

  forward(contextIds) {
    const V = this.tokenizer.size();
    const inDim = this.contextSize * this.embDim;

    const xEmb = new Float32Array(inDim);
    for (let c = 0; c < this.contextSize; c++) {
      const tokenVal = contextIds[c];
      const tid = (tokenVal !== undefined && tokenVal < V) ? tokenVal : 0;
      for (let d = 0; d < this.embDim; d++) {
        xEmb[c * this.embDim + d] = this.E[tid * this.embDim + d];
      }
    }

    const hOut = new Float32Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < inDim; i++) {
        sum += xEmb[i] * this.W1[i * this.hiddenDim + j];
      }
      hOut[j] = Math.tanh(sum);
    }

    const logits = new Float32Array(V);
    let maxLogit = -Infinity;
    for (let k = 0; k < V; k++) {
      let sum = this.b2[k];
      for (let j = 0; j < this.hiddenDim; j++) {
        sum += hOut[j] * this.W2[j * V + k];
      }
      logits[k] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    const probs = new Float32Array(V);
    let sumExp = 0;
    for (let k = 0; k < V; k++) {
      const expVal = Math.exp(logits[k] - maxLogit);
      probs[k] = expVal;
      sumExp += expVal;
    }
    for (let k = 0; k < V; k++) {
      probs[k] /= sumExp;
    }

    return { xEmb, hOut, probs };
  }

  trainStep(contextIds, targetId, lr) {
    const V = this.tokenizer.size();
    const inDim = this.contextSize * this.embDim;
    const { xEmb, hOut, probs } = this.forward(contextIds);

    const targetProb = Math.max(probs[targetId], 1e-10);
    const loss = -Math.log(targetProb);

    const dLogits = new Float32Array(probs);
    dLogits[targetId] -= 1.0;

    const dHOut = new Float32Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sumGrad = 0;
      for (let k = 0; k < V; k++) {
        const grad = dLogits[k];
        sumGrad += grad * this.W2[j * V + k];
        this.W2[j * V + k] -= lr * grad * hOut[j];
        this.b2[k] -= lr * grad;
      }
      dHOut[j] = sumGrad;
    }

    const dHIn = new Float32Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      dHIn[j] = dHOut[j] * (1.0 - hOut[j] * hOut[j]);
    }

    const dXEmb = new Float32Array(inDim);
    for (let i = 0; i < inDim; i++) {
      let sumGrad = 0;
      for (let j = 0; j < this.hiddenDim; j++) {
        const grad = dHIn[j];
        sumGrad += grad * this.W1[i * this.hiddenDim + j];
        this.W1[i * this.hiddenDim + j] -= lr * grad * xEmb[i];
      }
      dXEmb[i] = sumGrad;
    }

    for (let j = 0; j < this.hiddenDim; j++) {
      this.b1[j] -= lr * dHIn[j];
    }

    for (let c = 0; c < this.contextSize; c++) {
      const tid = contextIds[c];
      if (tid !== undefined && tid < V) {
        for (let d = 0; d < this.embDim; d++) {
          this.E[tid * this.embDim + d] -= lr * dXEmb[c * this.embDim + d];
        }
      }
    }

    return loss;
  }

  generate(prompt, maxTokens = 45, temperature = 0.35, repetitionPenalty = 1.35, topK = 5) {
    if (!this.E) return "Model has not been trained yet. Open Train Studio or Corpus Library to train me!";

    const semanticHit = this.memory.queryBestMatch(prompt);
    if (semanticHit && (semanticHit.score >= 0.72 || temperature <= 0.45)) {
      return semanticHit.text;
    }

    const hasTurnPrefix = /^(user|ai)\s*:/i.test(prompt.trim());
    const conditionedPrompt = hasTurnPrefix ? prompt : ('user: ' + prompt.trim() + '\nai:');

    const promptTokens = this.tokenizer.tokenize(conditionedPrompt);
    let encoded = this.tokenizer.encode(promptTokens);
    const padId = this.tokenizer.vocab.get(this.tokenizer.PAD);
    const unkId = this.tokenizer.vocab.get(this.tokenizer.UNK);
    const eosId = this.tokenizer.vocab.get(this.tokenizer.EOS);
    const sepId = this.tokenizer.vocab.get(this.tokenizer.SEP);

    if (encoded.length === 0) encoded = [padId];

    const generatedIds = [];
    const seenCounts = new Map();

    for (let step = 0; step < maxTokens; step++) {
      const currentSeq = [...encoded, ...generatedIds];
      const context = [];
      for (let c = this.contextSize; c > 0; c--) {
        const idx = currentSeq.length - c;
        context.push(idx >= 0 ? currentSeq[idx] : padId);
      }

      const { probs } = this.forward(context);

      const candidates = [];
      for (let i = 0; i < probs.length; i++) {
        if (i === padId || i === unkId) continue;
        const tokStr = this.tokenizer.idToToken[i];
        if (tokStr === 'user' || tokStr === 'ai' || tokStr === 'human') continue;

        let p = probs[i];
        if (seenCounts.has(i)) {
          p /= Math.pow(repetitionPenalty, seenCounts.get(i));
        }
        p = Math.pow(Math.max(p, 1e-10), 1 / Math.max(temperature, 0.1));
        candidates.push({ id: i, prob: p });
      }

      if (candidates.length === 0) break;

      candidates.sort((a, b) => b.prob - a.prob);
      const topCandidates = candidates.slice(0, Math.max(1, topK));
      const sum = topCandidates.reduce((acc, c) => acc + c.prob, 0);

      let rand = Math.random() * sum;
      let nextId = topCandidates[0].id;
      for (const cand of topCandidates) {
        rand -= cand.prob;
        if (rand <= 0) {
          nextId = cand.id;
          break;
        }
      }

      if (nextId === eosId || nextId === sepId) {
        if (generatedIds.length > 0) break;
        continue;
      }

      generatedIds.push(nextId);
      seenCounts.set(nextId, (seenCounts.get(nextId) || 0) + 1);

      const tokStr = this.tokenizer.idToToken[nextId];
      if (['.', '!', '?'].includes(tokStr) && generatedIds.length >= 4) {
        break;
      }
    }

    const decoded = this.tokenizer.decode(generatedIds);
    let result = decoded.join(' ')
      .replace(/\s+([,.?!':;])/g, '$1')
      .replace(/\s+'/g, "'");

    if (result.length > 0) {
      result = result.charAt(0).toUpperCase() + result.slice(1);
    }

    return result || (semanticHit ? semanticHit.text : "I am listening! Train me with more examples in Train Studio.");
  }

  toJSON() {
    return {
      type: 'neural',
      contextSize: this.contextSize,
      embDim: this.embDim,
      hiddenDim: this.hiddenDim,
      vocab: this.tokenizer.idToToken,
      lossHistory: this.lossHistory,
      epochsTrained: this.epochsTrained,
      rawCorpus: this.rawCorpus,
      E: Array.from(this.E || []),
      W1: Array.from(this.W1 || []),
      b1: Array.from(this.b1 || []),
      W2: Array.from(this.W2 || []),
      b2: Array.from(this.b2 || [])
    };
  }

  fromJSON(data) {
    this.contextSize = data.contextSize || 4;
    this.embDim = data.embDim || 20;
    this.hiddenDim = data.hiddenDim || 48;
    this.lossHistory = data.lossHistory || [];
    this.epochsTrained = data.epochsTrained || 0;
    this.rawCorpus = data.rawCorpus || '';

    this.tokenizer = new Tokenizer(false);
    if (data.vocab) data.vocab.forEach(t => this.tokenizer.addToken(t));
    if (this.rawCorpus) this.memory.indexText(this.rawCorpus);

    this.E = Float32Array.from(data.E || []);
    this.W1 = Float32Array.from(data.W1 || []);
    this.b1 = Float32Array.from(data.b1 || []);
    this.W2 = Float32Array.from(data.W2 || []);
    this.b2 = Float32Array.from(data.b2 || []);
    this.isTrained = true;
  }
}

export class TinyAttentionLM extends TinyNeuralLM {
  constructor(contextSize = 6, embDim = 20, hiddenDim = 48) {
    super(contextSize, embDim, hiddenDim);
    this.architecture = 'tiny-causal-attention-1head';
  }

  forward(contextIds) {
    const V = this.tokenizer.size();
    const inDim = this.contextSize * this.embDim;
    const ids = Array.from(contextIds || []).slice(-this.contextSize);
    while (ids.length < this.contextSize) ids.unshift(this.tokenizer.vocab.get(this.tokenizer.PAD));

    const xEmb = new Float32Array(inDim);
    const queryStart = (this.contextSize - 1) * this.embDim;
    const scores = [];
    for (let c = 0; c < this.contextSize; c++) {
      const tid = (ids[c] !== undefined && ids[c] < V) ? ids[c] : 0;
      let dot = 0;
      for (let d = 0; d < this.embDim; d++) {
        dot += this.E[ids[this.contextSize - 1] * this.embDim + d] * this.E[tid * this.embDim + d];
      }
      scores.push(dot / Math.sqrt(Math.max(1, this.embDim)));
    }

    const max = Math.max(...scores);
    const weights = scores.map(v => Math.exp(v - max));
    const sum = weights.reduce((a, b) => a + b, 0) || 1;

    for (let d = 0; d < this.embDim; d++) {
      let attended = 0;
      for (let c = 0; c < this.contextSize; c++) {
        const tid = (ids[c] !== undefined && ids[c] < V) ? ids[c] : 0;
        attended += (weights[c] / sum) * this.E[tid * this.embDim + d];
      }
      xEmb[queryStart + d] = attended;
    }

    for (let c = 0; c < this.contextSize - 1; c++) {
      const tid = ids[c];
      for (let d = 0; d < this.embDim; d++) {
        xEmb[c * this.embDim + d] = this.E[tid * this.embDim + d];
      }
    }

    const hOut = new Float32Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let v = this.b1[j];
      for (let i = 0; i < inDim; i++) v += xEmb[i] * this.W1[i * this.hiddenDim + j];
      hOut[j] = Math.tanh(v);
    }

    const logits = new Float32Array(V);
    let mx = -Infinity;
    for (let k = 0; k < V; k++) {
      let v = this.b2[k];
      for (let j = 0; j < this.hiddenDim; j++) v += hOut[j] * this.W2[j * V + k];
      logits[k] = v;
      if (v > mx) mx = v;
    }

    const probs = new Float32Array(V);
    let den = 0;
    for (let k = 0; k < V; k++) { probs[k] = Math.exp(logits[k] - mx); den += probs[k]; }
    for (let k = 0; k < V; k++) probs[k] /= den || 1;
    return { xEmb, hOut, probs, attention: weights.map(v => v / sum) };
  }

  toJSON() {
    const d = super.toJSON();
    d.type = 'attention';
    d.architecture = this.architecture;
    return d;
  }
}

export class HybridNGramEngine {
  constructor() {
    this.tokenizer = new Tokenizer();
    this.memory = new SemanticMemory();
    this.ngrams = new Map();
    this.isTrained = false;
    this.epochsTrained = 1;
    this.rawCorpus = '';
  }

  train(text) {
    this.rawCorpus = text;
    this.memory.indexText(text);
    this.tokenizer = new Tokenizer();
    this.ngrams = new Map();
    const tokens = this.tokenizer.buildVocabFromText(text);

    for (let order = 1; order <= 3; order++) {
      for (let i = 0; i < tokens.length - order; i++) {
        const context = tokens.slice(i, i + order).join('|');
        const nextToken = tokens[i + order];
        if (!this.ngrams.has(context)) this.ngrams.set(context, new Map());
        const countMap = this.ngrams.get(context);
        countMap.set(nextToken, (countMap.get(nextToken) || 0) + 1);
      }
    }
    this.isTrained = true;
  }

  async trainAsync(text, maxChars = 50000, shouldStop = () => false) {
    this.rawCorpus = String(text || '').slice(0, maxChars);
    this.memory.indexText(this.rawCorpus);
    this.tokenizer = new Tokenizer();
    const tokens = this.tokenizer.buildVocabFromText(this.rawCorpus);
    this.ngrams = new Map();
    let processed = 0;
    for (let order = 1; order <= 3; order++) {
      for (let i = 0; i < tokens.length - order; i++) {
        if (shouldStop()) { this.isTrained = this.ngrams.size > 0; return; }
        const context = tokens.slice(i, i + order).join('|');
        const nextToken = tokens[i + order];
        if (!this.ngrams.has(context)) this.ngrams.set(context, new Map());
        const countMap = this.ngrams.get(context);
        countMap.set(nextToken, (countMap.get(nextToken) || 0) + 1);
        processed++;
        if (processed % 1500 === 0) await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    this.isTrained = true;
  }

  getNumParameters() {
    let count = 0;
    this.ngrams.forEach(m => count += m.size);
    return count;
  }

  generate(prompt, maxTokens = 45, temperature = 0.35) {
    if (!this.isTrained) return "Model is empty. Load a pack from Corpus Library or train it in Train Studio!";

    const semanticHit = this.memory.queryBestMatch(prompt);
    if (semanticHit) return semanticHit.text;

    const promptTokens = this.tokenizer.tokenize('user: ' + prompt + '\nai:');
    let currentTokens = [...promptTokens];
    const generated = [];

    for (let step = 0; step < maxTokens; step++) {
      let nextToken = null;
      for (let order = 3; order >= 1; order--) {
        if (currentTokens.length >= order) {
          const ctx = currentTokens.slice(-order).join('|');
          if (this.ngrams.has(ctx)) {
            nextToken = this.sampleFromMap(this.ngrams.get(ctx), temperature);
            if (nextToken) break;
          }
        }
      }

      if (!nextToken || nextToken === 'user' || nextToken === '<eos>' || nextToken === '<sep>') break;
      if (nextToken === 'ai' || nextToken === ':') continue;

      generated.push(nextToken);
      currentTokens.push(nextToken);
      if (['.', '!', '?'].includes(nextToken) && generated.length >= 4) break;
    }

    let res = generated.join(' ')
      .replace(/\s+([,.?!':;])/g, '$1')
      .replace(/\s+'/g, "'");
    if (res.length > 0) res = res.charAt(0).toUpperCase() + res.slice(1);
    return res || "Tell me more so I can learn!";
  }

  sampleFromMap(countMap, temperature) {
    const items = Array.from(countMap.entries());
    let total = 0;
    const weights = items.map(([_, count]) => {
      const w = Math.pow(count, 1 / Math.max(temperature, 0.1));
      total += w;
      return w;
    });

    let rand = Math.random() * total;
    for (let i = 0; i < items.length; i++) {
      rand -= weights[i];
      if (rand <= 0) return items[i][0];
    }
    return items[0][0];
  }

  toJSON() {
    const ngramsObj = {};
    this.ngrams.forEach((map, key) => { ngramsObj[key] = Array.from(map.entries()); });
    return {
      type: 'ngram',
      vocab: this.tokenizer.idToToken,
      ngrams: ngramsObj,
      rawCorpus: this.rawCorpus
    };
  }

  fromJSON(data) {
    this.tokenizer = new Tokenizer(false);
    if (data.vocab) data.vocab.forEach(t => this.tokenizer.addToken(t));
    this.rawCorpus = data.rawCorpus || '';
    if (this.rawCorpus) this.memory.indexText(this.rawCorpus);
    this.ngrams = new Map();
    if (data.ngrams) {
      Object.keys(data.ngrams).forEach(key => this.ngrams.set(key, new Map(data.ngrams[key])));
    }
    this.isTrained = true;
  }
}
