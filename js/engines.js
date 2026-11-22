// The three trainable engines: Neural MLP, Tiny Attention LM and TF-IDF + N-Gram.

import { SemanticMemory } from './semantic-memory.js';
import { Tokenizer } from './tokenizer.js';
import { f32ToBase64, toFloat32 } from './binary-codec.js';

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
    const padId = this.tokenizer.vocab.get(this.tokenizer.PAD);
    const ctx = this.contextSize;

    const sampleStride = Number.isFinite(maxExamples) && maxExamples > 0 ? Math.max(1, Math.ceil(encoded.length / maxExamples)) : 1;
    // Flat typed arrays: X holds `count` rows of `contextSize` token ids, Y the target id of each row.
    const count = Math.ceil(encoded.length / sampleStride);
    const X = new Int32Array(count * ctx);
    const Y = new Int32Array(count);
    let row = 0;
    for (let i = 0; i < encoded.length; i += sampleStride) {
      for (let c = 0; c < ctx; c++) {
        const idx = i - (ctx - c);
        X[row * ctx + c] = idx >= 0 ? encoded[idx] : padId;
      }
      Y[row] = encoded[i];
      row++;
    }

    return { X, Y, count, totalTokens: tokens.length };
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

  // ---------------------------------------------------------------------------------------------
  // Mini-batch training (fast path). No allocations inside the hot loops, contiguous inner loops,
  // gradients are accumulated over the batch and the weights are updated once per batch.
  // The original one-example-at-a-time trainStep() is kept above as the reference implementation.
  // ---------------------------------------------------------------------------------------------

  _scratch(B) {
    const V = this.tokenizer.size();
    const D = this.embDim;
    const H = this.hiddenDim;
    const inD = this.contextSize * D;
    const s = this._scr;
    if (s && s.B >= B && s.V === V && s.D === D && s.H === H && s.inD === inD) return s;
    this._scr = {
      B, V, D, H, inD,
      xs: new Float64Array(B * inD),
      hs: new Float64Array(B * H),
      ps: new Float64Array(B * V),
      dHs: new Float64Array(B * H),
      dX: new Float64Array(inD),
      gE: new Float64Array(V * D),
      gW1: new Float64Array(inD * H),
      gb1: new Float64Array(H),
      gW2: new Float64Array(H * V),
      gb2: new Float64Array(V)
    };
    return this._scr;
  }

  // Writes the input vector of one example (row `off` of flat id array X) into xs at xo.
  _buildInput(X, off, xs, xo) {
    const ctx = this.contextSize;
    const D = this.embDim;
    const V = this.tokenizer.size();
    const E = this.E;
    for (let c = 0; c < ctx; c++) {
      let tid = X[off + c];
      if (tid < 0 || tid >= V) tid = 0;
      const src = tid * D;
      const dst = xo + c * D;
      for (let d = 0; d < D; d++) xs[dst + d] = E[src + d];
    }
  }

  // Forward pass for rows order[start..start+B) into scratch rows 0..B-1:
  // xs (inputs), hs (hidden activations) and ps (softmax probabilities).
  // The big output-layer loops are blocked over 4 examples so every W2 weight is read once per 4 examples.
  _fwdRows(X, order, start, B, scr) {
    const { xs, hs, ps, V, H, inD } = scr;
    const ctx = this.contextSize;
    const W1 = this.W1, b1 = this.b1, W2 = this.W2, b2 = this.b2;

    for (let b = 0; b < B; b++) {
      const xo = b * inD, ho = b * H;
      this._buildInput(X, order[start + b] * ctx, xs, xo);
      for (let j = 0; j < H; j++) hs[ho + j] = b1[j];
      for (let i = 0; i < inD; i++) {
        const xi = xs[xo + i];
        const row = i * H;
        for (let j = 0; j < H; j++) hs[ho + j] += xi * W1[row + j];
      }
      for (let j = 0; j < H; j++) hs[ho + j] = Math.tanh(hs[ho + j]);
      const po = b * V;
      for (let k = 0; k < V; k++) ps[po + k] = b2[k];
    }

    let b0 = 0;
    for (; b0 + 4 <= B; b0 += 4) {
      const o0 = b0 * V, o1 = o0 + V, o2 = o1 + V, o3 = o2 + V;
      const h0 = b0 * H, h1 = h0 + H, h2 = h1 + H, h3 = h2 + H;
      for (let j = 0; j < H; j++) {
        const a0 = hs[h0 + j], a1 = hs[h1 + j], a2 = hs[h2 + j], a3 = hs[h3 + j];
        const row = j * V;
        for (let k = 0; k < V; k++) {
          const w = W2[row + k];
          ps[o0 + k] += a0 * w;
          ps[o1 + k] += a1 * w;
          ps[o2 + k] += a2 * w;
          ps[o3 + k] += a3 * w;
        }
      }
    }
    for (; b0 < B; b0++) {
      const o = b0 * V, h = b0 * H;
      for (let j = 0; j < H; j++) {
        const a = hs[h + j];
        const row = j * V;
        for (let k = 0; k < V; k++) ps[o + k] += a * W2[row + k];
      }
    }

    for (let b = 0; b < B; b++) {
      const po = b * V;
      let max = -Infinity;
      for (let k = 0; k < V; k++) if (ps[po + k] > max) max = ps[po + k];
      let sum = 0;
      for (let k = 0; k < V; k++) {
        const e = Math.exp(ps[po + k] - max);
        ps[po + k] = e;
        sum += e;
      }
      const inv = 1 / sum;
      for (let k = 0; k < V; k++) ps[po + k] *= inv;
    }
  }

  // Average cross-entropy over order[from..to) without changing any weights.
  evalLoss(X, Y, order, from, to) {
    const n = to - from;
    if (n <= 0) return 0;
    const CH = 16;
    const scr = this._scratch(CH);
    const V = scr.V;
    let loss = 0;
    for (let s0 = from; s0 < to; s0 += CH) {
      const B = Math.min(CH, to - s0);
      this._fwdRows(X, order, s0, B, scr);
      for (let b = 0; b < B; b++) {
        const t = Y[order[s0 + b]];
        loss -= Math.log(Math.max(scr.ps[b * V + (t < V ? t : 0)], 1e-10));
      }
    }
    return loss / n;
  }

  // One SGD step on the mini-batch order[start..end). Gradients are averaged over the batch.
  // Returns the summed cross-entropy loss of the batch (before the update).
  trainBatch(X, Y, order, start, end, lr) {
    const B = end - start;
    if (B <= 0) return 0;
    const scr = this._scratch(B);
    const { xs, hs, ps, dHs, dX, gE, gW1, gb1, gW2, gb2, V, H, D, inD } = scr;
    const ctx = this.contextSize;
    const W1 = this.W1, W2 = this.W2, b1 = this.b1, b2 = this.b2, E = this.E;

    gE.fill(0); gW1.fill(0); gb1.fill(0); gW2.fill(0); gb2.fill(0);

    this._fwdRows(X, order, start, B, scr);

    let loss = 0;
    for (let b = 0; b < B; b++) {
      const po = b * V;
      const t = Y[order[start + b]];
      loss -= Math.log(Math.max(ps[po + t], 1e-10));
      ps[po + t] -= 1; // ps row now holds dLoss/dLogits
      for (let k = 0; k < V; k++) gb2[k] += ps[po + k];
    }

    // Output layer: gW2 += h^T d and dH = d W2^T, blocked over 4 examples.
    let b0 = 0;
    for (; b0 + 4 <= B; b0 += 4) {
      const o0 = b0 * V, o1 = o0 + V, o2 = o1 + V, o3 = o2 + V;
      const h0 = b0 * H, h1 = h0 + H, h2 = h1 + H, h3 = h2 + H;
      for (let j = 0; j < H; j++) {
        const a0 = hs[h0 + j], a1 = hs[h1 + j], a2 = hs[h2 + j], a3 = hs[h3 + j];
        const row = j * V;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let k = 0; k < V; k++) {
          const w = W2[row + k];
          const d0 = ps[o0 + k], d1 = ps[o1 + k], d2 = ps[o2 + k], d3 = ps[o3 + k];
          gW2[row + k] += a0 * d0 + a1 * d1 + a2 * d2 + a3 * d3;
          s0 += d0 * w; s1 += d1 * w; s2 += d2 * w; s3 += d3 * w;
        }
        dHs[h0 + j] = s0 * (1 - a0 * a0);
        dHs[h1 + j] = s1 * (1 - a1 * a1);
        dHs[h2 + j] = s2 * (1 - a2 * a2);
        dHs[h3 + j] = s3 * (1 - a3 * a3);
      }
    }
    for (; b0 < B; b0++) {
      const o = b0 * V, h = b0 * H;
      for (let j = 0; j < H; j++) {
        const a = hs[h + j];
        const row = j * V;
        let s = 0;
        for (let k = 0; k < V; k++) {
          const d = ps[o + k];
          gW2[row + k] += a * d;
          s += d * W2[row + k];
        }
        dHs[h + j] = s * (1 - a * a);
      }
    }

    // Hidden layer and embeddings (small compared with the output layer).
    for (let b = 0; b < B; b++) {
      const i = order[start + b];
      const ho = b * H, xo = b * inD;
      for (let j = 0; j < H; j++) gb1[j] += dHs[ho + j];
      for (let ii = 0; ii < inD; ii++) {
        const xi = xs[xo + ii];
        const row = ii * H;
        let s = 0;
        for (let j = 0; j < H; j++) {
          const dj = dHs[ho + j];
          gW1[row + j] += xi * dj;
          s += dj * W1[row + j];
        }
        dX[ii] = s;
      }
      for (let c = 0; c < ctx; c++) {
        const tid = X[i * ctx + c];
        if (tid >= 0 && tid < V) {
          const eRow = tid * D, xRow = c * D;
          for (let d = 0; d < D; d++) gE[eRow + d] += dX[xRow + d];
        }
      }
    }

    const scale = lr / B;
    for (let k = 0; k < W2.length; k++) W2[k] -= scale * gW2[k];
    for (let k = 0; k < b2.length; k++) b2[k] -= scale * gb2[k];
    for (let k = 0; k < W1.length; k++) W1[k] -= scale * gW1[k];
    for (let k = 0; k < b1.length; k++) b1[k] -= scale * gb1[k];
    for (let k = 0; k < E.length; k++) E[k] -= scale * gE[k];
    return loss;
  }

  // Everything a trainer needs to hand back (e.g. from a Web Worker) so another instance can adopt it.
  exportTrainedState() {
    return {
      contextSize: this.contextSize,
      hiddenDim: this.hiddenDim,
      embDim: this.embDim,
      vocab: this.tokenizer.idToToken.slice(),
      E: this.E, W1: this.W1, b1: this.b1, W2: this.W2, b2: this.b2
    };
  }

  applyTrainedState(state) {
    this.contextSize = state.contextSize;
    this.hiddenDim = state.hiddenDim;
    this.embDim = state.embDim;
    this.tokenizer = new Tokenizer(false);
    (state.vocab || []).forEach(t => this.tokenizer.addToken(t));
    this.E = toFloat32(state.E);
    this.W1 = toFloat32(state.W1);
    this.b1 = toFloat32(state.b1);
    this.W2 = toFloat32(state.W2);
    this.b2 = toFloat32(state.b2);
    this._scr = null;
    this.isTrained = true;
  }

  generate(prompt, maxTokens = 45, temperature = 0.35, repetitionPenalty = 1.35, topK = 5) {
    if (!this.E || !this.E.length) {
      const early = this.memory.queryBestMatch(prompt);
      return early ? early.text : "Model has not been trained yet. Open Train Studio or Corpus Library to train me!";
    }

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

  // opts.weights: 'array' (default, plain JSON numbers), 'f32' (Float32Array copies, for IndexedDB and undo
  // snapshots) or 'base64' (compact text for .json exports).
  toJSON(opts) {
    const mode = (opts && typeof opts === 'object' && opts.weights) || 'array';
    const pack = arr => {
      const f = arr || new Float32Array(0);
      return mode === 'base64' ? f32ToBase64(f) : (mode === 'f32' ? f.slice() : Array.from(f));
    };
    const data = {
      type: 'neural',
      contextSize: this.contextSize,
      embDim: this.embDim,
      hiddenDim: this.hiddenDim,
      vocab: this.tokenizer.idToToken,
      lossHistory: this.lossHistory,
      epochsTrained: this.epochsTrained,
      rawCorpus: this.rawCorpus,
      E: pack(this.E),
      W1: pack(this.W1),
      b1: pack(this.b1),
      W2: pack(this.W2),
      b2: pack(this.b2)
    };
    if (mode === 'base64') data.weightsEncoding = 'f32-base64';
    return data;
  }

  // Accepts weights as plain arrays (old saves), Float32Arrays (binary saves) or base64 text (compact exports).
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

    const hasWeights = Boolean(data.E && data.E.length);
    this.E = hasWeights ? toFloat32(data.E) : null;
    this.W1 = hasWeights ? toFloat32(data.W1) : null;
    this.b1 = hasWeights ? toFloat32(data.b1) : null;
    this.W2 = hasWeights ? toFloat32(data.W2) : null;
    this.b2 = hasWeights ? toFloat32(data.b2) : null;
    this._scr = null;
    this.isTrained = hasWeights;
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

  // Same attention mix as forward(), written into a flat scratch buffer for the batched trainer.
  _buildInput(X, off, xs, xo) {
    const ctx = this.contextSize;
    const D = this.embDim;
    const V = this.tokenizer.size();
    const E = this.E;
    const scores = this._attScores && this._attScores.length === ctx ? this._attScores : (this._attScores = new Float64Array(ctx));
    let q = X[off + ctx - 1];
    if (q < 0 || q >= V) q = 0;
    const qRow = q * D;
    const norm = Math.sqrt(Math.max(1, D));
    let max = -Infinity;
    for (let c = 0; c < ctx; c++) {
      let tid = X[off + c];
      if (tid < 0 || tid >= V) tid = 0;
      const row = tid * D;
      let dot = 0;
      for (let d = 0; d < D; d++) dot += E[qRow + d] * E[row + d];
      scores[c] = dot / norm;
      if (scores[c] > max) max = scores[c];
    }
    let sum = 0;
    for (let c = 0; c < ctx; c++) { scores[c] = Math.exp(scores[c] - max); sum += scores[c]; }
    sum = sum || 1;

    const qo = xo + (ctx - 1) * D;
    for (let d = 0; d < D; d++) xs[qo + d] = 0;
    for (let c = 0; c < ctx; c++) {
      let tid = X[off + c];
      if (tid < 0 || tid >= V) tid = 0;
      const row = tid * D;
      const w = scores[c] / sum;
      for (let d = 0; d < D; d++) xs[qo + d] += w * E[row + d];
      if (c < ctx - 1) {
        const dst = xo + c * D;
        for (let d = 0; d < D; d++) xs[dst + d] = E[row + d];
      }
    }
  }

  toJSON(opts) {
    const d = super.toJSON(opts);
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
