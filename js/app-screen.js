// Read my screen: screen capture and screenshot OCR, plus turning the text into training data.
// Methods here are copied onto AIApp in main.js, so `this` is the app.

export class AppScreen {
  /* ============ Screen reading (capture + OCR) ============ */
  ensureTesseract() {
    if (window.Tesseract) return Promise.resolve(window.Tesseract);
    if (this._tesseractPromise) return this._tesseractPromise;
    this._tesseractPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js';
      s.onload = () => window.Tesseract ? resolve(window.Tesseract) : reject(new Error('the OCR library loaded but is unavailable'));
      s.onerror = () => { this._tesseractPromise = null; reject(new Error('the OCR library could not be downloaded (are you offline?)')); };
      document.head.appendChild(s);
    });
    return this._tesseractPromise;
  }

  async ocrFromURL(source, onProgress) {
    const T = await this.ensureTesseract();
    const res = await T.recognize(source, 'eng', { logger: m => { if (onProgress && m && m.status && m.progress != null) onProgress(m.status, m.progress); } });
    return String(res?.data?.text || '').replace(/\n{3,}/g, '\n\n').trim();
  }

  async captureScreenCanvas() {
    if (!window.isSecureContext) throw new Error('screen capture needs a secure page (https:// or localhost). Open the app from localhost, or take a screenshot and use the image button instead.');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) throw new Error('this browser cannot capture the screen. Take a screenshot and use the image button instead.');
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
    try {
      const video = document.createElement('video');
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      await new Promise(r => setTimeout(r, 800));
      const w = video.videoWidth || 1280, h = video.videoHeight || 720;
      const scale = Math.min(1, 1600 / w);
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(w * scale));
      canvas.height = Math.max(1, Math.round(h * scale));
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/png');
    } finally {
      stream.getTracks().forEach(t => t.stop());
    }
  }

  async runScreenRead() {
    const P = this.panel;
    if (!P || P.busy) { this.showToast('Stop the current reply first.', 'error'); return; }
    P.messages.push({ role: 'user', text: 'Read my screen' });
    this.panelRender();
    const live = this.panelStreamBubble();
    try {
      live.setText('Waiting for screen share… choose a window or tab.');
      const image = await this.captureScreenCanvas();
      live.setText('Reading pixels with OCR… first run downloads the reader once.');
      const text = await this.ocrFromURL(image, (st, pr) => live.setText('OCR: ' + st + ' ' + Math.round(pr * 100) + '%'));
      live.wrapper.remove();
      this.lastScreenText = text;
      P.messages.push({ role: 'ai', text: this.buildScreenReply(text, 'your screen'), stats: { source: 'screen-ocr', ms: 0, tokens: 0, tps: 0 } });
    } catch (err) {
      live.wrapper.remove();
      P.messages.push({ role: 'ai', text: 'Screen read failed: ' + (err.message || err) + ' You can also take a screenshot and press the image button.', stats: null });
    }
    this.persistPanel();
    this.panelRender();
  }

  async ocrScreenshotFile(e) {
    const file = e && e.target && e.target.files ? e.target.files[0] : null;
    if (!file) return;
    if (e.target) e.target.value = '';
    const P = this.panel;
    if (!P || P.busy) { this.showToast('Stop the current reply first.', 'error'); return; }
    P.messages.push({ role: 'user', text: 'Read this screenshot' });
    this.panelRender();
    const live = this.panelStreamBubble();
    try {
      live.setText('Reading image with OCR… first run downloads the reader once.');
      const url = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = () => rej(new Error('could not read the image file')); r.readAsDataURL(file); });
      const text = await this.ocrFromURL(url, (st, pr) => live.setText('OCR: ' + st + ' ' + Math.round(pr * 100) + '%'));
      live.wrapper.remove();
      this.lastScreenText = text;
      P.messages.push({ role: 'ai', text: this.buildScreenReply(text, 'your screenshot'), stats: { source: 'screenshot-ocr', ms: 0, tokens: 0, tps: 0 } });
    } catch (err) {
      live.wrapper.remove();
      P.messages.push({ role: 'ai', text: 'Screenshot read failed: ' + (err.message || err), stats: null });
    }
    this.persistPanel();
    this.panelRender();
  }

  buildScreenReply(text, sourceLabel) {
    const chars = String(text || '').length;
    if (!chars) return 'I read ' + sourceLabel + ' but could not find readable text. Try a screenshot at a larger zoom, or heavier-contrast text.';
    const words = String(text).split(/\s+/).filter(Boolean).length;
    const preview = String(text).trim().slice(0, 400);
    return 'I read ' + chars.toLocaleString() + ' characters (' + words + ' words) from ' + sourceLabel + '. Preview:\n\n' + preview + (chars > 400 ? '…' : '') + '\n\nNext steps: say "turn this into training data" to convert it into user:/ai: pairs, "add this to my corpus" to store the raw text, or just ask me anything about it.';
  }

  materialToTraining(raw, label) {
    const text = String(raw || '').trim().slice(0, 30000);
    if (!text) { this.showToast('Read a screen or screenshot first.', 'error'); return 'I have not read anything yet. Press "Read my screen" or the image button first.'; }
    const paras = this.parseParagraphs(text);
    const output = [];
    for (const para of paras) {
      for (const pair of this.qaFromParagraph(para, 3)) output.push('user: ' + pair.q + '\nai: ' + pair.a);
    }
    const area = document.getElementById('training-text');
    if (!area) return 'Train Studio not found.';
    if (output.length) {
      area.value = this.mergeCorpus(area.value, output.join('\n\n'), false);
      this.updateCorpusStats();
      this.switchTab('train-tab');
      this.showToast('Added ' + output.length + ' Q&A pairs from ' + label + '.');
      return 'Converted ' + paras.length + ' paragraph(s) from ' + label + ' into ' + output.length + ' new user:/ai: pairs and appended them to Train Studio. Review them, then say "train it" (or press Merge & Train).';
    }
    area.value = this.mergeCorpus(area.value, text, false);
    this.updateCorpusStats();
    this.switchTab('train-tab');
    this.showToast('Appended raw ' + label + ' to the corpus.');
    return 'The text had no clear paragraphs, so I appended it to the corpus as raw knowledge. Run the Q&A Generator in Train Studio to shape it into pairs, then say "train it".';
  }

  appendRawToCorpus(raw, label) {
    const text = String(raw || '').trim().slice(0, 50000);
    if (!text) { this.showToast('Read a screen or screenshot first.', 'error'); return 'I have not read anything yet. Press "Read my screen" or the image button first.'; }
    const area = document.getElementById('training-text');
    area.value = this.mergeCorpus(area.value, text, false);
    this.updateCorpusStats();
    this.switchTab('train-tab');
    this.showToast('Appended ' + text.length.toLocaleString() + ' characters to the corpus.');
    return 'Stored the raw text from ' + label + ' into the training corpus. Review it in Train Studio, then say "train it".';
  }
}
