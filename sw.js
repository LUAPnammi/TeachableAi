// Teachable AI Studio service worker.
// Bump CACHE_VERSION every time you upload a new version, so phones drop the old copy.
const CACHE_VERSION = 'v0.2.1-1';
const CACHE_NAME = 'teachable-ai-' + CACHE_VERSION;
const APP_SHELL = [
  './', './index.html', './styles.css', './manifest.webmanifest', './icon.svg',
  './js/main.js', './js/app.js', './js/app-models.js', './js/app-storage.js', './js/app-chat.js',
  './js/app-training.js', './js/app-memory.js', './js/app-eval.js', './js/app-assistant.js',
  './js/app-qa-tools.js', './js/app-screen.js', './js/engines.js', './js/tokenizer.js',
  './js/semantic-memory.js', './js/doc-search.js', './js/corpus-library.js', './js/studio-notes.js',
  './js/binary-codec.js', './js/trainer.js', './js/train-worker.js', './js/app-backup.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => Promise.all(APP_SHELL.map(url => cache.add(url).catch(() => {}))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k.startsWith('teachable-ai-') && k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Network first for this site's own files (so updates show up), cache as the offline fallback.
// Everything on other domains (CDNs, model downloads) is left to the browser.
self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith(
    fetch(req)
      .then(res => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE_NAME).then(c => c.put(req, copy)); }
        return res;
      })
      .catch(() => caches.match(req).then(hit => hit || caches.match('./index.html')))
  );
});
