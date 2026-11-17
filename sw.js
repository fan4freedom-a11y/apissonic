// sw.js — offline cache so ApisSonic keeps working at apiaries with no signal.
// Bump CACHE whenever you deploy changed files.
const CACHE = 'apissonic-v1';
const ASSETS = [
  './',
  'index.html',
  'styles.css',
  'app.js',
  'dsp-engine.js',
  'ml-classifier.js',
  'calibration.js',
  'audio-store.js',
  'qr-report.js',
  'manifest.json',
  'test-suite.html',
  'tests/dsp-ml-tests.js',
  'ml/custom_mlp_weights.json',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
  'icons/apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Stale-while-revalidate: answer from cache at once, refresh the cache in the background.
// QR deep-links (?hive=...) resolve to the same cached page via ignoreSearch.
self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith(
    caches.open(CACHE).then(async (cache) => {
      const cached = await cache.match(req, { ignoreSearch: true });
      const network = fetch(req)
        .then((res) => {
          if (res && res.ok) cache.put(req.url.split('?')[0], res.clone());
          return res;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
