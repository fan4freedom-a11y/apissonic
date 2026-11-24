// audio-store.js — ApisSonic Phase 2
// 16-bit mono PCM .wav encoder + IndexedDB wrapper (inspections + wav_clips + scan_images).

const DB_NAME = 'apissonic';
const DB_VERSION = 2; // v2 adds scan_images (spectrum / waterfall / MFCC PNGs per scan)
export const STORE_INSPECTIONS = 'inspections';
export const STORE_CLIPS = 'wav_clips';
export const STORE_IMAGES = 'scan_images';

function flatten(chunks) {
  if (chunks instanceof Float32Array) return chunks;
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

/**
 * Downsample 2× (pair-averaged) and encode as 16-bit mono PCM WAV.
 * 44100 Hz in → 22050 Hz out; on a 48 kHz iPhone the header correctly says 24000 Hz.
 * Returns an ArrayBuffer (44-byte RIFF header + little-endian Int16 samples).
 */
export function encodeMonoWAVBuffer(chunks, inputSampleRate = 44100) {
  const input = flatten(chunks);
  const numSamples = Math.floor(input.length / 2);
  const outRate = Math.round(inputSampleRate / 2);
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const v = new DataView(buf);
  const ascii = (off, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  ascii(0, 'RIFF');
  v.setUint32(4, 36 + numSamples * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, outRate, true);
  v.setUint32(28, outRate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  v.setUint32(40, numSamples * 2, true);
  for (let i = 0; i < numSamples; i++) {
    let s = 0.5 * (input[2 * i] + input[2 * i + 1]);
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    v.setInt16(44 + i * 2, Math.round(s < 0 ? s * 32768 : s * 32767), true);
  }
  return buf;
}

export function encodeMonoWAV(chunks, inputSampleRate = 44100) {
  return new Blob([encodeMonoWAVBuffer(chunks, inputSampleRate)], { type: 'audio/wav' });
}

// ------------------------------------------------------------------ IndexedDB

let dbPromise = null;

function openDB() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_INSPECTIONS)) db.createObjectStore(STORE_INSPECTIONS, { keyPath: 'id' });
        if (!db.objectStoreNames.contains(STORE_CLIPS)) db.createObjectStore(STORE_CLIPS, { keyPath: 'id' });
        if (!db.objectStoreNames.contains(STORE_IMAGES)) db.createObjectStore(STORE_IMAGES, { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function run(store, mode, fn) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, mode);
    const req = fn(tx.objectStore(store));
    tx.oncomplete = () => resolve(req ? req.result : undefined);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export const saveInspection = (rec) => run(STORE_INSPECTIONS, 'readwrite', (s) => s.put(rec));
export const listInspections = async () => {
  const all = (await run(STORE_INSPECTIONS, 'readonly', (s) => s.getAll())) || [];
  return all.sort((a, b) => b.id - a.id);
};
export const saveClip = (id, blob) => run(STORE_CLIPS, 'readwrite', (s) => s.put({ id, blob }));
export const getClip = async (id) => {
  const rec = await run(STORE_CLIPS, 'readonly', (s) => s.get(id));
  return rec ? rec.blob : null;
};
/** images: { spectrum, waterfall, mfcc } PNG blobs for inspection `id`. */
export const saveImages = (id, images) => run(STORE_IMAGES, 'readwrite', (s) => s.put({ id, ...images }));
export const getImages = async (id) => {
  const rec = await run(STORE_IMAGES, 'readonly', (s) => s.get(id));
  if (!rec) return null;
  const { id: _id, ...images } = rec;
  return images;
};
export async function deleteInspection(id) {
  await run(STORE_INSPECTIONS, 'readwrite', (s) => s.delete(id));
  await run(STORE_CLIPS, 'readwrite', (s) => s.delete(id));
  await run(STORE_IMAGES, 'readwrite', (s) => s.delete(id));
}
export async function clearAll() {
  await run(STORE_INSPECTIONS, 'readwrite', (s) => s.clear());
  await run(STORE_CLIPS, 'readwrite', (s) => s.clear());
  await run(STORE_IMAGES, 'readwrite', (s) => s.clear());
}

/** Ask the browser not to evict our data (matters on iOS for rarely-opened apps). */
export async function requestPersistence() {
  try {
    if (navigator.storage && navigator.storage.persist) return await navigator.storage.persist();
  } catch (e) { /* unsupported */ }
  return false;
}
