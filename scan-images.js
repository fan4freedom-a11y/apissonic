// scan-images.js — ApisSonic Phase 5 add-on
// Records every analysis frame of a 15 s scan, then renders three PNG images that are
// saved with the inspection and shown in Yard triage:
//   • spectrum  — mean calibrated spectrum of the clean frames, with the diagnostic bands
//   • waterfall — spectrogram of the whole scan (frequency →, time ↓), rejected frames flagged
//   • mfcc      — cepstrogram of the 13 MFCCs across the scan (time →, C0 at the bottom)

export const IMAGE_KINDS = ['spectrum', 'waterfall', 'mfcc'];
export const IMAGE_NAMES = { spectrum: 'Spectrum', waterfall: 'Waterfall', mfcc: 'MFCC' };
export const IMAGE_MAX_HZ = 1200;

// Saved images use a fixed dark style so they look the same whichever theme was active.
const STYLE = {
  bg: '#14110b', plot: '#0e0c08', text: '#f3ecdc', muted: '#a89f8a', line: '#3a3324', accent: '#f2b01e',
  healthy: '#5cc47a', fanning: '#58a6e8', piping: '#e8c14a', preswarm: '#f08a3c', queenless: '#ef5a5a',
};
const BAND_SHADES = [
  ['fanning', 120, 200], ['healthy', 200, 290], ['preswarm', 300, 420], ['piping', 410, 540], ['queenless', 550, 1000],
];

export function heat(v) {
  // 0..1 → dark → violet → orange → amber → pale (same ramp as the live canvases)
  v = Math.min(1, Math.max(0, v));
  const stops = [[10, 8, 20], [70, 30, 90], [200, 80, 40], [245, 190, 50], [255, 250, 220]];
  const f = v * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), t = f - i;
  return stops[i].map((c, j) => Math.round(c + t * (stops[i + 1][j] - c)));
}

/** Collects per-frame spectra (0–IMAGE_MAX_HZ) and MFCCs while a scan runs. */
export class ScanImageRecorder {
  constructor() {
    this.reset();
  }

  reset() {
    this.frames = [];
    this.binHz = 0;
  }

  /** db: calibrated dB spectrum (Float32Array, N/2 bins); mfcc: 13 coefficients; clean: usable frame. */
  add(db, mfcc, sampleRate, clean) {
    const binHz = sampleRate / (db.length * 2);
    const kMax = Math.min(db.length - 1, Math.ceil(IMAGE_MAX_HZ / binHz));
    this.binHz = binHz;
    this.frames.push({ spec: Float32Array.from(db.subarray(0, kMax + 1)), mfcc: Float32Array.from(mfcc), clean: !!clean });
  }

  get length() {
    return this.frames.length;
  }
}

/**
 * Pure summary of recorded frames (no canvas): mean clean spectrum in dB (power-averaged),
 * its peak, and the dB range used to colour the waterfall.
 */
export function summarizeScanFrames(frames, binHz) {
  if (!frames.length) return null;
  const nBins = frames[0].spec.length;
  const use = frames.some((f) => f.clean) ? frames.filter((f) => f.clean) : frames;
  const lin = new Float64Array(nBins);
  for (const f of use) for (let k = 0; k < nBins; k++) lin[k] += Math.pow(10, f.spec[k] / 10);
  const meanDb = new Float32Array(nBins);
  let peakK = 3, top = -Infinity;
  for (let k = 0; k < nBins; k++) {
    meanDb[k] = 10 * Math.log10(lin[k] / use.length + 1e-20);
    if (k >= 3 && meanDb[k] > meanDb[peakK]) peakK = k;
  }
  for (const f of frames) for (let k = 3; k < nBins; k++) if (Number.isFinite(f.spec[k]) && f.spec[k] > top) top = f.spec[k];
  if (!Number.isFinite(top)) top = -30;
  return { meanDb, peakHz: peakK * binHz, topDb: Math.ceil((top + 3) / 10) * 10, usedFrames: use.length };
}

// ------------------------------------------------------------------ rendering

function makeCanvas(w, h) {
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const g = cv.getContext('2d');
  g.fillStyle = STYLE.bg;
  g.fillRect(0, 0, w, h);
  return { cv, g };
}

function header(g, W, title, meta) {
  g.fillStyle = STYLE.text;
  g.font = '600 22px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
  g.textBaseline = 'top';
  g.textAlign = 'left';
  g.fillText(title, 20, 14);
  g.fillStyle = STYLE.muted;
  g.font = '15px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
  g.fillText(meta, 20, 44);
  g.textAlign = 'right';
  g.fillText('ApisSonic', W - 20, 18);
}

const toBlob = (cv) => new Promise((resolve, reject) => {
  cv.toBlob((b) => (b ? resolve(b) : reject(new Error('Image encoding failed'))), 'image/png');
});

function renderSpectrum(rec, sum, meta) {
  const W = 960, H = 440;
  const { cv, g } = makeCanvas(W, H);
  header(g, W, `Spectrum — ${meta.hive}`, `${meta.when} · ${meta.state} · mean of ${sum.usedFrames} clean frames · peak ${sum.peakHz.toFixed(0)} Hz`);
  const P = { l: 64, r: 20, t: 78, b: 46 };
  const pw = W - P.l - P.r, ph = H - P.t - P.b;
  g.fillStyle = STYLE.plot;
  g.fillRect(P.l, P.t, pw, ph);
  const range = 70, top = sum.topDb;
  const xOf = (hz) => P.l + (hz / IMAGE_MAX_HZ) * pw;
  const yOf = (d) => P.t + Math.min(ph, Math.max(0, ((top - d) / range) * ph));
  g.font = '13px -apple-system, "Segoe UI", Roboto, sans-serif';
  for (const [name, lo, hi] of BAND_SHADES) {
    g.globalAlpha = 0.16;
    g.fillStyle = STYLE[name];
    g.fillRect(xOf(lo), P.t, xOf(hi) - xOf(lo), ph);
  }
  g.globalAlpha = 1;
  g.fillStyle = STYLE.line;
  for (let d = top; d >= top - range; d -= 10) g.fillRect(P.l, yOf(d), pw, 1);
  g.fillStyle = STYLE.muted;
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  for (let d = top; d >= top - range; d -= 10) g.fillText(`${d} dB`, P.l - 6, yOf(d));
  g.textAlign = 'center';
  g.textBaseline = 'top';
  for (let hz = 0; hz <= IMAGE_MAX_HZ; hz += 200) {
    g.fillRect(xOf(hz), P.t + ph, 1, 5);
    g.fillText(`${hz} Hz`, Math.min(W - 30, Math.max(P.l + 16, xOf(hz))), P.t + ph + 8);
  }
  g.beginPath();
  for (let k = 1; k < sum.meanDb.length; k++) {
    const x = xOf(k * rec.binHz), y = yOf(sum.meanDb[k]);
    if (k === 1) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.strokeStyle = STYLE.accent;
  g.lineWidth = 2;
  g.lineJoin = 'round';
  g.stroke();
  if (meta.meanHz > 0 && meta.meanHz < IMAGE_MAX_HZ) {
    g.fillStyle = STYLE.text;
    g.fillRect(xOf(meta.meanHz), P.t, 1, ph);
    g.textAlign = meta.meanHz > IMAGE_MAX_HZ * 0.75 ? 'right' : 'left';
    g.fillText(` mean ${meta.meanHz.toFixed(0)} Hz `, xOf(meta.meanHz), P.t + 6);
  }
  // Legend
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  let lx = P.l;
  for (const [name] of BAND_SHADES) {
    g.fillStyle = STYLE[name];
    g.fillRect(lx, H - 14, 10, 10);
    g.fillStyle = STYLE.muted;
    const label = meta.labelNames[name] || name;
    g.fillText(label, lx + 14, H - 9);
    lx += 22 + g.measureText(label).width;
  }
  return cv;
}

function renderWaterfall(rec, sum, meta) {
  const frames = rec.frames;
  const W = 960, P = { l: 64, r: 20, t: 78, b: 40 };
  const rowsH = Math.max(150, Math.min(600, frames.length * 2));
  const H = P.t + rowsH + P.b;
  const { cv, g } = makeCanvas(W, H);
  const rejected = frames.filter((f) => !f.clean).length;
  header(g, W, `Waterfall — ${meta.hive}`, `${meta.when} · ${meta.state} · ${meta.seconds.toFixed(1)} s · ${frames.length} frames (${rejected} rejected, marked red)`);
  const pw = W - P.l - P.r;
  const img = g.createImageData(pw, rowsH);
  const lo = sum.topDb - 70;
  for (let y = 0; y < rowsH; y++) {
    const f = frames[Math.min(frames.length - 1, Math.floor((y / rowsH) * frames.length))];
    for (let x = 0; x < pw; x++) {
      const k = Math.min(f.spec.length - 1, Math.round(((x / pw) * IMAGE_MAX_HZ) / rec.binHz));
      const [r, gg, b] = heat((f.spec[k] - lo) / 70);
      const o = (y * pw + x) * 4;
      img.data[o] = r;
      img.data[o + 1] = gg;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, P.l, P.t);
  // Rejected-frame markers in the left gutter
  g.fillStyle = STYLE.queenless;
  frames.forEach((f, i) => {
    if (!f.clean) g.fillRect(P.l - 8, P.t + (i / frames.length) * rowsH, 5, Math.max(1, rowsH / frames.length));
  });
  g.font = '13px -apple-system, "Segoe UI", Roboto, sans-serif';
  g.fillStyle = STYLE.muted;
  g.textAlign = 'center';
  g.textBaseline = 'top';
  const xOf = (hz) => P.l + (hz / IMAGE_MAX_HZ) * pw;
  for (let hz = 0; hz <= IMAGE_MAX_HZ; hz += 200) {
    g.fillRect(xOf(hz), P.t + rowsH, 1, 5);
    g.fillText(`${hz} Hz`, Math.min(W - 30, Math.max(P.l + 16, xOf(hz))), P.t + rowsH + 8);
  }
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  const step = meta.seconds > 10 ? 5 : meta.seconds > 4 ? 2 : 1;
  for (let s = 0; s <= meta.seconds + 1e-6; s += step) g.fillText(`${s} s`, P.l - 12, P.t + (s / meta.seconds) * rowsH);
  return cv;
}

function renderMfcc(rec, meta) {
  const frames = rec.frames;
  const W = 960, P = { l: 64, r: 20, t: 78, b: 40 };
  const nC = frames[0].mfcc.length;
  const rowH = 26, ph = nC * rowH, H = P.t + ph + P.b;
  const { cv, g } = makeCanvas(W, H);
  header(g, W, `MFCC — ${meta.hive}`, `${meta.when} · ${meta.state} · C0–C${nC - 1} over ${meta.seconds.toFixed(1)} s (60–2000 Hz mel bank)`);
  const pw = W - P.l - P.r;
  const colW = pw / frames.length;
  frames.forEach((f, i) => {
    for (let c = 0; c < nC; c++) {
      // C0 is overall level (large, negative); C1.. are spectral shape.
      const v = c === 0 ? (f.mfcc[0] + 500) / 400 : 0.5 + f.mfcc[c] / 60;
      const [r, gg, b] = heat(v);
      g.fillStyle = `rgb(${r},${gg},${b})`;
      g.fillRect(P.l + Math.floor(i * colW), P.t + (nC - 1 - c) * rowH, Math.ceil(colW) + 1, rowH);
    }
  });
  g.fillStyle = STYLE.queenless;
  frames.forEach((f, i) => {
    if (!f.clean) g.fillRect(P.l + i * colW, P.t + ph + 2, Math.max(1, colW), 4);
  });
  g.font = '13px -apple-system, "Segoe UI", Roboto, sans-serif';
  g.fillStyle = STYLE.muted;
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  for (let c = 0; c < nC; c++) g.fillText(`C${c}`, P.l - 8, P.t + (nC - 1 - c) * rowH + rowH / 2);
  g.textAlign = 'center';
  g.textBaseline = 'top';
  const step = meta.seconds > 10 ? 5 : meta.seconds > 4 ? 2 : 1;
  for (let s = 0; s <= meta.seconds + 1e-6; s += step) {
    const x = P.l + (s / meta.seconds) * pw;
    g.fillText(`${s} s`, Math.min(W - 24, Math.max(P.l + 12, x)), P.t + ph + 12);
  }
  return cv;
}

/**
 * Render the three scan images as PNG blobs.
 * meta: { hive, when, state, meanHz, seconds, labelNames }
 */
export async function renderScanImages(rec, meta) {
  if (!rec.frames.length || typeof document === 'undefined') return null;
  const sum = summarizeScanFrames(rec.frames, rec.binHz);
  const [spectrum, waterfall, mfcc] = await Promise.all([
    toBlob(renderSpectrum(rec, sum, meta)),
    toBlob(renderWaterfall(rec, sum, meta)),
    toBlob(renderMfcc(rec, meta)),
  ]);
  return { spectrum, waterfall, mfcc };
}
