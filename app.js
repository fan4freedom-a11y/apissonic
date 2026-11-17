// app.js — ApisSonic UI controller: wiring, canvas renderers, log / CSV / WAV / QR actions.

import { DSPEngine, ScanAccumulator, computeAllBands, BANDS, SCAN_SECONDS, SYNTH_STATES, FFT_SIZE } from './dsp-engine.js';
import {
  FeatureExtractor, mlpForward, bayesBandRatio, fuseEnsemble, validateWeights, argmax,
  DEFAULT_WEIGHTS, LABELS, LABEL_NAMES,
} from './ml-classifier.js';
import {
  MATERIALS, MicFloor, subtractFloor, applyMaterialToSpectrum, applyMaterialToBands,
  environmentPriors, autoTimeOfDay,
} from './calibration.js';
import {
  encodeMonoWAV, saveInspection, listInspections, saveClip, getClip, deleteInspection, clearAll, requestPersistence,
} from './audio-store.js';
import { parseDeepLink, buildDeepLink, generateQRMatrix, qrToSVG, buildTriage, buildReportHTML } from './qr-report.js';

const $ = (id) => document.getElementById(id);
const CFG_KEY = 'apissonic.cfg';
const WEIGHTS_KEY = 'apissonic.weights';
const THEME_KEY = 'apissonic.theme';
const MIN_CLEAN_FRAMES = 40;
const PLOT_MAX_HZ = 1200;

const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch (e) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) { /* private mode / quota: keep working in memory */ }
  },
  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch (e) { /* ignore */ }
  },
};

// ------------------------------------------------------------------ state

const cfg = Object.assign({ hive: 'Hive-01', material: 'landing_board', temp: 22, tod: 'auto' }, store.get(CFG_KEY, {}));
const link = parseDeepLink(location.search);
if (link.hive) cfg.hive = link.hive;
if (link.material && MATERIALS[link.material]) cfg.material = link.material;
if (link.temp !== undefined) cfg.temp = link.temp;

let weights = DEFAULT_WEIGHTS;
let customWeights = false;
try {
  const saved = store.get(WEIGHTS_KEY, null);
  if (saved) {
    weights = validateWeights(saved);
    customWeights = true;
  }
} catch (e) {
  store.remove(WEIGHTS_KEY);
}

let source = 'mic';
let inspections = [];
let smoothed = null; // EMA of fused probabilities (frozen during artifacts)
let floorUntil = 0;
let lastFrameInfo = null;
const extractor = new FeatureExtractor();
const micFloor = new MicFloor();
const scan = new ScanAccumulator(SCAN_SECONDS * 1000);
const scratchFloor = new Float32Array(FFT_SIZE / 2);
const scratchCal = new Float32Array(FFT_SIZE / 2);

const engine = new DSPEngine({
  onFrame: handleFrame,
  onEnded: () => {
    if (scan.active) finishScan();
    setRunning(false);
    toast('File finished');
  },
});

// ------------------------------------------------------------------ helpers

let toastTimer = null;
function toast(msg, ms = 3200) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let palette = {};
function refreshPalette() {
  for (const k of ['text', 'muted', 'line', 'accent', 'plot-bg', ...LABELS]) palette[k] = cssVar('--' + k);
}
const stateColor = (s) => palette[s] || palette.muted;
const timeOfDay = () => (cfg.tod === 'auto' ? autoTimeOfDay() : cfg.tod);
const fmtDate = (t) => new Date(t).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

// ------------------------------------------------------------------ frame pipeline

function handleFrame({ db, time, sampleRate, artifact }) {
  const now = performance.now();
  if (floorUntil) {
    micFloor.add(db);
    if (now >= floorUntil) {
      floorUntil = 0;
      toast(micFloor.finish() ? 'Mic floor calibrated' : 'Calibration failed');
      updateToolsMeta();
    }
  }
  // Phase 4: floor subtraction, then wall-material compensation.
  const dbFloor = subtractFloor(db, micFloor.floor, scratchFloor);
  const dbCal = applyMaterialToSpectrum(dbFloor, sampleRate, cfg.material, scratchCal);
  // Phase 3: features → MLP; band ratios → Bayes; 60/40 fusion.
  const feat = extractor.extract(dbCal, time, sampleRate);
  const mlp = mlpForward(feat.x, weights);
  const bands = applyMaterialToBands(computeAllBands(dbFloor, sampleRate), cfg.material);
  const bayes = bayesBandRatio(bands, environmentPriors(LABELS, cfg.temp, timeOfDay()));
  const fused = fuseEnsemble(mlp, bayes);

  const usable = !artifact.isArtifact && !artifact.quiet;
  if (usable) {
    if (!smoothed) smoothed = fused.slice();
    else for (let i = 0; i < fused.length; i++) smoothed[i] += 0.15 * (fused[i] - smoothed[i]);
  }
  if (scan.active) {
    const progress = scan.add({ isArtifact: artifact.isArtifact, quiet: artifact.quiet, dominantHz: feat.dominantHz, probs: fused }, now);
    $('scan-bar').style.width = Math.min(100, progress * 100) + '%';
    $('scan-label').textContent = `Scanning… ${Math.max(0, Math.ceil(SCAN_SECONDS * (1 - progress)))} s · ${scan.cleanFrames} clean frames`;
    if (progress >= 1) finishScan();
  }
  lastFrameInfo = { sampleRate, feat, artifact };
  renderDiagnosis(feat, artifact);
  drawSpectrum(dbCal, sampleRate, feat.dominantHz);
  drawWaterfall(dbCal, sampleRate);
  drawMfcc(feat.mfcc);
}

function renderDiagnosis(feat, artifact) {
  const badge = $('artifact-badge');
  const reasons = [];
  if (artifact.wind) reasons.push('wind rumble');
  if (artifact.bump) reasons.push('bump / knock');
  if (artifact.clatter) reasons.push('external clatter');
  if (artifact.isArtifact) {
    badge.hidden = false;
    badge.textContent = 'Frame rejected: ' + reasons.join(', ') + ' — reading frozen';
  } else if (artifact.quiet) {
    badge.hidden = false;
    badge.textContent = 'Signal too quiet — move the mic closer to the hive';
  } else badge.hidden = true;

  if (!smoothed) {
    $('diag-state').textContent = artifact.quiet ? 'No signal' : 'Listening…';
    $('diag-sub').textContent = artifact.isArtifact ? 'Waiting for a clean frame' : 'Waiting for a usable signal';
    return;
  }
  const top = argmax(smoothed);
  const el = $('diag-state');
  el.textContent = LABEL_NAMES[LABELS[top]];
  el.style.color = stateColor(LABELS[top]);
  $('diag-sub').textContent = `${Math.round(smoothed[top] * 100)}% confidence · live estimate`;
  if (!artifact.quiet) $('diag-hz').textContent = feat.dominantHz.toFixed(0);
  LABELS.forEach((l, i) => {
    probEls[l].fill.style.width = (smoothed[i] * 100).toFixed(1) + '%';
    probEls[l].val.textContent = Math.round(smoothed[i] * 100) + '%';
  });
}

// ------------------------------------------------------------------ canvases

function fitCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
  if (w && h && (cv.width !== w || cv.height !== h)) {
    cv.width = w;
    cv.height = h;
  }
  return dpr;
}

const BAND_SHADES = [
  ['fanning', BANDS.fanning], ['healthy', BANDS.healthy], ['preswarm', BANDS.preswarm],
  ['piping', BANDS.pipingFund], ['queenless', BANDS.queenless],
];
let plotTop = -30;

function drawSpectrum(db, sampleRate, dominantHz) {
  const cv = $('cv-fft');
  if (cv.hidden) return;
  const dpr = fitCanvas(cv);
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const padB = 18 * dpr, plotH = H - padB;
  g.fillStyle = palette['plot-bg'];
  g.fillRect(0, 0, W, H);
  const bw = sampleRate / (db.length * 2);
  const kMax = Math.min(db.length - 1, Math.ceil(PLOT_MAX_HZ / bw));
  let peak = -200;
  for (let k = 3; k <= kMax; k++) if (db[k] > peak) peak = db[k];
  if (Number.isFinite(peak)) plotTop += 0.1 * (Math.ceil((peak + 8) / 10) * 10 - plotTop);
  const range = 70;
  const xOf = (hz) => (hz / PLOT_MAX_HZ) * W;
  const yOf = (d) => Math.min(plotH, Math.max(0, ((plotTop - d) / range) * plotH));

  g.font = `${10 * dpr}px -apple-system, sans-serif`;
  g.textBaseline = 'top';
  for (const [name, [lo, hi]] of BAND_SHADES) {
    g.globalAlpha = 0.13;
    g.fillStyle = stateColor(name);
    g.fillRect(xOf(lo), 0, xOf(hi) - xOf(lo), plotH);
    g.globalAlpha = 1;
  }
  g.fillStyle = palette.muted;
  g.textAlign = 'center';
  g.textBaseline = 'alphabetic';
  for (let hz = 200; hz < PLOT_MAX_HZ; hz += 200) {
    g.fillRect(xOf(hz), plotH, 1, 4 * dpr);
    g.fillText(hz + ' Hz', xOf(hz), H - 4 * dpr);
  }
  g.beginPath();
  for (let k = 1; k <= kMax; k++) {
    const x = xOf(k * bw), y = yOf(db[k]);
    if (k === 1) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.strokeStyle = palette.accent;
  g.lineWidth = 1.5 * dpr;
  g.lineJoin = 'round';
  g.stroke();
  if (dominantHz > 0 && dominantHz < PLOT_MAX_HZ) {
    g.fillStyle = palette.text;
    g.fillRect(xOf(dominantHz), 0, 1, plotH);
    g.textAlign = dominantHz > PLOT_MAX_HZ * 0.8 ? 'right' : 'left';
    g.textBaseline = 'top';
    g.fillText(' ' + dominantHz.toFixed(0) + ' Hz ', xOf(dominantHz), 4 * dpr);
  }
}

function heat(v) {
  // 0..1 → dark → amber → pale
  v = Math.min(1, Math.max(0, v));
  const stops = [[10, 8, 20], [70, 30, 90], [200, 80, 40], [245, 190, 50], [255, 250, 220]];
  const f = v * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), t = f - i;
  return stops[i].map((c, j) => Math.round(c + t * (stops[i + 1][j] - c)));
}

function drawWaterfall(db, sampleRate) {
  const cv = $('cv-waterfall');
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  g.drawImage(cv, 0, 0, W, H - 1, 0, 1, W, H - 1);
  const row = g.createImageData(W, 1);
  const bw = sampleRate / (db.length * 2);
  for (let x = 0; x < W; x++) {
    const k = Math.min(db.length - 1, Math.round(((x / W) * PLOT_MAX_HZ) / bw));
    const [r, gg, b] = heat((db[k] - (plotTop - 70)) / 70);
    row.data[x * 4] = r;
    row.data[x * 4 + 1] = gg;
    row.data[x * 4 + 2] = b;
    row.data[x * 4 + 3] = 255;
  }
  g.putImageData(row, 0, 0);
}

function drawMfcc(mfcc) {
  const cv = $('cv-mfcc');
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  g.drawImage(cv, 1, 0, W - 1, H, 0, 0, W - 1, H);
  const rowH = H / mfcc.length;
  for (let c = 0; c < mfcc.length; c++) {
    // C0 is overall level (large, negative); C1.. are spectral shape.
    const v = c === 0 ? (mfcc[0] + 500) / 400 : 0.5 + mfcc[c] / 60;
    const [r, gg, b] = heat(v);
    g.fillStyle = `rgb(${r},${gg},${b})`;
    g.fillRect(W - 1, Math.round((mfcc.length - 1 - c) * rowH), 1, Math.ceil(rowH));
  }
}

function drawTrend() {
  const cv = $('cv-trend');
  if (cv.hidden) return;
  const dpr = fitCanvas(cv);
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  g.fillStyle = palette['plot-bg'];
  g.fillRect(0, 0, W, H);
  g.font = `${11 * dpr}px -apple-system, sans-serif`;
  const list = inspections.filter((r) => r.hiveId === cfg.hive).sort((a, b) => a.id - b.id);
  g.fillStyle = palette.muted;
  if (!list.length) {
    g.textAlign = 'center';
    g.fillText(`No scans saved for “${cfg.hive}” yet`, W / 2, H / 2);
    return;
  }
  const pad = { l: 40 * dpr, r: 14 * dpr, t: 14 * dpr, b: 22 * dpr };
  let lo = Infinity, hi = -Infinity;
  for (const r of list) {
    lo = Math.min(lo, r.meanHz - r.sdHz);
    hi = Math.max(hi, r.meanHz + r.sdHz);
  }
  lo = Math.floor((lo - 20) / 50) * 50;
  hi = Math.ceil((hi + 20) / 50) * 50;
  const xOf = (i) => pad.l + (list.length === 1 ? 0.5 : i / (list.length - 1)) * (W - pad.l - pad.r);
  const yOf = (hz) => pad.t + ((hi - hz) / (hi - lo)) * (H - pad.t - pad.b);
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  for (let hz = lo; hz <= hi; hz += (hi - lo) / 4) {
    g.fillStyle = palette.line;
    g.fillRect(pad.l, yOf(hz), W - pad.l - pad.r, 1);
    g.fillStyle = palette.muted;
    g.fillText(Math.round(hz) + '', pad.l - 5 * dpr, yOf(hz));
  }
  g.textAlign = 'left';
  g.textBaseline = 'alphabetic';
  g.fillText(`${cfg.hive} · mean hum Hz ± SD · ${list.length} scan(s)`, pad.l, H - 6 * dpr);
  g.strokeStyle = palette.muted;
  g.lineWidth = 1.5 * dpr;
  g.beginPath();
  list.forEach((r, i) => (i ? g.lineTo(xOf(i), yOf(r.meanHz)) : g.moveTo(xOf(i), yOf(r.meanHz))));
  g.stroke();
  list.forEach((r, i) => {
    g.strokeStyle = g.fillStyle = stateColor(r.state);
    g.beginPath();
    g.moveTo(xOf(i), yOf(r.meanHz - r.sdHz));
    g.lineTo(xOf(i), yOf(r.meanHz + r.sdHz));
    g.stroke();
    g.beginPath();
    g.arc(xOf(i), yOf(r.meanHz), 5 * dpr, 0, Math.PI * 2);
    g.fill();
  });
}

function clearScrollingCanvases() {
  for (const id of ['cv-waterfall', 'cv-mfcc']) {
    const cv = $(id), g = cv.getContext('2d');
    g.fillStyle = palette['plot-bg'];
    g.fillRect(0, 0, cv.width, cv.height);
  }
}

// ------------------------------------------------------------------ start / stop / scan

function resetDiagnosis(idle) {
  smoothed = null;
  const el = $('diag-state');
  el.textContent = idle ? '—' : 'Listening…';
  el.style.color = '';
  $('diag-sub').textContent = idle ? 'Start listening to see a live reading' : 'Waiting for a usable signal';
  $('diag-hz').textContent = '—';
  $('artifact-badge').hidden = true;
  for (const l of LABELS) {
    if (!probEls[l]) continue;
    probEls[l].fill.style.width = '0%';
    probEls[l].val.textContent = '—';
  }
}

function setRunning(on) {
  const btn = $('start-btn');
  btn.textContent = on ? 'Stop' : 'Start listening';
  btn.classList.toggle('stop', on);
  btn.classList.toggle('primary', !on);
  $('scan-btn').disabled = !on;
  const pill = $('status-pill');
  pill.classList.toggle('live', on);
  pill.textContent = on ? `Live · ${(engine.sampleRate / 1000).toFixed(1)} kHz` : 'Idle';
  if (!on) {
    $('scan-progress').hidden = true;
    scan.reset();
    resetDiagnosis(true);
  }
  updateToolsMeta();
}

async function start() {
  resetDiagnosis(false);
  extractor.reset();
  try {
    if (source === 'mic') await engine.startMic();
    else if (source === 'file') {
      const file = $('file-input').files[0];
      if (!file) {
        toast('Choose an audio file first');
        return;
      }
      await engine.startFile(file);
    } else await engine.startSimulator($('sim-state').value);
    setRunning(true);
    requestPersistence();
  } catch (err) {
    engine.stop();
    setRunning(false);
    const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    toast(denied ? 'Microphone access was blocked. Allow it in the browser settings and try again.' : 'Could not start: ' + (err && err.message ? err.message : err), 6000);
  }
}

function stop() {
  engine.stop();
  setRunning(false);
}

function beginScan() {
  if (!engine.running || scan.active) return;
  if (!cfg.hive.trim()) {
    toast('Enter a Hive ID first');
    return;
  }
  engine.startRecording();
  scan.start(performance.now());
  $('scan-progress').hidden = false;
  $('scan-bar').style.width = '0%';
  $('scan-btn').disabled = true;
}

async function finishScan() {
  const chunks = engine.stopRecording();
  const result = scan.result(LABELS);
  const sr = engine.sampleRate;
  const mode = engine.mode || source;
  scan.reset();
  $('scan-progress').hidden = true;
  $('scan-btn').disabled = !engine.running;
  if (result.cleanFrames < MIN_CLEAN_FRAMES) {
    toast(`Only ${result.cleanFrames} clean frames (${result.artifactFrames} rejected, ${result.quietFrames} too quiet). Not saved — hold still and rescan.`, 6000);
    return;
  }
  const id = Date.now();
  const rec = {
    id,
    hiveId: cfg.hive.trim(),
    material: cfg.material,
    tempC: cfg.temp,
    timeOfDay: timeOfDay(),
    source: mode,
    state: result.state,
    confidence: result.confidence,
    probs: result.probs,
    meanHz: result.meanHz,
    sdHz: result.sdHz,
    cleanFrames: result.cleanFrames,
    artifactFrames: result.artifactFrames,
    sampleRate: sr,
    customWeights,
    hasWav: false,
  };
  try {
    if (chunks.length) {
      await saveClip(id, encodeMonoWAV(chunks, sr));
      rec.hasWav = true;
    }
  } catch (e) {
    rec.hasWav = false;
  }
  try {
    await saveInspection(rec);
  } catch (e) {
    toast('Could not save to this device (storage blocked?)', 5000);
  }
  inspections.unshift(rec);
  renderLog();
  toast(`Saved: ${LABEL_NAMES[rec.state]} · ${rec.meanHz.toFixed(0)} Hz ± ${rec.sdHz.toFixed(1)}`);
}

// ------------------------------------------------------------------ log, triage, exports

function renderLog() {
  const list = $('log-list');
  list.textContent = '';
  if (!inspections.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'No scans yet. Start listening, then run a 15 s scan to save one.';
    list.appendChild(p);
  }
  for (const r of inspections) {
    const row = document.createElement('div');
    row.className = 'log-row';
    row.style.setProperty('--state', stateColor(r.state));
    const main = document.createElement('div');
    main.className = 'log-main';
    const strong = document.createElement('strong');
    strong.textContent = `${r.hiveId} · ${LABEL_NAMES[r.state] || r.state} ${Math.round(r.confidence * 100)}%`;
    const small = document.createElement('small');
    small.textContent = `${fmtDate(r.id)} · ${r.meanHz.toFixed(0)} Hz ± ${r.sdHz.toFixed(1)} · ${r.cleanFrames} clean / ${r.artifactFrames} rejected`;
    main.append(strong, small);
    const actions = document.createElement('div');
    actions.className = 'log-actions';
    if (r.hasWav) {
      const wav = document.createElement('button');
      wav.type = 'button';
      wav.textContent = '⬇ .wav';
      wav.setAttribute('aria-label', `Download recording for ${r.hiveId}`);
      wav.onclick = async () => {
        const blob = await getClip(r.id);
        if (!blob) return toast('Recording not found');
        const stamp = new Date(r.id).toISOString().replace(/[:.]/g, '-').slice(0, 19);
        download(blob, `apissonic_${r.hiveId.replace(/[^\w-]+/g, '_')}_${r.state}_${stamp}.wav`);
      };
      actions.appendChild(wav);
    }
    const del = document.createElement('button');
    del.type = 'button';
    del.textContent = '✕';
    del.setAttribute('aria-label', `Delete scan for ${r.hiveId}`);
    del.onclick = async () => {
      if (!confirm('Delete this scan and its recording?')) return;
      try {
        await deleteInspection(r.id);
      } catch (e) { /* ignore */ }
      inspections = inspections.filter((x) => x.id !== r.id);
      renderLog();
    };
    actions.appendChild(del);
    row.append(main, actions);
    list.appendChild(row);
  }
  renderTriage();
  drawTrend();
}

function renderTriage() {
  const grid = $('triage-grid');
  grid.textContent = '';
  const cards = buildTriage(inspections);
  if (!cards.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'Hives appear here, most urgent first, once you have saved scans.';
    grid.appendChild(p);
    return;
  }
  for (const c of cards) {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'triage-card';
    el.style.setProperty('--state', stateColor(c.state));
    el.style.textAlign = 'left';
    el.style.font = 'inherit';
    el.style.color = 'inherit';
    el.style.cursor = 'pointer';
    const drift = c.driftHz === null ? 'first scan' : `Δ ${c.driftHz >= 0 ? '+' : ''}${c.driftHz.toFixed(1)} Hz`;
    const rank = document.createElement('span');
    rank.className = 't-rank';
    rank.textContent = 'Rank ' + c.rank;
    const hive = document.createElement('div');
    hive.className = 't-hive';
    hive.textContent = c.hiveId;
    const st = document.createElement('div');
    st.className = 't-state';
    st.textContent = `${LABEL_NAMES[c.state] || c.state} ${Math.round(c.confidence * 100)}%`;
    const meta = document.createElement('div');
    meta.className = 't-meta';
    meta.textContent = `${c.meanHz.toFixed(0)} Hz · ${drift} · ${fmtDate(c.latest.id)}`;
    el.append(rank, hive, st, meta);
    el.onclick = () => {
      cfg.hive = c.hiveId;
      $('hive-id').value = c.hiveId;
      store.set(CFG_KEY, cfg);
      selectTab('trend');
      $('cv-trend').scrollIntoView({ block: 'center', behavior: 'smooth' });
    };
    grid.appendChild(el);
  }
}

function exportCSV() {
  if (!inspections.length) return toast('Nothing to export yet');
  const cols = ['timestamp', 'hiveId', 'state', 'confidence', 'meanHz', 'sdHz', 'cleanFrames', 'artifactFrames', 'material', 'tempC', 'timeOfDay', 'source', 'sampleRate', ...LABELS.map((l) => 'p_' + l)];
  const q = (v) => '"' + String(v === undefined || v === null ? '' : v).replace(/"/g, '""') + '"';
  const lines = [cols.join(',')];
  for (const r of inspections) {
    lines.push([
      new Date(r.id).toISOString(), r.hiveId, r.state, r.confidence.toFixed(4), r.meanHz.toFixed(2), r.sdHz.toFixed(2),
      r.cleanFrames, r.artifactFrames, r.material, r.tempC, r.timeOfDay, r.source, r.sampleRate,
      ...(r.probs || []).map((p) => p.toFixed(4)),
    ].map(q).join(','));
  }
  download(new Blob([lines.join('\r\n')], { type: 'text/csv' }), `apissonic_log_${new Date().toISOString().slice(0, 10)}.csv`);
}

function printReport() {
  if (!inspections.length) return toast('Nothing to report yet');
  $('print-area').innerHTML = buildReportHTML(inspections, LABEL_NAMES);
  window.print();
}

function currentTagUrl() {
  return buildDeepLink(location.origin + location.pathname, { hive: cfg.hive.trim(), material: cfg.material, temp: cfg.temp });
}

function openQR() {
  if (!cfg.hive.trim()) return toast('Enter a Hive ID first');
  const url = currentTagUrl();
  try {
    $('qr-holder').innerHTML = qrToSVG(generateQRMatrix(url));
  } catch (err) {
    return toast(err.message, 5000);
  }
  $('qr-caption').textContent = cfg.hive.trim();
  $('qr-url').textContent = url + (/^(localhost|127\.|192\.168\.)/.test(location.hostname) ? ' — note: this address only works on this network. Print tags from the hosted app.' : '');
  $('qr-modal').hidden = false;
  $('qr-close-btn').focus();
}

function printTag() {
  const area = $('print-area');
  area.textContent = '';
  const tag = document.createElement('div');
  tag.className = 'tag';
  tag.innerHTML = $('qr-holder').innerHTML;
  const name = document.createElement('strong');
  name.textContent = cfg.hive.trim();
  const small = document.createElement('small');
  small.textContent = `${MATERIALS[cfg.material].label} · ApisSonic`;
  tag.append(name, small);
  area.appendChild(tag);
  $('qr-modal').hidden = true;
  window.print();
}

function updateToolsMeta() {
  const parts = [];
  parts.push(customWeights ? 'Model: custom weights loaded' : 'Model: built-in weights (synthetic training data)');
  parts.push(micFloor.floor ? 'Mic floor: calibrated' : 'Mic floor: not calibrated');
  if (engine.ctx) parts.push(`Sample rate: ${engine.sampleRate} Hz (bin ${(engine.sampleRate / FFT_SIZE).toFixed(2)} Hz)`);
  if (engine.mode === 'mic' && engine.micSettings) {
    const s = engine.micSettings;
    const flags = ['echoCancellation', 'noiseSuppression', 'autoGainControl'].filter((k) => s[k] === true);
    parts.push(flags.length ? `⚠ Phone voice processing still ON: ${flags.join(', ')}` : 'Voice processing: off (as reported by the browser)');
  }
  $('tools-meta').textContent = parts.join(' · ');
  $('signal-meta').textContent = engine.ctx ? `4096-pt FFT · ${(engine.sampleRate / FFT_SIZE).toFixed(2)} Hz per bin · shaded bands: fanning, healthy, pre-swarm, piping, queenless` : '';
}

// ------------------------------------------------------------------ UI wiring

const probEls = {};
function buildProbBars() {
  const wrap = $('prob-bars');
  for (const l of LABELS) {
    const row = document.createElement('div');
    row.className = 'prob-row';
    const name = document.createElement('span');
    name.textContent = LABEL_NAMES[l];
    const track = document.createElement('div');
    track.className = 'prob-track';
    const fill = document.createElement('div');
    fill.className = 'prob-fill';
    fill.style.background = `var(--${l})`;
    track.appendChild(fill);
    const val = document.createElement('span');
    val.className = 'prob-val';
    val.textContent = '—';
    row.append(name, track, val);
    wrap.appendChild(row);
    probEls[l] = { fill, val };
  }
}

function selectTab(tab) {
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  for (const t of ['fft', 'waterfall', 'mfcc', 'trend']) $('cv-' + t).hidden = t !== tab;
  if (tab === 'trend') drawTrend();
}

function selectSource(next) {
  if (engine.running) stop();
  source = next;
  document.querySelectorAll('.segmented button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.source === next)));
  $('source-file').hidden = next !== 'file';
  $('source-sim').hidden = next !== 'sim';
  $('mic-hint').hidden = next !== 'mic';
}

function applyTheme(theme) {
  if (theme) document.documentElement.setAttribute('data-theme', theme);
  refreshPalette();
  clearScrollingCanvases();
  renderLog();
}

function init() {
  const mat = $('material');
  for (const [id, m] of Object.entries(MATERIALS)) mat.add(new Option(m.label, id));
  const sim = $('sim-state');
  for (const s of SYNTH_STATES) sim.add(new Option(LABEL_NAMES[s], s));
  $('hive-id').value = cfg.hive;
  mat.value = cfg.material;
  $('temp').value = cfg.temp;
  $('tod').value = cfg.tod;
  buildProbBars();

  const saveCfg = () => {
    cfg.hive = $('hive-id').value;
    cfg.material = mat.value;
    const t = parseFloat($('temp').value);
    cfg.temp = Number.isFinite(t) ? t : 22;
    cfg.tod = $('tod').value;
    store.set(CFG_KEY, cfg);
    drawTrend();
  };
  for (const id of ['hive-id', 'material', 'temp', 'tod']) $(id).addEventListener('change', saveCfg);
  store.set(CFG_KEY, cfg);

  document.querySelectorAll('.segmented button').forEach((b) => b.addEventListener('click', () => selectSource(b.dataset.source)));
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab)));
  $('start-btn').addEventListener('click', () => (engine.running ? stop() : start()));
  $('scan-btn').addEventListener('click', beginScan);
  $('sim-state').addEventListener('change', () => {
    if (engine.running && source === 'sim') start();
  });
  $('csv-btn').addEventListener('click', exportCSV);
  $('report-btn').addEventListener('click', printReport);
  $('qr-btn').addEventListener('click', openQR);
  $('qr-close-btn').addEventListener('click', () => ($('qr-modal').hidden = true));
  $('qr-print-btn').addEventListener('click', printTag);
  $('qr-modal').addEventListener('click', (e) => {
    if (e.target === $('qr-modal')) $('qr-modal').hidden = true;
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $('qr-modal').hidden = true;
  });
  $('clear-btn').addEventListener('click', async () => {
    if (!inspections.length || !confirm('Delete ALL saved scans and recordings on this device?')) return;
    try {
      await clearAll();
    } catch (e) { /* ignore */ }
    inspections = [];
    renderLog();
  });
  $('floor-btn').addEventListener('click', () => {
    if (!engine.running) return toast('Start listening first, with the mic away from the hive');
    micFloor.begin(FFT_SIZE / 2);
    floorUntil = performance.now() + 3000;
    toast('Measuring ambient noise for 3 s — keep quiet');
  });
  $('floor-clear-btn').addEventListener('click', () => {
    micFloor.clear();
    updateToolsMeta();
  });
  $('weights-btn').addEventListener('click', () => $('weights-input').click());
  $('weights-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    try {
      const parsed = validateWeights(JSON.parse(await file.text()));
      weights = parsed;
      customWeights = true;
      store.set(WEIGHTS_KEY, parsed);
      toast('Custom weights loaded');
    } catch (err) {
      toast('Invalid weights file: ' + err.message, 6000);
    }
    updateToolsMeta();
  });
  $('weights-reset-btn').addEventListener('click', () => {
    weights = DEFAULT_WEIGHTS;
    customWeights = false;
    store.remove(WEIGHTS_KEY);
    updateToolsMeta();
    toast('Using built-in weights');
  });
  $('theme-btn').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    const next = cur === 'light' ? 'dark' : 'light';
    store.set(THEME_KEY, next);
    applyTheme(next);
  });
  window.addEventListener('resize', drawTrend);
  document.addEventListener('visibilitychange', () => {
    // iOS suspends audio in the background; stop cleanly rather than show a frozen reading.
    if (document.hidden && engine.running && engine.mode === 'mic' && !scan.active) stop();
  });

  applyTheme(store.get(THEME_KEY, null));
  updateToolsMeta();
  if (link.hive) toast(`Hive tag loaded: ${cfg.hive}`);

  listInspections()
    .then((all) => {
      inspections = all;
      renderLog();
    })
    .catch(() => toast('Saved scans are unavailable in this browser mode'));

  if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
}

init();
