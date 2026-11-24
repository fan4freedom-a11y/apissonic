// tests/dsp-ml-tests.js — 17 deterministic DSP, MFCC, MLP, WAV, calibration, QR & scan-image checks.
// Browser: open test-suite.html.   Node: node tests/dsp-ml-tests.js

import * as dsp from './dsp-engine.js';
import * as ml from './ml-classifier.js';
import * as cal from './calibration.js';
import { encodeMonoWAVBuffer } from './audio-store.js';
import { parseDeepLink, buildDeepLink, generateQRMatrix, buildTriage } from './qr-report.js';
import { ScanImageRecorder, summarizeScanFrames, IMAGE_MAX_HZ } from './scan-images.js';

const tests = [];
const test = (name, fn) => tests.push({ name, fn });
function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed');
}
function near(a, b, tol, msg) {
  if (!(Math.abs(a - b) <= tol)) throw new Error(`${msg || 'value'}: expected ${b} ± ${tol}, got ${a}`);
}

const SR = 44100;
const sine = (hz, n = dsp.FFT_SIZE, amp = 0.5, sr = SR) => Float32Array.from({ length: n }, (_, i) => amp * Math.sin((2 * Math.PI * hz * i) / sr));
const flatDb = (v) => new Float32Array(dsp.FFT_SIZE / 2).fill(v);
const analyse = (frame) => new dsp.OfflineAnalyser(dsp.FFT_SIZE, 0).process(frame);

test('1. FFT bin width is 44100 / 4096 ≈ 10.77 Hz', () => {
  near(dsp.binWidth(44100), 10.7666, 0.001, 'bin width');
  near(dsp.binWidth(48000), 11.7188, 0.001, 'bin width at 48 kHz');
});

test('2. Band power of a flat spectrum equals the bin level', () => {
  near(dsp.computeBandPower(flatDb(-40), SR, 200, 290), -40, 1e-6, 'flat band');
});

test('3. Band power averages in linear power, not in dB', () => {
  const db = flatDb(-200);
  const bw = dsp.binWidth(SR);
  const k0 = Math.ceil(200 / bw), k1 = Math.floor(290 / bw);
  db[k0] = 0; // one bin at 0 dB, the rest silent
  near(dsp.computeBandPower(db, SR, 200, 290), -10 * Math.log10(k1 - k0 + 1), 1e-3, 'linear mean');
});

test('4. Dominant frequency of a 250 Hz tone is found within 1 Hz', () => {
  near(dsp.dominantFrequency(analyse(sine(250)), SR), 250, 1, 'dominant Hz');
  const at48k = new dsp.OfflineAnalyser(dsp.FFT_SIZE, 0).process(sine(250, dsp.FFT_SIZE, 0.5, 48000));
  near(dsp.dominantFrequency(at48k, 48000), 250, 1, 'dominant Hz at 48 kHz');
});

test('5. Wind rumble (< 85 Hz) is flagged as an artifact', () => {
  const frame = sine(50, dsp.FFT_SIZE, 0.5);
  const hum = sine(250, dsp.FFT_SIZE, 0.01);
  for (let i = 0; i < frame.length; i++) frame[i] += hum[i];
  const r = new dsp.ArtifactDetector().check(analyse(frame), frame, SR);
  assert(r.wind && r.isArtifact, 'wind not detected');
});

test('6. Mic bump: crest factor > 6.5 and a > 9.5 dB RMS jump are flagged', () => {
  const det = new dsp.ArtifactDetector();
  const hum = sine(250, dsp.FFT_SIZE, 0.05);
  for (let i = 0; i < 5; i++) assert(!det.check(analyse(hum), hum, SR).isArtifact, 'clean hum flagged');
  const click = Float32Array.from(hum);
  click[2000] = 0.9;
  assert(det.check(analyse(click), click, SR).bump, 'crest-factor bump missed');
  const loud = sine(250, dsp.FFT_SIZE, 0.05 * Math.pow(10, 12 / 20));
  assert(det.check(analyse(loud), loud, SR).bump, '12 dB RMS jump missed');
});

test('7. External clatter (1250–3500 Hz) is flagged; a clean hum is not', () => {
  const hum = sine(250, dsp.FFT_SIZE, 0.05);
  const noisy = Float32Array.from(hum);
  // Clatter is broadband: 60 partials spread across 1250–3500 Hz.
  for (let j = 0; j < 60; j++) {
    const clang = sine(1250 + j * 37.5, dsp.FFT_SIZE, 0.1);
    for (let i = 0; i < noisy.length; i++) noisy[i] += clang[i];
  }
  assert(new dsp.ArtifactDetector().check(analyse(noisy), noisy, SR).clatter, 'clatter missed');
  const clean = new dsp.ArtifactDetector().check(analyse(hum), hum, SR);
  assert(!clean.isArtifact && !clean.quiet, 'clean hum rejected');
});

test('8. Hz ↔ Mel conversion round-trips', () => {
  near(ml.hzToMel(1000), 1000, 0.1, 'mel(1000 Hz)');
  for (const f of [60, 250, 700, 2000]) near(ml.melToHz(ml.hzToMel(f)), f, 1e-6, 'round trip');
});

test('9. Mel filterbank: 24 triangular filters spanning 60–2000 Hz', () => {
  const fb = ml.buildMelFilterbank(SR, dsp.FFT_SIZE);
  assert(fb.length === 24, 'expected 24 filters');
  const bw = dsp.binWidth(SR);
  assert(fb[0].start * bw >= 60, 'first filter starts below 60 Hz');
  const last = fb[23];
  assert((last.start + last.weights.length - 1) * bw <= 2000, 'last filter ends above 2000 Hz');
  for (const f of fb) {
    assert(f.weights.length >= 2, 'a filter is empty');
    let peak = 0;
    for (const w of f.weights) {
      assert(w >= 0 && w <= 1, 'weight out of range');
      peak = Math.max(peak, w);
    }
    assert(peak > 0.5, 'filter has no usable peak');
  }
});

test('10. DCT-II is orthonormal: flat energies give C0 = √M·E and C1..C12 = 0', () => {
  const C = ml.mfccFromLogMel(new Float64Array(24).fill(-50));
  assert(C.length === 13, 'expected 13 MFCCs');
  near(C[0], Math.sqrt(24) * -50, 1e-9, 'C0');
  for (let c = 1; c < 13; c++) near(C[c], 0, 1e-9, 'C' + c);
});

test('11. MFCC C1..C12 are invariant to microphone gain (+20 dB)', () => {
  const frame = dsp.synthesizeColony('healthy', 0.2, SR, dsp.seededRandom(3)).subarray(0, dsp.FFT_SIZE);
  const db = analyse(frame);
  const fb = ml.buildMelFilterbank(SR, dsp.FFT_SIZE);
  const a = ml.mfccFromLogMel(ml.logMelEnergies(db, fb));
  const b = ml.mfccFromLogMel(ml.logMelEnergies(db.map((v) => v + 20), fb));
  for (let c = 1; c < 13; c++) near(b[c], a[c], 1e-3, 'C' + c);
  near(b[0] - a[0], 20 * Math.sqrt(24), 1e-2, 'C0 shift');
});

test('12. Roll-off, flux and zero-crossing rate behave as specified', () => {
  const db = analyse(sine(400));
  near(ml.spectralRolloff(db, SR), 400, 15, '85 % roll-off of a 400 Hz tone');
  near(ml.zeroCrossingRate(sine(300), SR), 600, 12, 'ZCR of a 300 Hz tone');
  const p = ml.normalizedPower(db, SR);
  near(ml.spectralFlux(p, p), 0, 1e-12, 'flux of identical frames');
  assert(ml.spectralFlux(p, ml.normalizedPower(analyse(sine(800)), SR)) > 0.1, 'flux should rise when the spectrum changes');
});

test('13. MLP 18→8→5: LeakyReLU slope 0.05, stable softmax, valid built-in weights', () => {
  near(ml.leakyRelu(-2), -0.1, 1e-12, 'LeakyReLU(-2)');
  near(ml.leakyRelu(3), 3, 1e-12, 'LeakyReLU(3)');
  const big = ml.softmax([1000, 999, -1000, 0, 5]);
  assert(big.every(Number.isFinite), 'softmax overflowed');
  near(big.reduce((a, b) => a + b, 0), 1, 1e-12, 'softmax sum');
  const w = ml.validateWeights(ml.DEFAULT_WEIGHTS);
  assert(w.W1.length === 8 && w.W1[0].length === 18 && w.W2.length === 5 && w.W2[0].length === 8, 'weight shapes');
  const out = ml.mlpForward(new Float64Array(18), w);
  assert(out.length === 5, 'five outputs');
  near(out.reduce((a, b) => a + b, 0), 1, 1e-9, 'probabilities sum');
  let threw = false;
  try {
    ml.validateWeights({ W1: [[1]], b1: [], W2: [], b2: [] });
  } catch (e) {
    threw = true;
  }
  assert(threw, 'bad weights accepted');
});

test('14. WAV encoder: 44-byte RIFF header, 2× downsample, clamped 16-bit samples', () => {
  const input = new Float32Array(2000).fill(0.25);
  input[0] = input[1] = 2; // must clamp
  input[2] = input[3] = -2;
  const buf = encodeMonoWAVBuffer([input.subarray(0, 800), input.subarray(800)], 44100);
  const v = new DataView(buf);
  const tag = (o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  assert(tag(0) === 'RIFF' && tag(8) === 'WAVE' && tag(12) === 'fmt ' && tag(36) === 'data', 'chunk tags');
  assert(buf.byteLength === 44 + 1000 * 2, 'file length');
  assert(v.getUint32(4, true) === 36 + 1000 * 2, 'RIFF chunk size');
  assert(v.getUint16(20, true) === 1 && v.getUint16(22, true) === 1, 'PCM mono');
  assert(v.getUint32(24, true) === 22050 && v.getUint32(28, true) === 44100, 'sample / byte rate');
  assert(v.getUint16(32, true) === 2 && v.getUint16(34, true) === 16, 'block align / bits');
  assert(v.getInt16(44, true) === 32767 && v.getInt16(46, true) === -32768, 'clamping');
  assert(v.getInt16(48, true) === Math.round(0.25 * 32767), 'sample value');
  assert(new DataView(encodeMonoWAVBuffer(input, 48000)).getUint32(24, true) === 24000, '48 kHz input → 24 kHz header');
});

test('15. Calibration: material gains and temperature / night priors', () => {
  const bands = { fanning: -60, healthy: -60, preswarm: -60, pipingFund: -60, pipingHarm: -60, queenless: -60 };
  const eps = cal.applyMaterialToBands(bands, 'polystyrene_eps');
  near(eps.healthy, -58.8, 1e-9, 'EPS low');
  near(eps.preswarm, -56.8, 1e-9, 'EPS swarm');
  near(eps.pipingFund, -54.2, 1e-9, 'EPS piping');
  near(eps.queenless, -52.5, 1e-9, 'EPS high');
  const board = cal.applyMaterialToBands(bands, 'landing_board');
  assert(Object.values(board).every((v) => v === -60), 'landing board must be 0 dB');
  near(cal.applyMaterialToBands(bands, 'wood_langstroth').queenless, -56.2, 1e-9, 'wood high');
  near(cal.applyMaterialToBands(bands, 'thick_timber').pipingFund, -56.0, 1e-9, 'timber piping');
  near(cal.materialOffsetDb('polystyrene_eps', 900), 7.5, 1e-9, 'per-bin EPS high');
  near(cal.environmentMultipliers(30, 'day').fanning, 1.11, 1e-9, 'fanning at 30 °C');
  near(cal.environmentMultipliers(45, 'day').fanning, 1.65, 1e-9, 'fanning cap');
  near(cal.environmentMultipliers(22, 'day').fanning, 1, 1e-9, 'mild day');
  const cold = cal.environmentMultipliers(10, 'night');
  near(cold.fanning, 0.65, 1e-9, 'cold fanning');
  near(cold.queenless, 1.15 * 1.18, 1e-9, 'cold night queenless');
  near(cold.piping, 1.22, 1e-9, 'night piping');
  near(cal.environmentPriors(ml.LABELS, 10, 'night').reduce((a, b) => a + b, 0), 1, 1e-12, 'priors normalised');
});

test('16. End to end: synthesized colonies classify correctly; scan, triage and QR tag work', () => {
  const priors = cal.environmentPriors(ml.LABELS, 22, 'day');
  for (const sr of [44100, 48000]) {
    ml.LABELS.forEach((state, si) => {
      const x = dsp.synthesizeColony(state, 2, sr, dsp.seededRandom(100 + si + sr));
      const an = new dsp.OfflineAnalyser();
      const fx = new ml.FeatureExtractor();
      const det = new dsp.ArtifactDetector();
      const scan = new dsp.ScanAccumulator(1e9);
      scan.start(0);
      const hop = Math.round(0.05 * sr);
      let i = 0;
      for (let pos = dsp.FFT_SIZE; pos <= x.length; pos += hop, i++) {
        const frame = x.subarray(pos - dsp.FFT_SIZE, pos);
        const db = an.process(frame);
        const art = det.check(db, frame, sr);
        const f = fx.extract(db, frame, sr);
        if (i < 10) continue; // analyser smoothing warm-up
        const fused = ml.fuseEnsemble(ml.mlpForward(f.x, ml.DEFAULT_WEIGHTS), ml.bayesBandRatio(dsp.computeAllBands(db, sr), priors));
        scan.add({ isArtifact: art.isArtifact, quiet: art.quiet, dominantHz: f.dominantHz, probs: fused }, i);
      }
      const r = scan.result(ml.LABELS);
      assert(r.state === state, `${state} @ ${sr} Hz classified as ${r.state}`);
      assert(r.artifactFrames === 0 && r.cleanFrames > 15, `${state}: unexpected artifacts`);
      assert(r.sdHz >= 0 && r.meanHz > 100 && r.meanHz < 1000, 'scan statistics');
    });
  }
  const cards = buildTriage([
    { id: 1, hiveId: 'A', state: 'healthy', confidence: 0.9, meanHz: 240 },
    { id: 2, hiveId: 'A', state: 'healthy', confidence: 0.9, meanHz: 252 },
    { id: 3, hiveId: 'B', state: 'queenless', confidence: 0.7, meanHz: 610 },
    { id: 4, hiveId: 'C', state: 'preswarm', confidence: 0.8, meanHz: 350 },
  ]);
  assert(cards.map((c) => c.hiveId).join('') === 'BCA', 'triage order');
  near(cards[2].driftHz, 12, 1e-9, 'drift ΔHz');
  const url = buildDeepLink('https://example.org/app/', { hive: 'Hive 07', material: 'polystyrene_eps', temp: 31.5 });
  const back = parseDeepLink(url);
  assert(back.hive === 'Hive 07' && back.material === 'polystyrene_eps' && back.temp === 31.5, 'deep-link round trip');
  const small = generateQRMatrix('HIVE-07');
  assert(small.size === 21 && small.modules.length === 21, 'short payload should be 21×21');
  const finder = ['1111111', '1000001', '1011101', '1011101', '1011101', '1000001', '1111111'];
  for (let y = 0; y < 7; y++) assert(small.modules[y].slice(0, 7).map((b) => (b ? 1 : 0)).join('') === finder[y], 'finder pattern');
  const big = generateQRMatrix(url);
  assert(big.size > 21 && big.size <= 41, 'URL payload picks a larger version');
});

test('17. Scan images: recorder keeps 0–1200 Hz per frame and the saved spectrum averages clean frames in linear power', () => {
  const rec = new ScanImageRecorder();
  const n = dsp.FFT_SIZE / 2;
  const binHz = SR / dsp.FFT_SIZE;
  const frameAt = (peakHz, level) => {
    const db = new Float32Array(n).fill(-90);
    db[Math.round(peakHz / binHz)] = level;
    return db;
  };
  const mfcc = new Float32Array(13).fill(1);
  rec.add(frameAt(250, -20), mfcc, SR, true);
  rec.add(frameAt(250, -30), mfcc, SR, true);
  rec.add(frameAt(900, 0), mfcc, SR, false); // rejected frame: shown in waterfall, excluded from the mean
  assert(rec.length === 3, 'frame count');
  assert(rec.frames[0].spec.length === Math.ceil(IMAGE_MAX_HZ / binHz) + 1, 'spectrum cropped to 0–1200 Hz');
  assert(rec.frames[0].mfcc.length === 13 && rec.frames[0].mfcc !== mfcc, 'MFCCs copied per frame');
  const sum = summarizeScanFrames(rec.frames, rec.binHz);
  assert(sum.usedFrames === 2, 'only clean frames averaged');
  near(sum.peakHz, Math.round(250 / binHz) * binHz, 1e-9, 'peak of mean spectrum');
  near(sum.meanDb[Math.round(250 / binHz)], 10 * Math.log10((1e-2 + 1e-3) / 2), 1e-4, 'power-domain mean');
  assert(sum.topDb >= 0, 'colour range covers the loudest (rejected) frame too');
  rec.reset();
  assert(rec.length === 0 && summarizeScanFrames(rec.frames, binHz) === null, 'reset');
});

export async function runAllTests() {
  const results = [];
  for (const t of tests) {
    try {
      await t.fn();
      results.push({ name: t.name, ok: true });
    } catch (err) {
      results.push({ name: t.name, ok: false, error: err.message });
    }
  }
  return results;
}

// Node entry point: `node tests/dsp-ml-tests.js`
if (typeof window === 'undefined' && typeof process !== 'undefined') {
  runAllTests().then((results) => {
    for (const r of results) console.log((r.ok ? 'PASS  ' : 'FAIL  ') + r.name + (r.ok ? '' : '\n      ' + r.error));
    const passed = results.filter((r) => r.ok).length;
    console.log(`\n${passed}/${results.length} tests passed`);
    process.exit(passed === results.length ? 0 : 1);
  });
}
