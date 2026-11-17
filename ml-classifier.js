// ml-classifier.js — ApisSonic Phase 3
// 24-band Mel filterbank → orthonormal DCT-II (13 MFCCs) + roll-off / flux / ZCR
// → 18→8→5 MLP (LeakyReLU + softmax), a Bayesian band-ratio classifier, and 60/40 fusion.

import { dominantFrequency, spectralCentroid } from './dsp-engine.js';

export const N_MEL = 24;
export const N_MFCC = 13;
export const MEL_LO_HZ = 60;
export const MEL_HI_HZ = 2000;
export const N_FEATURES = 18;
export const N_HIDDEN = 8;
export const LABELS = ['healthy', 'preswarm', 'queenless', 'piping', 'fanning'];
export const LABEL_NAMES = {
  healthy: 'Healthy',
  preswarm: 'Pre-Swarm',
  queenless: 'Queenless',
  piping: 'Queen Piping',
  fanning: 'Fanning',
};
export const MLP_WEIGHT = 0.6;
export const BAYES_WEIGHT = 0.4;

export const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
export const melToHz = (m) => 700 * (Math.pow(10, m / 2595) - 1);

/** Returns nMel sparse triangular filters: [{ start, weights: Float64Array }]. */
export function buildMelFilterbank(sampleRate, fftSize, nMel = N_MEL, loHz = MEL_LO_HZ, hiHz = MEL_HI_HZ) {
  const bins = fftSize / 2;
  const bw = sampleRate / fftSize;
  const mLo = hzToMel(loHz), mHi = hzToMel(hiHz);
  const edges = [];
  for (let i = 0; i < nMel + 2; i++) edges.push(melToHz(mLo + ((mHi - mLo) * i) / (nMel + 1)));
  const filters = [];
  for (let m = 0; m < nMel; m++) {
    const f0 = edges[m], f1 = edges[m + 1], f2 = edges[m + 2];
    const k0 = Math.max(0, Math.ceil(f0 / bw));
    const k1 = Math.min(bins - 1, Math.floor(f2 / bw));
    const w = new Float64Array(Math.max(0, k1 - k0 + 1));
    for (let k = k0; k <= k1; k++) {
      const f = k * bw;
      w[k - k0] = f <= f1 ? (f - f0) / (f1 - f0) : (f2 - f) / (f2 - f1);
      if (w[k - k0] < 0) w[k - k0] = 0;
    }
    filters.push({ start: k0, weights: w });
  }
  return filters;
}

/** E_m = 10·log10(Σ P_lin[k]·H_m[k] + 1e-10) */
export function logMelEnergies(db, filters) {
  const E = new Float64Array(filters.length);
  for (let m = 0; m < filters.length; m++) {
    const { start, weights } = filters[m];
    let s = 0;
    for (let i = 0; i < weights.length; i++) s += Math.pow(10, db[start + i] / 10) * weights[i];
    E[m] = 10 * Math.log10(s + 1e-10);
  }
  return E;
}

/** Orthonormal DCT-II. C0 carries overall level; C1.. use mean-removed energies (gain invariant). */
export function mfccFromLogMel(E, nCoeffs = N_MFCC) {
  const M = E.length;
  let mean = 0;
  for (let m = 0; m < M; m++) mean += E[m];
  mean /= M;
  const C = new Float64Array(nCoeffs);
  for (let c = 0; c < nCoeffs; c++) {
    const scale = c === 0 ? Math.sqrt(1 / M) : Math.sqrt(2 / M);
    let s = 0;
    for (let m = 0; m < M; m++) {
      const e = c === 0 ? E[m] : E[m] - mean;
      s += e * Math.cos((Math.PI * c * (m + 0.5)) / M);
    }
    C[c] = scale * s;
  }
  return C;
}

function analysisRange(db, sampleRate, loHz, hiHz) {
  const bw = sampleRate / (db.length * 2);
  return [Math.max(0, Math.ceil(loHz / bw)), Math.min(db.length - 1, Math.floor(hiHz / bw)), bw];
}

/** Lowest frequency where cumulative linear power reaches 85 % of the total (60–2000 Hz). */
export function spectralRolloff(db, sampleRate, fraction = 0.85, loHz = MEL_LO_HZ, hiHz = MEL_HI_HZ) {
  const [k0, k1, bw] = analysisRange(db, sampleRate, loHz, hiHz);
  let total = 0;
  for (let k = k0; k <= k1; k++) total += Math.pow(10, db[k] / 10);
  if (total <= 0) return 0;
  let cum = 0;
  for (let k = k0; k <= k1; k++) {
    cum += Math.pow(10, db[k] / 10);
    if (cum >= fraction * total) return k * bw;
  }
  return k1 * bw;
}

/** Bin power normalised to unit sum over 60–2000 Hz (input to spectral flux). */
export function normalizedPower(db, sampleRate, loHz = MEL_LO_HZ, hiHz = MEL_HI_HZ) {
  const [k0, k1] = analysisRange(db, sampleRate, loHz, hiHz);
  const p = new Float64Array(k1 - k0 + 1);
  let total = 0;
  for (let k = k0; k <= k1; k++) {
    p[k - k0] = Math.pow(10, db[k] / 10);
    total += p[k - k0];
  }
  if (total > 0) for (let i = 0; i < p.length; i++) p[i] /= total;
  return p;
}

/** L2 norm of the frame-to-frame change in normalised bin power. */
export function spectralFlux(prev, cur) {
  if (!prev || prev.length !== cur.length) return 0;
  let s = 0;
  for (let i = 0; i < cur.length; i++) {
    const d = cur[i] - prev[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Sign changes per second across the time buffer. */
export function zeroCrossingRate(time, sampleRate) {
  let n = 0;
  for (let i = 1; i < time.length; i++) if (time[i - 1] >= 0 !== time[i] >= 0) n++;
  return (n * sampleRate) / time.length;
}

/** Stateful extractor: caches the filterbank and the previous frame for flux. */
export class FeatureExtractor {
  constructor() {
    this.filters = null;
    this.key = '';
    this.prevNorm = null;
  }
  reset() {
    this.prevNorm = null;
  }
  /** Returns { x (18 features), mfcc, dominantHz, centroidHz, rolloffHz, flux, zcr }. */
  extract(db, time, sampleRate) {
    const key = sampleRate + ':' + db.length;
    if (key !== this.key) {
      this.filters = buildMelFilterbank(sampleRate, db.length * 2);
      this.key = key;
      this.prevNorm = null;
    }
    const mfcc = mfccFromLogMel(logMelEnergies(db, this.filters));
    const dominantHz = dominantFrequency(db, sampleRate);
    const centroidHz = spectralCentroid(db, sampleRate);
    const rolloffHz = spectralRolloff(db, sampleRate);
    const norm = normalizedPower(db, sampleRate);
    const flux = spectralFlux(this.prevNorm, norm);
    this.prevNorm = norm;
    const zcr = zeroCrossingRate(time, sampleRate);
    const x = new Float64Array(N_FEATURES);
    x.set(mfcc, 0);
    x[13] = dominantHz;
    x[14] = centroidHz;
    x[15] = rolloffHz;
    x[16] = flux;
    x[17] = zcr;
    return { x, mfcc, dominantHz, centroidHz, rolloffHz, flux, zcr };
  }
}

// ------------------------------------------------------------------ the MLP

export const leakyRelu = (z) => Math.max(0.05 * z, z);

export function softmax(z) {
  let max = -Infinity;
  for (const v of z) if (v > max) max = v;
  const e = Array.from(z, (v) => Math.exp(v - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** weights: { W1: 8×18, b1: 8, W2: 5×8, b2: 5 }. Returns 5 probabilities in LABELS order. */
export function mlpForward(x, weights) {
  const { W1, b1, W2, b2 } = weights;
  const h = new Array(b1.length);
  for (let j = 0; j < b1.length; j++) {
    let s = b1[j];
    for (let i = 0; i < x.length; i++) s += W1[j][i] * x[i];
    h[j] = leakyRelu(s);
  }
  const z = new Array(b2.length);
  for (let o = 0; o < b2.length; o++) {
    let s = b2[o];
    for (let j = 0; j < h.length; j++) s += W2[o][j] * h[j];
    z[o] = s;
  }
  return softmax(z);
}

/** Throws a readable Error if the object is not a valid 18→8→5 weight set. */
export function validateWeights(w) {
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const matrix = (m, r, c, name) => {
    if (!Array.isArray(m) || m.length !== r) throw new Error(`${name} must have ${r} rows`);
    m.forEach((row, i) => {
      if (!Array.isArray(row) || row.length !== c || !row.every(isNum)) throw new Error(`${name}[${i}] must be ${c} numbers`);
    });
  };
  const vector = (v, n, name) => {
    if (!Array.isArray(v) || v.length !== n || !v.every(isNum)) throw new Error(`${name} must be ${n} numbers`);
  };
  if (!w || typeof w !== 'object') throw new Error('Weights file is not a JSON object');
  matrix(w.W1, N_HIDDEN, N_FEATURES, 'W1');
  vector(w.b1, N_HIDDEN, 'b1');
  matrix(w.W2, LABELS.length, N_HIDDEN, 'W2');
  vector(w.b2, LABELS.length, 'b2');
  if (w.labels && w.labels.join() !== LABELS.join()) throw new Error('labels must be ' + LABELS.join(', '));
  return { W1: w.W1, b1: w.b1, W2: w.W2, b2: w.b2 };
}

// ------------------------------------------- Bayesian band-ratio classifier

/**
 * bands: calibrated band powers in dB (see dsp-engine computeAllBands).
 * priors: 5 probabilities in LABELS order. Returns posterior in LABELS order.
 */
export function bayesBandRatio(bands, priors, tempDb = 4) {
  // Piping = fundamental AND 2nd harmonic, with a quiet gap between them.
  // Queenless = broadband 550–1000 Hz, i.e. the gap is filled in.
  const contrast = bands.pipingHarm - bands.pipingGap;
  const score = {
    healthy: bands.healthy,
    preswarm: bands.preswarm,
    fanning: bands.fanning,
    piping: 0.5 * (bands.pipingFund + bands.pipingHarm) + Math.max(-6, Math.min(6, contrast - 6)),
    queenless: bands.queenless + 3 - Math.max(0, Math.min(9, contrast - 3)),
  };
  let max = -Infinity;
  for (const l of LABELS) if (score[l] > max) max = score[l];
  const post = LABELS.map((l, i) => priors[i] * Math.exp((score[l] - max) / tempDb));
  const s = post.reduce((a, b) => a + b, 0);
  return post.map((v) => v / s);
}

/** 60 % neural MLP + 40 % Bayesian band-ratio. */
export function fuseEnsemble(mlp, bayes, wMlp = MLP_WEIGHT, wBayes = BAYES_WEIGHT) {
  const out = mlp.map((p, i) => wMlp * p + wBayes * bayes[i]);
  const s = out.reduce((a, b) => a + b, 0);
  return out.map((v) => v / s);
}

export function argmax(a) {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
}

// Trained by ml/train_hive_classifier.py --synthetic-demo (input normalisation folded into W1/b1).
// These are SYNTHETIC-DATA weights: retrain on real recordings before trusting a diagnosis.
/*WEIGHTS-BEGIN*/
export const DEFAULT_WEIGHTS = {"W1":[[-0.001463,0.007664,-0.003714,-0.020208,-0.007728,-0.008456,-0.030878,0.008012,-0.021262,-0.033722,-0.048874,0.033239,0.025866,0.002803,-0.000663,-0.000531,-15.26729,-5.2e-05],[0.000991,-0.068615,0.013433,0.022816,-0.010602,-0.023735,0.075316,0.080493,-0.022107,-0.035283,0.104859,0.089171,0.079677,-0.000384,0.003971,0.001326,-25.275299,0.000818],[0.005038,-0.008724,-0.02988,0.093367,0.060983,-0.114081,-0.094976,-0.03181,0.005656,0.026607,0.016112,0.001375,-0.04097,0.004671,0.002891,-0.000213,13.219302,0.000653],[-0.001062,0.041872,0.015016,-0.026876,-0.039293,0.002118,-0.050102,-0.043742,0.023799,0.040627,-0.061065,0.052712,0.005491,-0.003213,-0.002804,-0.000414,-12.730994,-0.000543],[-0.000543,0.0734,-0.016864,0.150016,0.040193,0.097927,0.018367,-0.015052,-0.007359,0.01147,0.026705,-0.022768,-0.049143,0.000414,0.000378,8.7e-05,-20.596242,2.5e-05],[0.001235,0.00921,0.015617,-0.053793,0.002945,0.104576,0.092264,0.014345,0.002835,-0.036752,-0.022324,-0.070559,-0.077944,-0.000627,0.000554,-0.000204,-8.456561,-0.000855],[-0.002867,-0.040575,-0.000659,-0.024302,-0.06701,-0.216079,0.052874,0.055063,0.061815,0.008268,-0.006847,-0.062576,-0.007788,0.001047,-0.00138,-0.000124,18.206647,-7e-06],[-0.003005,0.059757,-0.009595,-0.090703,0.009051,0.029645,-0.041462,-0.045257,-0.069675,-0.026266,-0.018678,0.087961,0.053757,-0.000447,-0.001651,-0.001073,-21.805636,-0.000533]],"b1":[-1.824042,-0.720319,-3.35861,2.762304,0.109944,3.291967,-0.97603,-0.879878],"W2":[[0.624192,-0.265713,-0.284184,0.77412,-0.20408,-0.054247,-0.708038,1.076569],[-0.005041,-1.012945,0.03872,0.656813,-1.037005,-0.165812,0.775248,-0.815089],[0.150324,-0.181244,0.620412,-0.748799,0.505443,-0.251354,-0.82388,0.109505],[0.371286,0.960335,-0.483184,-1.026346,-0.525608,0.360379,0.029975,-0.031824],[-0.502478,-0.691489,-0.653868,0.90623,0.773134,1.09747,-0.775315,-0.58894]],"b2":[-0.389358,0.734332,0.232886,-0.247158,-0.231752]};
/*WEIGHTS-END*/
