// calibration.js — ApisSonic Phase 4
// Hive wall material transfer functions, temperature / time-of-day priors, mic noise floor.

export const MATERIALS = {
  landing_board: { label: 'Landing board / entrance', low: 0.0, swarm: 0.0, piping: 0.0, high: 0.0 },
  wood_langstroth: { label: 'Wood Langstroth wall', low: 0.5, swarm: 1.5, piping: 2.8, high: 3.8 },
  polystyrene_eps: { label: 'Polystyrene (EPS) wall', low: 1.2, swarm: 3.2, piping: 5.8, high: 7.5 },
  thick_timber: { label: 'Thick timber / log hive', low: 0.8, swarm: 2.2, piping: 4.0, high: 5.2 },
};

/** Which material gain applies to which diagnostic band. */
export const BAND_GAIN_KEY = {
  fanning: 'low',
  healthy: 'low',
  preswarm: 'swarm',
  pipingFund: 'piping',
  pipingHarm: 'high',
  pipingGap: 'high',
  queenless: 'high',
};

export function getMaterial(id) {
  return MATERIALS[id] || MATERIALS.landing_board;
}

/** Add the material's dB compensation to raw band powers (returns a new object). */
export function applyMaterialToBands(bands, materialId) {
  const m = getMaterial(materialId);
  const out = {};
  for (const k of Object.keys(bands)) out[k] = bands[k] + (BAND_GAIN_KEY[k] ? m[BAND_GAIN_KEY[k]] : 0);
  return out;
}

/** Smooth per-frequency version of the same curve (used for the spectrum fed to the MFCCs). */
export function materialOffsetDb(materialId, hz) {
  const m = getMaterial(materialId);
  const pts = [[250, m.low], [360, m.swarm], [475, m.piping], [775, m.high]];
  if (hz <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (hz <= pts[i][0]) {
      const t = (hz - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
      return pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]);
    }
  }
  return pts[pts.length - 1][1];
}

export function applyMaterialToSpectrum(db, sampleRate, materialId, out = new Float32Array(db.length)) {
  const bw = sampleRate / (db.length * 2);
  for (let k = 0; k < db.length; k++) out[k] = db[k] + materialOffsetDb(materialId, k * bw);
  return out;
}

// ------------------------------------------------------------ mic noise floor

/** Accumulates an ambient spectrum (hive closed / away from hive) to subtract later. */
export class MicFloor {
  constructor() {
    this.floor = null; // linear power per bin
    this._sum = null;
    this._n = 0;
  }
  begin(bins) {
    this._sum = new Float64Array(bins);
    this._n = 0;
  }
  add(db) {
    if (!this._sum) return;
    for (let k = 0; k < db.length; k++) this._sum[k] += Math.pow(10, db[k] / 10);
    this._n++;
  }
  finish() {
    if (this._sum && this._n > 0) {
      this.floor = new Float64Array(this._sum.length);
      for (let k = 0; k < this._sum.length; k++) this.floor[k] = this._sum[k] / this._n;
    }
    this._sum = null;
    return this.floor !== null;
  }
  clear() {
    this.floor = null;
  }
}

/** Spectral subtraction, never removing more than 10 dB from any bin. */
export function subtractFloor(db, floor, out = new Float32Array(db.length)) {
  if (!floor || floor.length !== db.length) {
    if (out !== db) out.set(db);
    return out;
  }
  for (let k = 0; k < db.length; k++) {
    const p = Math.pow(10, db[k] / 10);
    out[k] = 10 * Math.log10(Math.max(p - floor[k], 0.1 * p) + 1e-30);
  }
  return out;
}

// ------------------------------------------------- temperature & time priors

export const BASE_PRIORS = { healthy: 0.4, preswarm: 0.15, queenless: 0.15, piping: 0.15, fanning: 0.15 };

/** Multipliers applied to the base priors. timeOfDay: 'day' | 'night'. */
export function environmentMultipliers(ambientTempC, timeOfDay) {
  const m = { healthy: 1, preswarm: 1, queenless: 1, piping: 1, fanning: 1 };
  const T = Number(ambientTempC);
  if (Number.isFinite(T)) {
    if (T >= 30) m.fanning *= Math.min(1.65, 1.0 + (T - 28) * 0.055);
    else if (T <= 14) {
      m.fanning *= 0.65;
      m.queenless *= 1.15;
    }
  }
  if (timeOfDay === 'night') {
    m.piping *= 1.22;
    m.queenless *= 1.18;
  }
  return m;
}

/** Normalised priors in the order of `labels`. */
export function environmentPriors(labels, ambientTempC, timeOfDay) {
  const m = environmentMultipliers(ambientTempC, timeOfDay);
  const p = labels.map((l) => BASE_PRIORS[l] * m[l]);
  const s = p.reduce((a, b) => a + b, 0);
  return p.map((v) => v / s);
}

export function autoTimeOfDay(date = new Date()) {
  const h = date.getHours();
  return h < 6 || h >= 20 ? 'night' : 'day';
}
