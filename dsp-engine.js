// dsp-engine.js — ApisSonic Phase 1 & 2
// Web Audio graph, 4096-pt FFT, 5-band integration, transient rejection, 15 s scan,
// plus an offline analyser + colony synthesizer so the same maths runs in Node tests.
//
// NOTE: every frequency calculation uses the *actual* AudioContext sample rate.
// iPhones normally run at 48 kHz, not the nominal 44.1 kHz of the blueprint.

export const FFT_SIZE = 4096;
export const SMOOTHING = 0.75;
export const NOMINAL_SAMPLE_RATE = 44100;
export const ANALYSIS_HOP_MS = 50;
export const SCAN_SECONDS = 15;

export const BANDS = {
  fanning: [120, 200],
  healthy: [200, 290],
  preswarm: [300, 420],
  pipingFund: [410, 540],
  pipingHarm: [800, 960],
  queenless: [550, 1000],
  pipingGap: [570, 780],
  windRumble: [25, 85],
  clatter: [1250, 3500],
};

export const ARTIFACT = {
  windDb: 8,
  clatterDb: 6,
  rmsJumpDb: 9.5,
  crest: 6.5,
  quietDb: -80,
};

// ---------------------------------------------------------------- pure maths

export function binWidth(sampleRate, fftSize = FFT_SIZE) {
  return sampleRate / fftSize;
}

/** Mean linear power across [minHz, maxHz], returned in dB. `db` is an analyser dB spectrum. */
export function computeBandPower(db, sampleRate, minHz, maxHz) {
  const bw = sampleRate / (db.length * 2);
  let k0 = Math.max(0, Math.ceil(minHz / bw));
  let k1 = Math.min(db.length - 1, Math.floor(maxHz / bw));
  if (k1 < k0) k1 = k0;
  let sum = 0;
  for (let k = k0; k <= k1; k++) sum += Math.pow(10, db[k] / 10);
  return 10 * Math.log10(sum / (k1 - k0 + 1) + 1e-30);
}

export function computeAllBands(db, sampleRate) {
  const out = {};
  for (const name of ['fanning', 'healthy', 'preswarm', 'pipingFund', 'pipingHarm', 'queenless', 'pipingGap']) {
    out[name] = computeBandPower(db, sampleRate, BANDS[name][0], BANDS[name][1]);
  }
  return out;
}

export function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
}

export function crestFactor(x) {
  const r = rms(x);
  if (r < 1e-9) return 0;
  let p = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    if (a > p) p = a;
  }
  return p / r;
}

/** Peak frequency in [minHz, maxHz] with parabolic interpolation on the dB values. */
export function dominantFrequency(db, sampleRate, minHz = 60, maxHz = 1000) {
  const bw = sampleRate / (db.length * 2);
  const k0 = Math.max(1, Math.ceil(minHz / bw));
  const k1 = Math.min(db.length - 2, Math.floor(maxHz / bw));
  let best = k0;
  for (let k = k0; k <= k1; k++) if (db[k] > db[best]) best = k;
  const a = db[best - 1], b = db[best], c = db[best + 1];
  let delta = 0;
  const denom = a - 2 * b + c;
  if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && Math.abs(denom) > 1e-9) {
    delta = 0.5 * (a - c) / denom;
    if (delta > 0.5) delta = 0.5;
    if (delta < -0.5) delta = -0.5;
  }
  return (best + delta) * bw;
}

/** Power-weighted mean frequency over [minHz, maxHz]. */
export function spectralCentroid(db, sampleRate, minHz = 60, maxHz = 2000) {
  const bw = sampleRate / (db.length * 2);
  const k0 = Math.max(0, Math.ceil(minHz / bw));
  const k1 = Math.min(db.length - 1, Math.floor(maxHz / bw));
  let num = 0, den = 0;
  for (let k = k0; k <= k1; k++) {
    const p = Math.pow(10, db[k] / 10);
    num += p * k * bw;
    den += p;
  }
  return den > 0 ? num / den : 0;
}

// ------------------------------------------------------- transient rejection

export class ArtifactDetector {
  constructor() {
    this.baselineDb = null;
  }
  reset() {
    this.baselineDb = null;
  }
  /** Returns { isArtifact, wind, bump, clatter, quiet, crest, rmsDb }. */
  check(db, timeData, sampleRate) {
    const ref = computeBandPower(db, sampleRate, BANDS.healthy[0], BANDS.healthy[1]);
    const wind = computeBandPower(db, sampleRate, BANDS.windRumble[0], BANDS.windRumble[1]) > ref + ARTIFACT.windDb;
    const clatter = computeBandPower(db, sampleRate, BANDS.clatter[0], BANDS.clatter[1]) > ref + ARTIFACT.clatterDb;
    const r = rms(timeData);
    const rmsDb = 20 * Math.log10(r + 1e-12);
    const crest = crestFactor(timeData);
    let bump = crest > ARTIFACT.crest;
    if (this.baselineDb !== null && rmsDb > this.baselineDb + ARTIFACT.rmsJumpDb) bump = true;
    const quiet = rmsDb < ARTIFACT.quietDb;
    const isArtifact = wind || bump || clatter;
    // Rolling baseline: follow clean frames quickly, artifact frames slowly (so a
    // genuine level change cannot lock the detector on forever).
    if (this.baselineDb === null) this.baselineDb = rmsDb;
    else this.baselineDb += (isArtifact ? 0.02 : 0.1) * (rmsDb - this.baselineDb);
    return { isArtifact, wind, bump, clatter, quiet, crest, rmsDb };
  }
}

// ------------------------------------------------------------ 15 s timed scan

export class ScanAccumulator {
  constructor(durationMs = SCAN_SECONDS * 1000) {
    this.durationMs = durationMs;
    this.reset();
  }
  reset() {
    this.startedAt = null;
    this.hz = [];
    this.probSum = null;
    this.cleanFrames = 0;
    this.artifactFrames = 0;
    this.quietFrames = 0;
  }
  start(now) {
    this.reset();
    this.startedAt = now;
  }
  get active() {
    return this.startedAt !== null;
  }
  /** Add one analysis frame. Returns progress 0..1 (>= 1 means finished). */
  add(frame, now) {
    if (this.startedAt === null) return 0;
    if (frame.isArtifact) this.artifactFrames++;
    else if (frame.quiet) this.quietFrames++;
    else {
      this.cleanFrames++;
      this.hz.push(frame.dominantHz);
      if (!this.probSum) this.probSum = new Array(frame.probs.length).fill(0);
      for (let i = 0; i < frame.probs.length; i++) this.probSum[i] += frame.probs[i];
    }
    return (now - this.startedAt) / this.durationMs;
  }
  result(labels) {
    const n = this.cleanFrames;
    let mean = 0, sd = 0;
    if (n > 0) {
      for (const h of this.hz) mean += h;
      mean /= n;
      for (const h of this.hz) sd += (h - mean) * (h - mean);
      sd = Math.sqrt(sd / n);
    }
    const probs = this.probSum ? this.probSum.map((p) => p / n) : null;
    let top = -1;
    if (probs) {
      top = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
    }
    return {
      meanHz: mean,
      sdHz: sd,
      cleanFrames: n,
      artifactFrames: this.artifactFrames,
      quietFrames: this.quietFrames,
      probs,
      state: top >= 0 && labels ? labels[top] : null,
      confidence: top >= 0 ? probs[top] : 0,
    };
  }
}

// --------------------------------------- offline analyser (AnalyserNode twin)

function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Reproduces Web Audio AnalyserNode: Blackman window, |X|/N, time smoothing, 20·log10. */
export class OfflineAnalyser {
  constructor(fftSize = FFT_SIZE, smoothing = SMOOTHING) {
    this.fftSize = fftSize;
    this.smoothing = smoothing;
    this.window = new Float64Array(fftSize);
    for (let n = 0; n < fftSize; n++) {
      this.window[n] = 0.42 - 0.5 * Math.cos((2 * Math.PI * n) / fftSize) + 0.08 * Math.cos((4 * Math.PI * n) / fftSize);
    }
    this.prev = new Float64Array(fftSize / 2);
    this.re = new Float64Array(fftSize);
    this.im = new Float64Array(fftSize);
  }
  reset() {
    this.prev.fill(0);
  }
  /** frame: fftSize time samples → Float32Array of dB values (fftSize / 2 bins). */
  process(frame) {
    const N = this.fftSize;
    for (let n = 0; n < N; n++) {
      this.re[n] = frame[n] * this.window[n];
      this.im[n] = 0;
    }
    fftInPlace(this.re, this.im);
    const out = new Float32Array(N / 2);
    for (let k = 0; k < N / 2; k++) {
      const mag = Math.hypot(this.re[k], this.im[k]) / N;
      const s = this.smoothing * this.prev[k] + (1 - this.smoothing) * mag;
      this.prev[k] = s;
      out[k] = 20 * Math.log10(s);
    }
    return out;
  }
}

// --------------------------------------------- colony bioacoustic synthesizer

export const SYNTH_STATES = ['healthy', 'preswarm', 'queenless', 'piping', 'fanning'];

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function seededRandom(seed) {
  return mulberry32(seed);
}

function pinkNoise(n, sampleRate, rng) {
  // Paul Kellet pink filter followed by a one-pole 120 Hz high-pass (phone mics
  // roll off the lows, and it keeps the synth clear of the wind-rumble detector).
  const out = new Float32Array(n);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  const rc = 1 / (2 * Math.PI * 120);
  const a = rc / (rc + 1 / sampleRate);
  let px = 0, py = 0, sq = 0;
  for (let i = 0; i < n; i++) {
    const w = rng() * 2 - 1;
    b0 = 0.99886 * b0 + w * 0.0555179;
    b1 = 0.99332 * b1 + w * 0.0750759;
    b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856;
    b4 = 0.55 * b4 + w * 0.5329522;
    b5 = -0.7616 * b5 - w * 0.016898;
    const x = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
    b6 = w * 0.115926;
    const y = a * (py + x - px);
    px = x; py = y;
    out[i] = y;
    sq += y * y;
  }
  const norm = 1 / (Math.sqrt(sq / n) + 1e-12);
  for (let i = 0; i < n; i++) out[i] *= norm;
  return out;
}

const uni = (rng, lo, hi) => lo + (hi - lo) * rng();

/**
 * Synthesize a colony sound: harmonic hum + LFO modulation + pink noise.
 * Recipes are mirrored exactly in ml/train_hive_classifier.py.
 */
export function synthesizeColony(state, seconds, sampleRate, rng = Math.random, gain = 0.2) {
  const n = Math.round(seconds * sampleRate);
  const out = new Float32Array(n);
  const TAU = 2 * Math.PI;
  const dt = 1 / sampleRate;

  const addTone = (f0, harm, amp, opt = {}) => {
    const amRate = opt.amRate || 0, amDepth = opt.amDepth || 0;
    const fmRate = opt.fmRate || 0, fmDev = opt.fmDev || 0;
    const pulseRate = opt.pulseRate || 0;
    const ph0 = harm.map(() => rng() * TAU);
    const amPh = rng() * TAU, fmPh = rng() * TAU, pPh = rng() * TAU;
    let phase = 0;
    for (let i = 0; i < n; i++) {
      const t = i * dt;
      const f = f0 + (fmDev ? fmDev * Math.sin(TAU * fmRate * t + fmPh) : 0);
      phase += TAU * f * dt;
      let env = 1 + (amDepth ? amDepth * Math.sin(TAU * amRate * t + amPh) : 0);
      if (pulseRate) {
        const p = 0.5 + 1.5 * Math.sin(TAU * pulseRate * t + pPh);
        env *= 0.35 + 0.65 * Math.min(1, Math.max(0, p));
      }
      let s = 0;
      for (let h = 0; h < harm.length; h++) s += harm[h] * Math.sin((h + 1) * phase + ph0[h]);
      out[i] += amp * env * s;
    }
  };
  const addNoise = (level) => {
    const p = pinkNoise(n, sampleRate, rng);
    for (let i = 0; i < n; i++) out[i] += level * p[i];
  };

  switch (state) {
    case 'fanning':
      addTone(uni(rng, 140, 185), [1, 0.2, 0.06], 1, { amRate: uni(rng, 3, 5), amDepth: 0.25 });
      addNoise(0.1);
      break;
    case 'preswarm':
      addTone(uni(rng, 325, 395), [1, 0.3, 0.1], 1, {
        amRate: uni(rng, 1, 2), amDepth: 0.2, fmRate: uni(rng, 5, 8), fmDev: 8,
      });
      addTone(uni(rng, 230, 250), [1], 0.2);
      addNoise(0.12);
      break;
    case 'piping':
      addTone(uni(rng, 230, 250), [1, 0.2], 0.25);
      addTone(uni(rng, 430, 478), [1, 0.7, 0.15], 1, { pulseRate: uni(rng, 2.5, 4) });
      addNoise(0.08);
      break;
    case 'queenless': {
      addTone(uni(rng, 210, 255), [1, 0.2], 0.3, { amRate: uni(rng, 0.5, 1.2), amDepth: 0.1 });
      const K = 48, amp = Math.sqrt(2 / K);
      for (let j = 0; j < K; j++) {
        const f = uni(rng, 550, 1000), ph = rng() * TAU;
        for (let i = 0; i < n; i++) out[i] += amp * Math.sin(TAU * f * i * dt + ph);
      }
      addNoise(0.12);
      break;
    }
    case 'healthy':
    default:
      addTone(uni(rng, 225, 265), [1, 0.25, 0.08], 1, { amRate: uni(rng, 0.6, 1.5), amDepth: 0.15 });
      addNoise(0.1);
  }
  for (let i = 0; i < n; i++) out[i] *= gain;
  return out;
}

// ------------------------------------------------------- live Web Audio engine

const TAP_WORKLET = `
class ApisTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('apis-tap', ApisTap);`;

export class DSPEngine {
  /** onFrame({ db, time, sampleRate, artifact }) fires every ANALYSIS_HOP_MS while running. */
  constructor({ onFrame, onEnded } = {}) {
    this.onFrame = onFrame || (() => {});
    this.onEnded = onEnded || (() => {});
    this.ctx = null;
    this.analyser = null;
    this.detector = new ArtifactDetector();
    this.sourceNode = null;
    this.stream = null;
    this.timer = null;
    this.mode = null;
    this.recording = null;
    this.tapReady = false;
    this.tapNode = null;
  }

  get sampleRate() {
    return this.ctx ? this.ctx.sampleRate : NOMINAL_SAMPLE_RATE;
  }
  get running() {
    return this.timer !== null;
  }

  /** Must be called from a user gesture (iOS requirement). */
  ensureContext() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
    }
    const resumed = this.ctx.state === 'suspended' ? this.ctx.resume() : Promise.resolve();
    if (!this.analyser) {
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = FFT_SIZE;
      this.analyser.smoothingTimeConstant = SMOOTHING;
      this.analyser.minDecibels = -120;
      this.analyser.maxDecibels = 0;
      this.db = new Float32Array(this.analyser.frequencyBinCount);
      this.time = new Float32Array(FFT_SIZE);
      this.silent = this.ctx.createGain();
      this.silent.gain.value = 0;
      this.silent.connect(this.ctx.destination);
      this.monitor = this.ctx.createGain();
      this.monitor.gain.value = 0.6;
      this.monitor.connect(this.ctx.destination);
    }
    return resumed;
  }

  async _ensureTap() {
    if (this.tapNode) return;
    const onChunk = (chunk) => {
      if (this.recording) this.recording.push(chunk);
    };
    try {
      if (this.ctx.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
        const url = URL.createObjectURL(new Blob([TAP_WORKLET], { type: 'application/javascript' }));
        await this.ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        this.tapNode = new AudioWorkletNode(this.ctx, 'apis-tap', { numberOfInputs: 1, numberOfOutputs: 1 });
        this.tapNode.port.onmessage = (e) => onChunk(e.data);
      }
    } catch (err) {
      this.tapNode = null;
    }
    if (!this.tapNode) {
      // Fallback for browsers without AudioWorklet.
      const sp = this.ctx.createScriptProcessor(4096, 1, 1);
      sp.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
      this.tapNode = sp;
    }
    this.analyser.connect(this.tapNode);
    this.tapNode.connect(this.silent);
  }

  async _begin(node, mode, audible) {
    this.sourceNode = node;
    this.mode = mode;
    node.connect(this.analyser);
    if (audible) node.connect(this.monitor);
    await this._ensureTap();
    this.detector.reset();
    this.timer = setInterval(() => this._tick(), ANALYSIS_HOP_MS);
  }

  async startMic() {
    const resumed = this.ensureContext();
    this.stop();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Microphone needs HTTPS (or localhost). Open the app from an https:// address.');
    }
    // Voice processing must be OFF or the phone filters the steady colony hum as "noise".
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    await resumed;
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this.stream = stream;
    const track = stream.getAudioTracks()[0];
    this.micSettings = track && track.getSettings ? track.getSettings() : {};
    await this._begin(this.ctx.createMediaStreamSource(stream), 'mic', false);
  }

  async startFile(file) {
    const resumed = this.ensureContext();
    this.stop();
    const bytes = await file.arrayBuffer();
    await resumed;
    const buffer = await new Promise((res, rej) => {
      const p = this.ctx.decodeAudioData(bytes, res, rej);
      if (p && p.then) p.then(res, rej);
    });
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.onended = () => {
      if (this.sourceNode === src) {
        this.onEnded(); // lets the app collect an in-progress recording first
        this.stop();
      }
    };
    await this._begin(src, 'file', true);
    src.start();
  }

  async startSimulator(state) {
    const resumed = this.ensureContext();
    this.stop();
    await resumed;
    const sr = this.ctx.sampleRate;
    const data = synthesizeColony(state, 20, sr);
    const fade = Math.round(0.05 * sr);
    for (let i = 0; i < fade; i++) {
      const g = i / fade;
      data[i] *= g;
      data[data.length - 1 - i] *= g;
    }
    const buffer = this.ctx.createBuffer(1, data.length, sr);
    buffer.getChannelData(0).set(data);
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    await this._begin(src, 'sim', true);
    src.start();
  }

  stop() {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.sourceNode) {
      try { this.sourceNode.onended = null; } catch (e) { /* not a buffer source */ }
      try { this.sourceNode.stop && this.sourceNode.stop(); } catch (e) { /* already stopped */ }
      try { this.sourceNode.disconnect(); } catch (e) { /* ignore */ }
      this.sourceNode = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.recording = null;
    this.mode = null;
  }

  startRecording() {
    this.recording = [];
  }
  /** Returns the Float32Array chunks captured since startRecording(). */
  stopRecording() {
    const chunks = this.recording || [];
    this.recording = null;
    return chunks;
  }

  _tick() {
    if (!this.analyser) return;
    this.analyser.getFloatFrequencyData(this.db);
    if (this.analyser.getFloatTimeDomainData) this.analyser.getFloatTimeDomainData(this.time);
    else {
      const bytes = new Uint8Array(FFT_SIZE);
      this.analyser.getByteTimeDomainData(bytes);
      for (let i = 0; i < FFT_SIZE; i++) this.time[i] = (bytes[i] - 128) / 128;
    }
    const sr = this.ctx.sampleRate;
    const artifact = this.detector.check(this.db, this.time, sr);
    this.onFrame({ db: this.db, time: this.time, sampleRate: sr, artifact });
  }
}
