// qr-report.js — ApisSonic Phase 5
// URL deep-links, QR generator (SVG), apiary triage ranking, printable report.
//
// QR note: a 21×21 (version 1) code holds at most 17 bytes, which is too small for
// a full https:// deep-link. The generator therefore picks the smallest version
// (1–6, i.e. 21×21 up to 41×41) that fits, so short payloads are still 21×21.

// ------------------------------------------------------------------ deep links

export const DEEP_LINK_KEYS = ['hive', 'material', 'temp'];

/** Parse "?hive=..&material=..&temp=.." → { hive, material, temp } (missing keys omitted). */
export function parseDeepLink(search) {
  const out = {};
  const q = new URLSearchParams(String(search || '').replace(/^[^?]*\?/, ''));
  const hive = q.get('hive');
  if (hive) out.hive = hive.slice(0, 40);
  const material = q.get('material');
  if (material) out.material = material;
  const temp = q.get('temp');
  if (temp !== null && temp !== '' && Number.isFinite(Number(temp))) out.temp = Number(temp);
  return out;
}

export function buildDeepLink(baseUrl, { hive, material, temp }) {
  const q = new URLSearchParams();
  if (hive) q.set('hive', hive);
  if (material) q.set('material', material);
  if (temp !== undefined && temp !== null && temp !== '') q.set('temp', String(temp));
  return String(baseUrl).replace(/[?#].*$/, '') + '?' + q.toString();
}

// ------------------------------------------------------------------ QR encoder

const TOTAL_CODEWORDS = { 1: 26, 2: 44, 3: 70, 4: 100, 5: 134, 6: 172 };
// [error-correction codewords per block, number of blocks]
const EC_TABLE = {
  M: { 1: [10, 1], 2: [16, 1], 3: [26, 1], 4: [18, 2], 5: [24, 2], 6: [16, 4] },
  L: { 1: [7, 1], 2: [10, 1], 3: [15, 1], 4: [20, 1], 5: [26, 1], 6: [18, 2] },
};
const FORMAT_BITS = { L: 1, M: 0 };

const GF_EXP = new Uint8Array(512);
const GF_LOG = new Uint8Array(256);
(() => {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    GF_EXP[i] = x;
    GF_LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
})();
const gfMul = (a, b) => (a && b ? GF_EXP[GF_LOG[a] + GF_LOG[b]] : 0);

function rsGenerator(degree) {
  let g = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array(g.length + 1).fill(0);
    for (let j = 0; j < g.length; j++) {
      next[j] ^= g[j];
      next[j + 1] ^= gfMul(g[j], GF_EXP[i]);
    }
    g = next;
  }
  return g;
}

function rsRemainder(data, gen) {
  const n = gen.length - 1;
  const res = new Array(n).fill(0);
  for (const b of data) {
    const factor = b ^ res.shift();
    res.push(0);
    for (let j = 0; j < n; j++) res[j] ^= gfMul(gen[j + 1], factor);
  }
  return res;
}

function utf8Bytes(text) {
  if (typeof TextEncoder !== 'undefined') return Array.from(new TextEncoder().encode(text));
  return Array.from(unescape(encodeURIComponent(text)), (c) => c.charCodeAt(0));
}

function chooseVersion(byteLen) {
  for (let v = 1; v <= 6; v++) {
    for (const level of ['M', 'L']) {
      const [ec, blocks] = EC_TABLE[level][v];
      const dataCw = TOTAL_CODEWORDS[v] - ec * blocks;
      if (byteLen <= dataCw - 2) return { version: v, level, ec, blocks, dataCw };
    }
  }
  throw new Error('Text is too long for a hive tag QR code (max 134 bytes). Shorten the Hive ID.');
}

function buildCodewords(bytes, cfg) {
  const bits = [];
  const push = (val, len) => {
    for (let i = len - 1; i >= 0; i--) bits.push((val >>> i) & 1);
  };
  push(0b0100, 4); // byte mode
  push(bytes.length, 8);
  for (const b of bytes) push(b, 8);
  const capacity = cfg.dataCw * 8;
  push(0, Math.min(4, capacity - bits.length));
  while (bits.length % 8) bits.push(0);
  const data = [];
  for (let i = 0; i < bits.length; i += 8) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
    data.push(b);
  }
  for (let pad = 0xec; data.length < cfg.dataCw; pad ^= 0xec ^ 0x11) data.push(pad);

  const gen = rsGenerator(cfg.ec);
  const blockLen = cfg.dataCw / cfg.blocks;
  const dataBlocks = [], ecBlocks = [];
  for (let b = 0; b < cfg.blocks; b++) {
    const d = data.slice(b * blockLen, (b + 1) * blockLen);
    dataBlocks.push(d);
    ecBlocks.push(rsRemainder(d, gen));
  }
  const out = [];
  for (let i = 0; i < blockLen; i++) for (const d of dataBlocks) out.push(d[i]);
  for (let i = 0; i < cfg.ec; i++) for (const e of ecBlocks) out.push(e[i]);
  return out;
}

const MASKS = [
  (x, y) => (x + y) % 2 === 0,
  (x, y) => y % 2 === 0,
  (x, y) => x % 3 === 0,
  (x, y) => (x + y) % 3 === 0,
  (x, y) => (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0,
  (x, y) => ((x * y) % 2) + ((x * y) % 3) === 0,
  (x, y) => (((x * y) % 2) + ((x * y) % 3)) % 2 === 0,
  (x, y) => (((x + y) % 2) + ((x * y) % 3)) % 2 === 0,
];

function penalty(m) {
  const n = m.length;
  let score = 0;
  const line = (get) => {
    for (let a = 0; a < n; a++) {
      let run = 1;
      for (let b = 1; b <= n; b++) {
        if (b < n && get(a, b) === get(a, b - 1)) run++;
        else {
          if (run >= 5) score += run - 2;
          run = 1;
        }
      }
      // finder-like 1:1:3:1:1 patterns with 4 light modules on one side
      for (let b = 0; b + 10 < n; b++) {
        const s = [];
        for (let i = 0; i < 11; i++) s.push(get(a, b + i) ? 1 : 0);
        const str = s.join('');
        if (str === '10111010000' || str === '00001011101') score += 40;
      }
    }
  };
  line((a, b) => m[a][b]);
  line((a, b) => m[b][a]);
  let dark = 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (m[y][x]) dark++;
      if (x + 1 < n && y + 1 < n && m[y][x] === m[y][x + 1] && m[y][x] === m[y + 1][x] && m[y][x] === m[y + 1][x + 1]) score += 3;
    }
  }
  score += Math.floor(Math.abs((dark * 20) / (n * n) - 10)) * 10;
  return score;
}

/** Encode text → { size, version, level, mask, modules: boolean[size][size] } (true = dark). */
export function generateQRMatrix(text) {
  const bytes = utf8Bytes(text);
  const cfg = chooseVersion(bytes.length);
  const size = 17 + 4 * cfg.version;
  const mod = Array.from({ length: size }, () => new Array(size).fill(false));
  const func = Array.from({ length: size }, () => new Array(size).fill(false));
  const set = (x, y, dark) => {
    if (x < 0 || y < 0 || x >= size || y >= size) return;
    mod[y][x] = dark;
    func[y][x] = true;
  };

  for (let i = 0; i < size; i++) {
    set(6, i, i % 2 === 0);
    set(i, 6, i % 2 === 0);
  }
  for (const [cx, cy] of [[3, 3], [size - 4, 3], [3, size - 4]]) {
    for (let dy = -4; dy <= 4; dy++) {
      for (let dx = -4; dx <= 4; dx++) {
        const d = Math.max(Math.abs(dx), Math.abs(dy));
        set(cx + dx, cy + dy, d !== 2 && d !== 4);
      }
    }
  }
  if (cfg.version >= 2) {
    const c = size - 7;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) set(c + dx, c + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
  }
  const drawFormat = (mask) => {
    const data = (FORMAT_BITS[cfg.level] << 3) | mask;
    let rem = data;
    for (let i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >>> 9) * 0x537);
    const bits = ((data << 10) | rem) ^ 0x5412;
    const bit = (i) => ((bits >>> i) & 1) !== 0;
    for (let i = 0; i <= 5; i++) set(8, i, bit(i));
    set(8, 7, bit(6));
    set(8, 8, bit(7));
    set(7, 8, bit(8));
    for (let i = 9; i < 15; i++) set(14 - i, 8, bit(i));
    for (let i = 0; i < 8; i++) set(size - 1 - i, 8, bit(i));
    for (let i = 8; i < 15; i++) set(8, size - 15 + i, bit(i));
    set(8, size - 8, true);
  };
  drawFormat(0); // reserve the format areas before placing data

  const codewords = buildCodewords(bytes, cfg);
  const bits = [];
  for (const cw of codewords) for (let i = 7; i >= 0; i--) bits.push((cw >>> i) & 1);
  let idx = 0;
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5;
    for (let vert = 0; vert < size; vert++) {
      for (let j = 0; j < 2; j++) {
        const x = right - j;
        const upward = ((right + 1) & 2) === 0;
        const y = upward ? size - 1 - vert : vert;
        if (!func[y][x] && idx < bits.length) mod[y][x] = bits[idx++] === 1;
      }
    }
  }

  const applyMask = (k) => {
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (!func[y][x] && MASKS[k](x, y)) mod[y][x] = !mod[y][x];
  };
  let best = 0, bestScore = Infinity;
  for (let k = 0; k < 8; k++) {
    applyMask(k);
    drawFormat(k);
    const s = penalty(mod);
    if (s < bestScore) {
      bestScore = s;
      best = k;
    }
    applyMask(k); // undo
  }
  applyMask(best);
  drawFormat(best);
  return { size, version: cfg.version, level: cfg.level, mask: best, modules: mod };
}

/** Render a QR matrix as a standalone SVG string (with a 4-module quiet zone). */
export function qrToSVG(qr, { pixels = 240, margin = 4, dark = '#000', light = '#fff' } = {}) {
  const n = qr.size + margin * 2;
  let d = '';
  for (let y = 0; y < qr.size; y++) {
    for (let x = 0; x < qr.size; x++) if (qr.modules[y][x]) d += `M${x + margin},${y + margin}h1v1h-1z`;
  }
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${n} ${n}" width="${pixels}" height="${pixels}" shape-rendering="crispEdges" role="img" aria-label="Hive QR tag">` +
    `<rect width="${n}" height="${n}" fill="${light}"/><path d="${d}" fill="${dark}"/></svg>`
  );
}

// ------------------------------------------------------------------ triage

export const SEVERITY_RANK = { queenless: 1, preswarm: 2, piping: 3, fanning: 4, healthy: 5 };

/**
 * Group inspection logs by hiveId → one card per hive, sorted most urgent first.
 * Each card: { hiveId, state, rank, confidence, meanHz, driftHz, count, latest, history }.
 */
export function buildTriage(inspections) {
  const byHive = new Map();
  for (const rec of inspections) {
    const id = rec.hiveId || 'Unnamed';
    if (!byHive.has(id)) byHive.set(id, []);
    byHive.get(id).push(rec);
  }
  const cards = [];
  for (const [hiveId, list] of byHive) {
    list.sort((a, b) => b.id - a.id);
    const latest = list[0], previous = list[1];
    cards.push({
      hiveId,
      state: latest.state,
      rank: SEVERITY_RANK[latest.state] || 5,
      confidence: latest.confidence,
      meanHz: latest.meanHz,
      driftHz: previous ? latest.meanHz - previous.meanHz : null,
      count: list.length,
      latest,
      history: list,
    });
  }
  cards.sort((a, b) => a.rank - b.rank || b.confidence - a.confidence || a.hiveId.localeCompare(b.hiveId));
  return cards;
}

// ------------------------------------------------------------------ report

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** HTML fragment for the print view. `names` maps state id → display name. */
export function buildReportHTML(inspections, names = {}, title = 'ApisSonic Apiary Report') {
  const cards = buildTriage(inspections);
  const fmtDate = (t) => new Date(t).toLocaleString();
  let html = `<h1>${esc(title)}</h1><p class="rep-meta">Generated ${esc(fmtDate(Date.now()))} · ${cards.length} hive(s) · ${inspections.length} inspection(s)</p>`;
  html += '<h2>Yard triage (most urgent first)</h2><table><thead><tr><th>#</th><th>Hive</th><th>Status</th><th>Confidence</th><th>Mean Hz</th><th>Drift ΔHz</th><th>Scans</th><th>Last scan</th></tr></thead><tbody>';
  for (const c of cards) {
    const drift = c.driftHz === null ? '—' : (c.driftHz >= 0 ? '+' : '') + c.driftHz.toFixed(1);
    html += `<tr><td>${c.rank}</td><td>${esc(c.hiveId)}</td><td>${esc(names[c.state] || c.state)}</td><td>${Math.round(c.confidence * 100)}%</td><td>${c.meanHz.toFixed(1)}</td><td>${drift}</td><td>${c.count}</td><td>${esc(fmtDate(c.latest.id))}</td></tr>`;
  }
  html += '</tbody></table><h2>Inspection log</h2><table><thead><tr><th>Date</th><th>Hive</th><th>Status</th><th>Mean Hz ± SD</th><th>Clean / rejected frames</th><th>Material</th><th>Temp °C</th></tr></thead><tbody>';
  for (const r of [...inspections].sort((a, b) => b.id - a.id)) {
    html += `<tr><td>${esc(fmtDate(r.id))}</td><td>${esc(r.hiveId)}</td><td>${esc(names[r.state] || r.state)} (${Math.round(r.confidence * 100)}%)</td><td>${r.meanHz.toFixed(1)} ± ${r.sdHz.toFixed(1)}</td><td>${r.cleanFrames} / ${r.artifactFrames}</td><td>${esc(r.material)}</td><td>${esc(r.tempC)}</td></tr>`;
  }
  html += '</tbody></table><p class="rep-meta">Acoustic screening only — confirm with a physical inspection before acting.</p>';
  return html;
}
