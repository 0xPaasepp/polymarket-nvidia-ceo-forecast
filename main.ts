// polymarket-nvidia-ceo-forecast-01 entry (Node 22, stdlib only).
// Model: per-market EWMA volatility (short halflife 12 min + 30% of long
// halflife-1440 vol as a jump-risk floor), normal quantiles, no drift.
// Practice pinball loss 0.2761 vs house persistence baseline 0.3636
// (validated locally against Polymarket CLOB history; scorer reproduces
// the baseline's published 0.363589 exactly).
//
// Tuned parameters (grid search over 6 practice cases, flat optimum so
// the choice is robust, not knife-edge): SIGMA_MIN 0.0002, SIGMA_MAX
// 0.004, HL_SHORT 12, HL_LONG 1440, LONG_K 0.3. Both EWMA windows use
// the last 360 returns: the long window's halflife is what makes it a
// jump-risk floor, not a longer lookback — a 4320-minute lookback was
// tested and scores worse (0.403) because it drags in stale jumps from
// when the markets were new and volatile.
//
// Why it beats persistence: Polymarket Yes prices move in tiny ticks and
// sit still ~89% of minutes; the baseline's fixed sigma 0.0025 spreads
// every market far too wide. Recent per-market vol is mostly 0.0002-0.002,
// so tighter intervals win. Mean reversion and drift both hurt on
// practice (tested); the cross-market sum constraint is already
// preserved by the medians, so no normalization is applied.
//
// Runs offline, no network, under the 8,000 ms per-case cap.
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
const SIGMA_MIN = 0.0002;
const SIGMA_MAX = 0.004;
const HL_SHORT = 12; // minutes, EWMA halflife for recent vol
const HL_LONG = 1440; // minutes, EWMA halflife for jump-risk floor
const LONG_K = 0.3; // floor = LONG_K * long-term vol
const LOOKBACK_SHORT = 360; // minutes of returns for short vol
const LOOKBACK_LONG = 360; // minutes of returns for long vol (matches tuned prototype)

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** EWMA std of 1-min returns over the last `lookback` minutes, halflife `hl`. */
function ewmaVol(prices: number[], halflife: number, lookback: number): number {
  const n = prices.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - lookback - 1);
  const alpha = 1 - Math.pow(0.5, 1 / halflife);
  let variance = 0;
  let wsum = 0;
  let w = 1;
  // most recent first: returns[i] = prices[i+1]-prices[i], i from n-2 down
  for (let i = n - 2; i >= start; i--) {
    const r = prices[i + 1]! - prices[i]!;
    variance += w * r * r;
    wsum += w;
    w *= 1 - alpha;
  }
  return wsum > 0 ? Math.sqrt(variance / wsum) : 0;
}

function forecast(m: CaseMarket, horizon: number): number[][] {
  const p = m.prices;
  if (p.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = p[p.length - 1]!;
  const sigShort = ewmaVol(p, HL_SHORT, LOOKBACK_SHORT);
  const sigLong = ewmaVol(p, HL_LONG, LOOKBACK_LONG);
  const sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, sigShort, LONG_K * sigLong));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * Math.sqrt(h);
    rows.push(Z.map((z) => price6(x + z * s)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
