# Polymarket NVIDIA CEO forecast — entry

Per-market EWMA volatility model (Node 22 stdlib only).

Practice pinball loss **0.2761** vs house persistence baseline **0.3636** (24% better, validated locally against Polymarket CLOB history; the local scorer reproduces the baseline's published 0.363589 exactly).

Model: sigma = clamp(max(EWMA vol halflife 12 min, 0.3 × EWMA vol halflife 1440 min), 0.0002, 0.004); quantiles q = clamp(x + z·sigma·√h, 0, 1). No drift, no mean reversion (both hurt on practice), no cross-market normalization (medians already preserve the sum).

Why it wins: Polymarket Yes prices sit still ~89% of minutes and move in 0.0005 ticks; the baseline's fixed sigma 0.0025 spreads every market far too wide. Recent per-market vol is mostly 0.0002–0.002.
