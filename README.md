# Polymarket NVIDIA CEO forecast — entry

Per-market EWMA volatility + Student-t(3) quantiles + mean reversion to the sum-normalized price + sub-sqrt horizon scaling + level vol floor (Node 22 stdlib only).

Practice pinball loss **0.2559** vs house persistence baseline **0.3636** (30% better, validated locally against Polymarket CLOB history; the local scorer reproduces the baseline's published 0.363589 exactly).

Model: sigma = clamp(max(EWMA vol halflife 8 min, 0.3 × EWMA vol halflife 1440 min, 0.005·sqrt(x)), 0.0003, 0.008); center reverts toward x/sum at 0.008/min; quantiles q = clamp(center + z·sigma·h^0.43, 0, 1) with t(3) z-values [-1.359, -0.472, 0, 0.472, 1.359].

Why it wins: Polymarket Yes prices sit still ~89% of minutes and move in 0.0005 ticks; the baseline's fixed sigma 0.0025 spreads every market far too wide. The return distribution is peaked with fat tails (t3's tighter inner quantiles win), Yes prices sum to ~1.0–1.09 and the overround decays (big markets drift down), realized vol grows slower than a random walk (h^0.43, not sqrt), and bigger markets need more absolute vol than the EWMA alone gives them.
