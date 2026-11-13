# Polymarket NVIDIA CEO forecast — entry

Per-market EWMA volatility + point-mass mixture over a Student-t(3) jump distribution + mean reversion to the sum-normalized price + sub-sqrt horizon scaling + level vol floor + asymmetric up/down jump scales (Node 22 stdlib only).

Practice pinball loss **0.2524** vs house persistence baseline **0.3636** (31% better, validated locally against Polymarket CLOB history; the local scorer reproduces the baseline's published 0.363589 exactly).

Model: sigma = clamp(max(EWMA vol halflife 8 min, 0.3 × EWMA vol halflife 1440 min, 0.005·sqrt(x)), 0.0003, 0.008); center reverts toward x/sum at 0.008/min; P(price unchanged after h min) = (1-p_move)^h with p_move from the last 120 min; conditional on moving, jumps follow Student-t(3) (exact SD-unit quantiles -1.3587/-0.4416/0/+0.4416/+1.3587) with up scale 0.8 and down scale 1.2; width ~ sigma·h^0.41.

Why it wins: Polymarket Yes prices sit still ~89% of minutes and move in 0.0005 ticks; the baseline's fixed sigma 0.0025 spreads every market far too wide. The point-mass mixture collapses inner quantiles onto the center at short horizons. The return distribution is peaked with fat tails (t3's tighter inner quantiles win). Yes prices sum to ~1.0–1.09 and the overround decays (big markets drift down). Down moves are bigger than up moves. Realized vol grows slower than a random walk (h^0.41, not sqrt). Bigger markets need more absolute vol than the EWMA alone gives them.
