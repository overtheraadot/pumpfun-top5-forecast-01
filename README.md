# pumpfun-top5-forecast-01

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/overtheraadot/pumpfun-top5-forecast-01/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/overtheraadot/pumpfun-top5-forecast-01/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

Monte-Carlo forecaster for pump.fun's top-5 coins by market cap 24h out: per-coin growth, fade and death hazards, newcomer mass for coins born after the cut.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads `$MAKEFUN_INPUT_DIR` (else `/input`) `history.ndjson` + `task.json` and writes `$MAKEFUN_OUTPUT_DIR` (else `/output`) `forecast.json` with exactly three keys: `probabilities`, `top5`, `sizes`. Python 3.13, standard library only, no network, deterministic (fixed seed).

## How it works

- Each ranked coin's log10 market cap (in SOL) is tracked over the marks; a recent growth rate is fit by least squares over the last 2 hours, shrunk toward the median slope and clamped.
- Forecast mean log-cap at horizon `h` is `x0 + g·τ·(1−exp(−h/τ)) − fade·(1−exp(−h/τ_fade))`: growth decays, and top coins fade. A venue-dependent death hazard (curve coins die fast) can collapse a coin to a low log-normal draw; death times are drawn per Monte-Carlo scenario so death persists across the seven answer reads.
- One seeded Monte-Carlo over the top 100 candidates at each answer read gives place probabilities (rank frequencies) and size distributions (k-th order statistics of log-cap).
- Newcomer mass: coins born after the history cut — or never ranked — take a share of each place that grows with horizon (about 40% at once, rising toward 90%), because in the practice data the top 5 turns over within minutes and 3 of the 5 coins at the last practice read were never in the history. Named probabilities are scaled by the keep-share and the rest is left unnamed, so the scorer shares it over the answer chart's unnamed coins. Sizes get the same mixture with a broad log-normal newcomer size distribution.
- Everything (places, size-bin edges, answer reads, output name) is read from `task.json`; no practice time or mint is hardcoded, so the same code runs the 2-hour rehearsal and the 11-hour final.

## Backtest

Rolling-origin backtest on the practice history (train on marks up to 20:10–20:40, predict each later mark with the exact log-score formula from the rules) plus the real last practice answer read (order only — its top 5 is published in the rules):

- Rolling mean: **−44.01** (order −23.64, size −20.37) vs house baseline −49.23 (order −23.17, size −26.06).
- Real last practice read, order only: **−21.08** vs baseline −23.20. (House baseline scores −40.41 total on all seven practice reads: order −21.38, size −19.03.)

## Files

- `main.py` — the forecaster
- `README.md` — this file
- `LICENSE` — MIT license
- `.makefun/owner.json` — the wallet that owns this entry

## Links

- Bounty: https://make.fun/bounty/pumpfun-top5-forecast-01

---

<sub>Entered on [make.fun](https://make.fun/?ref=8bg6hn2r&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
