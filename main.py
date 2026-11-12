#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry (Python 3.13, stdlib only, no network).

Forecasts the top 5 pump.fun coins by market cap at the seven answer reads
in task.json, plus each place's size distribution.

Method
------
1. Build each ranked coin's log10(cap in SOL) series over the marks.
2. Estimate a per-coin recent growth rate (least-squares slope over the last
   GROWTH_WIN_H hours, shrunk toward the global median slope, clamped).
   Forecast mean log-cap at horizon h:
       mu(h) = x0 + g * TAU * (1 - exp(-h / TAU))     (growth decays)
   with a persistent per-coin shock (quality) and per-read noise, and a
   venue-dependent death hazard (curve coins die fast; a dead coin's cap
   collapses to a low log-normal draw). Death times are drawn once per coin
   so death persists across reads.
3. Monte-Carlo (seeded) over the top CAND_N candidates at each answer read:
   place probabilities from rank frequencies, size distributions from the
   k-th order statistics of log-cap.
4. Newcomer mass: coins born after the cut (or never ranked) take a share
   u_k(h) of each place, growing with horizon; the named probabilities are
   scaled by (1 - u) and the rest is left unnamed so the scorer shares it
   over the answer chart's unnamed coins. Sizes get the same mixture with a
   broad log-normal newcomer size distribution.

Everything (places, size edges, reads, output name) is read from task.json;
no practice time or mint is hardcoded.
"""
import json
import math
import os
import random
import sys
from datetime import datetime

PARAMS = {
    "seed": 20261112,
    "cand_n": 100,
    "samples": 2500,
    "growth_win_h": 2.0,
    "shrink_n0": 6.0,          # pseudo-count for growth shrinkage
    "growth_clamp": (-0.5, 0.8),
    "tau_h": 2.5,              # growth decay time constant (hours)
    "rev_dex": 0.0,            # typical fade of a top coin's log-cap (dex);
                               # grid preferred 0.0: uniform fade does not help
                               # (turnover is carried by death + newcomer mass)
    "rev_tau_h": 2.0,          # fade time constant (hours)
    "sig_persist": 0.15,       # per-coin persistent shock sd (log10)
    "sig_rate": 0.15,          # per-read noise sd per sqrt(hour)
    "death_pool_h": 8.0,       # mean death time, pool venue (hours)
    "death_curve_h": 2.5,      # mean death time, curve venue (hours)
    "dead_mu": 1.7, "dead_sd": 0.4,  # dead coin log10 cap distribution
    "u0": 0.4,                 # newcomer share even at horizon 0 (jump process:
                               # a coin born minutes ago can take a place at once)
    "u_max": 0.9,              # asymptotic newcomer share per place
    "u_tau_h": 1.0,            # newcomer share growth time constant
    "u_place": [1.15, 1.05, 1.0, 0.95, 0.90],  # per-place factors
    "new_size_shift": 0.05,    # newcomer size mean shift vs incumbent order stat
    "new_size_sd": 0.55,       # newcomer size sd (log10)
    "size_uniform_mix": 0.02,  # uniform floor-protection mix for sizes
    "min_p": 0.0005,           # drop named probs below this
}

def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def _erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r

def _norm_cdf(x, mu, sd):
    if sd <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * _erfc(-(x - mu) / (sd * math.sqrt(2.0)))

def _bin_of(logcap, edges):
    if logcap < edges[0]:
        return 0
    lo, hi = 0, len(edges)
    # bin i (1..80): edges[i-1] <= x < edges[i]; bin 81: x >= edges[80]
    for i, e in enumerate(edges):
        if logcap < e:
            return i + 1 if i + 1 <= 80 else 81
    return 81

def build_forecast(marks, task, params=None):
    P = dict(PARAMS)
    if params:
        P.update(params)
    edges = task["size"]["edges"]
    n_places = task["places"]
    t0 = _parse_ts(task["t0"])
    ok_marks = [m for m in marks if m.get("ok") is True and isinstance(m.get("rows"), list)]
    if not ok_marks:
        raise ValueError("the history holds no ok mark")
    # last mark with rows supplies the chart; its time is the forecast base
    base_mark = None
    for m in reversed(ok_marks):
        if m["rows"]:
            base_mark = m
            break
    if base_mark is None:
        raise ValueError("no mark has rows")
    t_base = _parse_ts(base_mark["at"])
    reads = [_parse_ts(r) for r in task.get("reads", [])]
    if not reads:
        reads = [t_base]
    horizons = [max(0.0, (r - t_base).total_seconds() / 3600.0) for r in reads]

    # ---- per-coin series: mint -> list of (t_hours, logcap, venue) ----
    series = {}
    created = {}
    for m in ok_marks:
        t = (_parse_ts(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            lc = math.log10(int(r["market_cap_lamports"])) - 9.0
            series.setdefault(r["mint"], []).append((t, lc, r.get("venue", "curve")))
            created.setdefault(r["mint"], r.get("created_at"))
    t_last = (_parse_ts(base_mark["at"]) - t0).total_seconds() / 3600.0
    win = P["growth_win_h"]

    # ---- growth slopes (least squares over the recent window) ----
    slopes = {}
    for mint, obs in series.items():
        recent = [(t, lc) for (t, lc, _v) in obs if t >= t_last - win]
        if len(recent) >= 2:
            n = len(recent)
            mt = sum(t for t, _ in recent) / n
            ml = sum(l for _, l in recent) / n
            den = sum((t - mt) ** 2 for t, _ in recent)
            slope = sum((t - mt) * (l - ml) for t, l in recent) / den if den > 0 else 0.0
        elif recent:
            slope = 0.0
        else:
            slope = 0.0
        slopes[mint] = (slope, len(recent))
    all_slopes = sorted(s for s, n in slopes.values() if n >= 2)
    prior = all_slopes[len(all_slopes) // 2] if all_slopes else 0.0
    growth = {}
    for mint, (s, n) in slopes.items():
        g = (n * s + P["shrink_n0"] * prior) / (n + P["shrink_n0"])
        lo, hi = P["growth_clamp"]
        growth[mint] = max(lo, min(hi, g))

    # ---- candidates: last log-cap + asymptotic growth ----
    last = {}
    for mint, obs in series.items():
        t, lc, venue = obs[-1]
        last[mint] = (lc, venue, t)
    scored = []
    for mint, (lc, venue, _t) in last.items():
        asym = lc + growth[mint] * P["tau_h"]
        scored.append((asym, mint))
    scored.sort(key=lambda x: (-x[0], x[1]))
    cand = [mint for _a, mint in scored[:P["cand_n"]]]
    # always include the last mark's top 20
    top20 = [r["mint"] for r in sorted(base_mark["rows"], key=lambda r: r["rank"])[:20]]
    for mint in top20:
        if mint not in cand:
            cand.append(mint)
    n_cand = len(cand)
    cand_idx = {m: i for i, m in enumerate(cand)}

    rng = random.Random(P["seed"])
    x0 = [last[m][0] for m in cand]
    g = [growth[m] for m in cand]
    tau = P["tau_h"]
    mean_death = [P["death_pool_h"] if last[m][1] == "pool" else P["death_curve_h"] for m in cand]

    # ---- Monte Carlo per read ----
    # One scenario = one draw of each coin's persistent quality shock, death
    # time and dead value, shared across the reads of that scenario (a coin's
    # quality persists across reads); per-read noise is drawn per read.
    # Drawing the persistent shocks once globally would condition the whole
    # forecast on a single arbitrary draw, so they are redrawn per scenario.
    place_counts = [[0] * n_cand for _ in range(n_places)]
    order_stat_hist = [[0.0] * 82 for _ in range(n_places)]
    order_stat_mean = [[0.0] * 1 for _ in range(n_places)]
    newcomer_u = []  # per read, per place
    n_reads = len(reads)
    S = P["samples"]
    # per-read forecast means
    mus = []
    sig_rs = []
    for h in horizons:
        decay = tau * (1.0 - math.exp(-h / tau)) if h > 0 else 0.0
        fade = P["rev_dex"] * (1.0 - math.exp(-h / P["rev_tau_h"])) if h > 0 else 0.0
        mus.append([x0[i] + g[i] * decay - fade for i in range(n_cand)])
        sig_rs.append(P["sig_rate"] * math.sqrt(h) if h > 0 else 0.0)
        u_k = []
        for k in range(n_places):
            u = P["u0"] + (P["u_max"] - P["u0"]) * (1.0 - math.exp(-h / P["u_tau_h"])) if h > 0 else P["u0"]
            u_k.append(min(0.85, max(0.0, u * P["u_place"][k])))
        newcomer_u.append(u_k)
    for _s in range(S):
        z = [rng.gauss(0.0, 1.0) for _ in range(n_cand)]
        dead_val = [rng.gauss(P["dead_mu"], P["dead_sd"]) for _ in range(n_cand)]
        death_time = []
        for i in range(n_cand):
            u = rng.random()
            death_time.append(-mean_death[i] * math.log(max(1e-12, 1.0 - u)))
        for ri, h in enumerate(horizons):
            mu = mus[ri]
            sig_r = sig_rs[ri]
            vals = []
            for i in range(n_cand):
                if death_time[i] < h:
                    v = dead_val[i]
                else:
                    v = mu[i] + P["sig_persist"] * z[i]
                    if sig_r > 0:
                        v += sig_r * rng.gauss(0.0, 1.0)
                vals.append((v, i))
            vals.sort(key=lambda x: (-x[0], cand[x[1]]))
            for k in range(n_places):
                v, i = vals[k]
                place_counts[k][i] += 1
                order_stat_hist[k][_bin_of(v, edges)] += 1
                order_stat_mean[k][0] += v / S

    # ---- average over reads, apply newcomer scaling ----
    # place_counts / order_stat_hist accumulate over ALL reads, so their
    # means are total / (S * n_reads); each place is scaled by the mean
    # keep-share over reads, sizes get the mean newcomer share of the
    # broad newcomer size distribution.
    probs = [[0.0] * n_cand for _ in range(n_places)]
    sizes = [[0.0] * 82 for _ in range(n_places)]
    for k in range(n_places):
        mean_keep = sum(1.0 - newcomer_u[ri][k] for ri in range(n_reads)) / n_reads
        mean_u = 1.0 - mean_keep
        for i in range(n_cand):
            probs[k][i] = mean_keep * (place_counts[k][i] / (S * n_reads))
        mc_mean = order_stat_mean[k][0] / n_reads
        nm_mu = mc_mean + P["new_size_shift"]
        nm_sd = P["new_size_sd"]
        for b in range(82):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            pb = _norm_cdf(hi, nm_mu, nm_sd) - _norm_cdf(lo, nm_mu, nm_sd)
            sizes[k][b] = mean_keep * (order_stat_hist[k][b] / (S * n_reads)) + mean_u * pb
    # uniform floor-protection mix on sizes
    for k in range(n_places):
        for b in range(82):
            sizes[k][b] = (1.0 - P["size_uniform_mix"]) * sizes[k][b] + P["size_uniform_mix"] / 82.0

    def _cap_rows(rows):
        # rows: list of (key, p); round to 6dp, cap total at 0.999999
        out = [(key, math.floor(p * 1e6) / 1e6) for key, p in rows if p >= 0.000001]
        tot = sum(p for _k, p in out)
        if tot > 0.999999:
            f = 0.999999 / tot
            out = [(key, math.floor(p * f * 1e6) / 1e6) for key, p in out]
        return out

    probabilities = []
    for k in range(n_places):
        rows = [(cand[i], probs[k][i]) for i in range(n_cand) if probs[k][i] >= P["min_p"]]
        rows.sort(key=lambda x: (-x[1], x[0]))
        for mint, p in _cap_rows(rows):
            probabilities.append({"mint": mint, "place": k + 1, "p": p})
    sizes_out = []
    for k in range(n_places):
        rows = [(b, sizes[k][b]) for b in range(82) if sizes[k][b] >= 0.000001]
        for b, p in _cap_rows(rows):
            sizes_out.append({"place": k + 1, "bin": b, "p": p})
    # top5: greedy by averaged place probability
    used = set()
    top5 = []
    for k in range(n_places):
        best, bestp = None, -1.0
        for i in range(n_cand):
            if cand[i] in used:
                continue
            if probs[k][i] > bestp:
                best, bestp = cand[i], probs[k][i]
        if best is None:
            break
        used.add(best)
        top5.append(best)
    # fill from last mark if short (should not happen)
    for r in sorted(base_mark["rows"], key=lambda r: r["rank"]):
        if len(top5) >= n_places:
            break
        if r["mint"] not in used:
            used.add(r["mint"])
            top5.append(r["mint"])
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes_out}

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(in_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                marks.append(json.loads(line))
    fc = build_forecast(marks, task)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(fc) + "\n")

if __name__ == "__main__":
    main()
