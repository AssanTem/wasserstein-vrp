#!/usr/bin/env python3
"""
Wasserstein distance and the Volatility Risk Premium -- v2 pipeline

Usage
-----
  pip install numpy pandas scipy statsmodels matplotlib yfinance
  python vrp_wasserstein_v2.py                 # real data (Yahoo Finance)
  python vrp_wasserstein_v2.py --synthetic     # smoke test without internet

What changed vs. the first version
----------------------------------
1. P&L: short 21-day variance-swap ladder, marked to market daily
   (delta-neutral vol exposure), instead of Pos_{t-1} * R_t (directional equity).
   VRP is defined ex-ante/forward: IV_t^2 - RV^2_{t,t+21}.
2. Sample: data from 2005 (warm-up), evaluation from 2007-01-01.
   Covers GFC 2008, 2011, Volmageddon (Feb 2018), Q4 2018, COVID, 2022.
3. Multiple testing: Holm / Bonferroni / BH-FDR over every hypothesis family.
4. Discretisation noise: W1, W2, KDE-smoothed W1, and a moving-block-bootstrap
   noise-corrected W1 (W - E0[W]); plus controls for VIX/RV level (scale vs shape).
5. Tail metrics: VaR99, ES99, Cornish-Fisher modified VaR, Pezier-White adjusted
   Sharpe, skew/kurtosis, max drawdown, crisis-window table.
6. Anatomy of trades excluded by the regime filter (why 70th != 75th).
7. Placebo benchmark: is W better than VIX level, trailing RV or a time-shifted copy of W?
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.special import ndtr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

INDICES = {"SPX": ("^GSPC", "^VIX"), "COMP": ("^IXIC", "^VXN"), "DJIA": ("^DJI", "^VXD")}
H = 21          # variance-swap tenor / forecast horizon (trading days)
WIN = 20        # Wasserstein window length
TD = 252
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
CRISES = {
    "GFC 2008-09": ("2008-09-01", "2009-03-31"),
    "Aug-Oct 2011": ("2011-08-01", "2011-10-31"),
    "Volmageddon 2018-02": ("2018-01-26", "2018-02-28"),
    "Q4 2018": ("2018-10-01", "2018-12-31"),
    "COVID 2020": ("2020-02-15", "2020-04-30"),
    "Hiking 2022": ("2022-01-01", "2022-10-31"),
}


# --------------------------------------------------------------------------- data
def load_real(start):
    import yfinance as yf
    out = {}
    for name, (px, vx) in INDICES.items():
        d = yf.download([px, vx], start=start, auto_adjust=False, progress=False)["Close"]
        d = d.rename(columns={px: "px", vx: "vol"}).dropna()
        out[name] = d
        print(f"{name}: {d.index[0].date()} -> {d.index[-1].date()}  ({len(d)} rows)")
    return out


def load_synthetic(n=5200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    out = {}
    for name in INDICES:
        v = 1e-4
        var = np.empty(n)
        r = np.empty(n)
        for t in range(n):
            var[t] = v
            jump = rng.standard_normal() * 0.04 if rng.random() < 0.003 else 0.0
            r[t] = np.sqrt(v) * rng.standard_normal() + jump
            v = 2e-6 + 0.90 * v + 0.08 * r[t] ** 2
        vol = np.clip(100 * np.sqrt(252 * np.r_[var[1:], var[-1]]) * 1.15 + rng.standard_normal(n), 9, None)
        out[name] = pd.DataFrame({"px": 1000 * np.exp(np.cumsum(r)), "vol": vol}, index=idx)
    return out


# ------------------------------------------------------------------ Wasserstein
def windows(r, win=WIN):
    """P (last `win` returns ending at t) and Q (preceding `win`), rows aligned to t>=2win-1."""
    sw = sliding_window_view(r, win)
    tt = np.arange(2 * win - 1, len(r))
    return tt, sw[tt - win + 1], sw[tt - 2 * win + 1]


def w_emp(P, Q, p):
    d = np.abs(np.sort(P, 1) - np.sort(Q, 1))
    return d.mean(1) if p == 1 else np.sqrt((d ** 2).mean(1))


def w_kde(P, Q, p, ngrid=400, nu=199):
    """W_p between Gaussian-KDE-smoothed windows (common Silverman bandwidth)."""
    u = (np.arange(nu) + 0.5) / nu
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        pooled = np.concatenate([P[i], Q[i]])
        h = 1.06 * pooled.std(ddof=1) * len(pooled) ** -0.2
        if not h > 0:
            continue
        g = np.linspace(pooled.min() - 4 * h, pooled.max() + 4 * h, ngrid)
        q = [np.interp(u, ndtr((g[:, None] - s[None, :]) / h).mean(1), g) for s in (P[i], Q[i])]
        d = np.abs(q[0] - q[1])
        out[i] = d.mean() if p == 1 else np.sqrt((d ** 2).mean())
    return out


def w_null_mean(P, Q, B=100, L=5, seed=0):
    """Moving-block-bootstrap null: both windows redrawn from the pooled 2*WIN sample.
    Returns E0[W1] (small-sample bias) so that W1 - E0 is a noise-corrected distance."""
    rng = np.random.default_rng(seed)
    win = P.shape[1]
    nb = win // L
    mu = np.empty(len(P))
    off = np.arange(L)[None, None, :]
    for i in range(len(P)):
        pooled = np.concatenate([Q[i], P[i]])
        starts = rng.integers(0, 2 * win - L + 1, size=(B, 2 * nb))
        samp = pooled[(starts[:, :, None] + off).reshape(B, 2 * nb * L)]
        mu[i] = np.abs(np.sort(samp[:, :win], 1) - np.sort(samp[:, win:], 1)).mean(1).mean()
    return mu


# ------------------------------------------------------------------ data prep
def prepare(d, boot):
    df = pd.DataFrame({"r": np.log(d["px"]).diff(), "iv2": (d["vol"] / 100) ** 2}).iloc[1:]
    r2 = df["r"] ** 2
    df["rv2_tr"] = TD * r2.rolling(H).mean()                       # trailing (ex-ante) RV^2
    cs = r2.cumsum()
    df["rv2_fwd"] = TD / H * (cs.shift(-H) - cs)                   # realised RV^2 over (t, t+H]
    df["vrp_fwd"] = df["iv2"] - df["rv2_fwd"]                      # forward VRP (variance pts)
    tt, P, Q = windows(df["r"].values)
    ix = df.index[tt]
    W = pd.DataFrame(index=ix)
    W["W1"] = w_emp(P, Q, 1)
    W["W2"] = w_emp(P, Q, 2)
    W["W1_kde"] = w_kde(P, Q, 1)
    W["W1_bc"] = W["W1"] - w_null_mean(P, Q, B=boot)
    return df, W


def ladder_pnl(df, w, half_spread):
    """Daily MTM P&L (variance points) of a ladder of short 21d variance swaps.
    Slice opened at close t with weight w_t/H, strike IV_t^2.
    MTM_j = IV_t^2 - (252/H) * sum_{i<=j} r_{t+i}^2 - (1 - j/H) * IV_{t+j}^2 ;  MTM_H = IV_t^2 - RV^2_fwd."""
    r2, iv2 = df["r"] ** 2, df["iv2"]
    cs = r2.cumsum()
    w = w.reindex(df.index).fillna(0.0)
    pnl = pd.Series(0.0, index=df.index)
    prev = pd.Series(0.0, index=df.index)
    for j in range(1, H + 1):
        mtm = iv2 - (TD / H) * (cs.shift(-j) - cs) - (1 - j / H) * iv2.shift(-j)
        pnl = pnl + (w * (mtm - prev) / H).shift(j).fillna(0.0)
        prev = mtm
    cost = w.abs() / H * 2 * np.sqrt(iv2) * half_spread            # half-spread paid at slice entry
    return pnl, pnl - cost


def rank_of(x):
    """Percentile rank of x_t within its trailing 252 days (includes t)."""
    return x.rolling(TD).apply(lambda a: (a <= a[-1]).mean(), raw=True)


def normal_of(x, q):
    """1 if x_t <= expanding q-quantile of strictly earlier values, NaN before warm-up."""
    thr = x.expanding(min_periods=TD).quantile(q).shift(1)
    return (x <= thr).astype(float).where(thr.notna())


def make_rules(df, W, primary):
    x = W[primary].reindex(df.index)
    sig = (df["iv2"] > df["rv2_tr"]).astype(float)
    rank = rank_of(x)
    rules, normal = {"Unfiltered": sig}, {}
    for q in THRESHOLDS:
        nm = normal_of(x, q)
        normal[q] = nm
        rules[f"Filt_{int(q*100)}"] = sig * nm
    rules["Aggressive"] = sig * (1 + rank)
    rules["Continuous"] = sig * (1 - 0.8 * rank)
    return rules, sig, normal


# ------------------------------------------------------------------ placebo benchmark
def ladder_precompute(df):
    """Daily MTM increments D[j-1][t] of a slice opened at t (same maths as ladder_pnl, reusable & fast)."""
    r2, iv2 = df["r"] ** 2, df["iv2"]
    cs = r2.cumsum()
    prev = pd.Series(0.0, index=df.index)
    D = []
    for j in range(1, H + 1):
        mtm = iv2 - (TD / H) * (cs.shift(-j) - cs) - (1 - j / H) * iv2.shift(-j)
        D.append((mtm - prev).fillna(0.0).values)
        prev = mtm
    return np.array(D)


def ladder_fast(D, w):
    n = len(w)
    pnl = np.zeros(n)
    for j in range(1, H + 1):
        pnl[j:] += (w * D[j - 1])[: n - j] / H
    return pnl


def _stats(x):
    x = x[~np.isnan(x)]
    q = np.quantile(x, 0.01)
    c = np.cumsum(x)
    return {"Sharpe": x.mean() / x.std() * np.sqrt(TD), "ES99%": -100 * x[x <= q].mean(),
            "MaxDD%": 100 * (c - np.maximum.accumulate(c)).min()}


def placebo_test(df, sig, W, primary, smask, scale, n_draw, seed=7):
    """Is W a better de-risking signal than (a) VIX level, (b) trailing RV, (c) the SAME W series
    circularly shifted in time (keeps its distribution and persistence, destroys its timing)?
    Rules compared: Continuous (1-0.8*rank) and Filter q=0.75."""
    D = ladder_precompute(df)
    n = len(df)
    s = sig.values
    rng = np.random.default_rng(seed)

    def build(rk, nm):
        rk = np.nan_to_num(rk, nan=0.0)
        nm = np.nan_to_num(nm, nan=0.0)
        return {"Continuous": s * (1 - 0.8 * rk), "Filt_75": s * nm}

    def evaluate(wd):
        return {k: _stats(ladder_fast(D, w)[smask] * scale) for k, w in wd.items()}

    sources = {primary: W[primary].reindex(df.index), "VIX level": df["iv2"], "Trailing RV": df["rv2_tr"]}
    rows, cache = [], {}
    for name, x in sources.items():
        rk, nm = rank_of(x).values, normal_of(x, 0.75).values
        cache[name] = (rk, nm)
        for rule, st in evaluate(build(rk, nm)).items():
            rows.append({"signal": name, "rule": rule, **st})
    # placebo draws: circular time-shift of the primary W rank / regime series
    rk0, nm0 = cache[primary]
    ok_r, ok_n = ~np.isnan(rk0), ~np.isnan(nm0)
    draws = []
    for _ in range(n_draw):
        k = int(rng.integers(TD, min(ok_r.sum(), ok_n.sum()) - TD))
        rk, nm = np.full(n, np.nan), np.full(n, np.nan)
        rk[ok_r] = np.roll(rk0[ok_r], k)
        nm[ok_n] = np.roll(nm0[ok_n], k)
        for rule, st in evaluate(build(rk, nm)).items():
            draws.append({"rule": rule, **st})
    pl = pd.DataFrame(draws)
    res = pd.DataFrame(rows)
    summ = []
    for rule in ("Continuous", "Filt_75"):
        w = res[(res.signal == primary) & (res.rule == rule)].iloc[0]
        d = pl[pl.rule == rule]
        summ.append({"rule": rule, "W_Sharpe": w.Sharpe, "W_ES99%": w["ES99%"], "W_MaxDD%": w["MaxDD%"],
                     "placebo_med_Sharpe": d.Sharpe.median(), "placebo_med_ES99%": d["ES99%"].median(),
                     "placebo_med_MaxDD%": d["MaxDD%"].median(),
                     "p_Sharpe": (1 + (d.Sharpe >= w.Sharpe).sum()) / (1 + len(d)),
                     "p_ES99": (1 + (d["ES99%"] <= w["ES99%"]).sum()) / (1 + len(d)),
                     "p_MaxDD": (1 + (d["MaxDD%"] >= w["MaxDD%"]).sum()) / (1 + len(d))})
    return res, pd.DataFrame(summ)


# ------------------------------------------------------------------ metrics
def metrics(x):
    x = pd.Series(x).dropna()
    mu, sd = x.mean(), x.std()
    sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
    sr = mu / sd
    asr = sr * (1 + sk / 6 * sr - (ku - 3) / 24 * sr ** 2)         # Pezier-White adjusted Sharpe
    q01 = np.quantile(x, 0.01)
    z = stats.norm.ppf(0.01)
    zcf = z + (z**2 - 1) * sk / 6 + (z**3 - 3 * z) * (ku - 3) / 24 - (2 * z**3 - 5 * z) * sk**2 / 36
    cum = x.cumsum()
    return {"AnnRet%": 100 * mu * TD, "AnnVol%": 100 * sd * np.sqrt(TD), "Sharpe": sr * np.sqrt(TD),
            "AdjSharpe_PW": asr * np.sqrt(TD), "Skew": sk, "ExKurt": ku - 3,
            "VaR99%": -100 * q01, "ES99%": -100 * x[x <= q01].mean(), "CF_VaR99%": -100 * (mu + zcf * sd),
            "MaxDD%": 100 * (cum - cum.cummax()).min(), "WorstDay%": 100 * x.min()}


def _sharpe_v(a):
    return a.mean(1) / a.std(1)


def _es_v(a):
    k = max(1, int(np.ceil(0.01 * a.shape[1])))
    return -np.sort(a, 1)[:, :k].mean(1)


def block_boot_diff(a, b, stat, B=1000, L=21, seed=1):
    """Two-sided circular-block-bootstrap p-value for stat(a) - stat(b), paired."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    obs = stat(a[None, :])[0] - stat(b[None, :])[0]
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / L))
    idx = ((rng.integers(0, n, size=(B, nblk))[:, :, None] + np.arange(L)) % n).reshape(B, -1)[:, :n]
    d = stat(a[idx]) - stat(b[idx])
    return obs, float(np.mean(np.abs(d - obs) >= abs(obs)))


# ------------------------------------------------------------------ regression
def hac_reg(y, X, lags=H):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def oos_test(y, Xm, Xb, refit=21, first=0.5):
    """Expanding-window OOS with H-day embargo. R2_OS vs controls-only benchmark + Clark-West."""
    n = len(y)
    Y = y.values
    A = np.column_stack([np.ones(n), Xm.values])
    Bm = np.column_stack([np.ones(n), Xb.values])
    yt, pm, pb = [], [], []
    for t in range(int(n * first), n, refit):
        tr = slice(0, t - H)
        te = slice(t, min(t + refit, n))
        pm.append(A[te] @ np.linalg.lstsq(A[tr], Y[tr], rcond=None)[0])
        pb.append(Bm[te] @ np.linalg.lstsq(Bm[tr], Y[tr], rcond=None)[0])
        yt.append(Y[te])
    yt, pm, pb = map(np.concatenate, (yt, pm, pb))
    r2 = 1 - ((yt - pm) ** 2).sum() / ((yt - pb) ** 2).sum()
    f = (yt - pb) ** 2 - ((yt - pm) ** 2 - (pb - pm) ** 2)
    cw = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": H})
    return r2, float(1 - stats.norm.cdf(cw.tvalues[0]))


# ------------------------------------------------------------------ anatomy
def episodes(mask):
    grp = (mask != mask.shift()).cumsum()
    return [g.index for _, g in mask[mask].groupby(grp[mask])]


def anatomy(name, df, sig, normal, unit_scale, sample, out):
    trade = (df["vrp_fwd"] * unit_scale).loc[sample].dropna()      # final P&L of a full-size swap opened at t
    # (contribution to capital = trade / H, since each slice has weight 1/H); last H days have no outcome yet
    base = sig.loc[trade.index] == 1
    rows, eprows = [], []
    q10, q90 = trade[base].quantile(0.10), trade[base].quantile(0.90)
    excl_by_q = {}
    for q, nm in normal.items():
        ex = base & (nm.loc[trade.index] == 0)
        excl_by_q[q] = ex
        eps = episodes(ex)
        pnl_ex = trade[ex]
        rows.append({"index": name, "q": q, "trades_unfiltered": int(base.sum()), "trades_excluded": int(ex.sum()),
                     "episodes": len(eps), "mean_ep_len": ex.sum() / max(len(eps), 1),
                     "PnL_excluded_sum": pnl_ex.sum(), "avoided_losses": pnl_ex[pnl_ex < 0].sum(),
                     "forgone_gains": pnl_ex[pnl_ex > 0].sum(),
                     "share_worst10%_avoided": (ex & (trade <= q10) & base).sum() / max(((trade <= q10) & base).sum(), 1),
                     "share_best10%_forgone": (ex & (trade >= q90) & base).sum() / max(((trade >= q90) & base).sum(), 1)})
        for e in eps:
            eprows.append({"index": name, "q": q, "start": e[0].date(), "end": e[-1].date(), "n": len(e),
                           "PnL": trade[e].sum()})
    pd.DataFrame(rows).to_csv(f"{out}/anatomy_summary_{name}.csv", index=False)
    ep = pd.DataFrame(eprows)
    ep["absPnL"] = ep["PnL"].abs()
    ep.sort_values(["q", "absPnL"], ascending=[True, False]).groupby("q").head(5).drop(columns="absPnL") \
        .to_csv(f"{out}/anatomy_top_episodes_{name}.csv", index=False)
    # trades excluded at 70th but kept at 75th: the exact source of the 70 vs 75 gap
    diff = excl_by_q[0.70] & ~excl_by_q[0.75]
    dd = pd.DataFrame({"PnL": trade[diff]})
    dd["year"] = dd.index.year
    dd.sort_values("PnL").to_csv(f"{out}/anatomy_70_vs_75_{name}.csv")
    print(f"\n[{name}] trades excluded at 70th but kept at 75th: n={int(diff.sum())}, "
          f"sum P&L={dd['PnL'].sum():.3f}; 5 worst: ")
    print(dd.sort_values("PnL").head(5).to_string())
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def run(a):
    os.makedirs(a.out, exist_ok=True)
    data = load_synthetic() if a.synthetic else load_real(a.data_start)
    reg_rows, test_rows, met_rows, crisis_rows, oos_rows, curves = [], [], [], [], [], {}
    pl_alt, pl_sum = [], []
    for name, d in data.items():
        print(f"\n=== {name}: building features ===")
        df, W = prepare(d, a.boot)
        rules, sig, normal = make_rules(df, W, a.primary)
        first_valid = max(s.first_valid_index() for s in rules.values() if s.first_valid_index() is not None)
        start = max(pd.Timestamp(a.sample_start), df.index[df.index.get_loc(first_valid) + H])
        sample = df.index[(df.index >= start)]
        pnl_g, pnl_n = {}, {}
        for k, w in rules.items():
            pnl_g[k], pnl_n[k] = ladder_pnl(df, w, a.half_spread)
        scale = 0.10 / (pnl_g["Unfiltered"].loc[sample].std() * np.sqrt(TD))   # unfiltered -> 10% ann. vol
        unit = df["iv2"].loc[sample].mean()
        G = pd.DataFrame(pnl_g).loc[sample] * scale
        N = pd.DataFrame(pnl_n).loc[sample] * scale
        curves[name] = G.cumsum()

        # ---- tail-risk metrics (gross + net Sharpe)
        for k in G:
            m = metrics(G[k])
            m["NetSharpe"] = metrics(N[k])["Sharpe"]
            met_rows.append({"index": name, "rule": k, **m})

        # ---- crisis windows
        for cname, (s, e) in CRISES.items():
            for k in G:
                x = G[k].loc[s:e]
                if len(x) < 5:
                    continue
                c = x.cumsum()
                crisis_rows.append({"index": name, "window": cname, "rule": k, "cum_PnL%": 100 * x.sum(),
                                    "maxDD%": 100 * (c - c.cummax()).min()})

        # ---- predictive regressions (forward 21d VRP), raw + controlled
        ys = df["vrp_fwd"].iloc[:-H]
        ys = (ys / unit).loc[start:]
        ctrl = pd.DataFrame({"iv2": df["iv2"] / unit, "rv2_tr": df["rv2_tr"] / unit}).reindex(ys.index)
        base_r2 = hac_reg(ys, ctrl).rsquared
        for pname in W.columns:
            x = W[pname].reindex(ys.index)
            x = (x - x.mean()) / x.std()
            for spec, X in (("raw", x.to_frame(pname)), ("ctrl", pd.concat([x.rename(pname), ctrl], axis=1))):
                ok = X.notna().all(axis=1)
                m = hac_reg(ys[ok], X[ok])
                reg_rows.append({"index": name, "predictor": pname, "spec": spec, "beta": m.params[pname],
                                 "t_HAC": m.tvalues[pname], "p": m.pvalues[pname], "R2": m.rsquared,
                                 "dR2_vs_controls": (m.rsquared - base_r2) if spec == "ctrl" else np.nan, "N": int(ok.sum())})
        x = W[a.primary].reindex(ys.index)
        ok = x.notna() & ctrl.notna().all(axis=1)
        Xm = pd.concat([((x - x.mean()) / x.std()).rename("W"), ctrl], axis=1)[ok]
        r2os, pcw = oos_test(ys[ok], Xm, ctrl[ok])
        oos_rows.append({"index": name, "predictor": a.primary, "R2_OOS_vs_controls": r2os, "ClarkWest_p": pcw})

        # ---- strategy tests vs unconditional (paired block bootstrap)
        for k in G:
            if k == "Unfiltered":
                continue
            ds, ps = block_boot_diff(G[k].values, G["Unfiltered"].values, _sharpe_v)
            de, pe = block_boot_diff(G[k].values, G["Unfiltered"].values, _es_v)
            test_rows.append({"index": name, "rule": k, "dSharpe": ds * np.sqrt(TD), "p_Sharpe": ps,
                              "dES99%": 100 * de, "p_ES99": pe})

        anatomy(name, df, sig, normal, scale, sample, a.out)

        if a.placebo_draws > 0:
            alt, sm = placebo_test(df, sig, W, a.primary, (df.index >= start), scale, a.placebo_draws)
            alt.insert(0, "index", name)
            sm.insert(0, "index", name)
            pl_alt.append(alt)
            pl_sum.append(sm)

    # ---- multiple-testing corrections, applied per hypothesis family
    def adjust(t, pcol):
        p = t[pcol].values
        for meth, lab in (("bonferroni", "p_bonf"), ("holm", "p_holm"), ("fdr_bh", "q_BH")):
            t[f"{lab}"] = multipletests(p, method=meth)[1]
        return t
    reg = adjust(pd.DataFrame(reg_rows), "p")
    tst = pd.DataFrame(test_rows)
    ts = adjust(tst[["index", "rule", "dSharpe", "p_Sharpe"]].copy(), "p_Sharpe")
    te = adjust(tst[["index", "rule", "dES99%", "p_ES99"]].copy(), "p_ES99")

    met, cri, oos = pd.DataFrame(met_rows), pd.DataFrame(crisis_rows), pd.DataFrame(oos_rows)
    for nm, t in (("regressions", reg), ("rule_tests_sharpe", ts), ("rule_tests_ES99", te),
                  ("metrics", met), ("crisis_windows", cri), ("oos", oos)):
        t.to_csv(f"{a.out}/{nm}.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== Predictive regressions (HAC, lags=21), family size", len(reg), "===")
    print(reg.to_string(index=False))
    print("\n=== OOS (expanding, embargo) ===")
    print(oos.to_string(index=False))
    print("\n=== Tail metrics (10% vol-scaled, gross) ===")
    print(met.to_string(index=False))
    print("\n=== Rule vs Unfiltered: Sharpe difference (block bootstrap, adj. p) ===")
    print(ts.to_string(index=False))
    print("\n=== Rule vs Unfiltered: ES99 difference (block bootstrap, adj. p) ===")
    print(te.to_string(index=False))

    if pl_sum:
        pa, ps = pd.concat(pl_alt), pd.concat(pl_sum)
        pa.to_csv(f"{a.out}/placebo_signals.csv", index=False)
        ps.to_csv(f"{a.out}/placebo_summary.csv", index=False)
        print("\n=== Placebo: same rules driven by VIX level / trailing RV ===")
        print(pa.to_string(index=False))
        print("\n=== Placebo: W vs. time-shifted W (p = share of placebo draws at least as good as W) ===")
        print(ps.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(curves), 1, figsize=(11, 3.6 * len(curves)), sharex=True)
        for ax, (nm, c) in zip(np.atleast_1d(axs), curves.items()):
            for k in ("Unfiltered", "Filt_75", "Aggressive", "Continuous"):
                ax.plot(c.index, 100 * c[k], label=k, lw=1.2)
            for s, e in CRISES.values():
                ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15)
            ax.set_title(f"{nm}: cumulative P&L, % of capital (unfiltered scaled to 10% vol)")
            ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{a.out}/cum_pnl.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"\nAll tables written to {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-start", default="2005-01-01", help="download start (warm-up before sample)")
    ap.add_argument("--sample-start", default="2007-01-01")
    ap.add_argument("--primary", default="W1", choices=["W1", "W2", "W1_kde", "W1_bc"],
                    help="distance used for regime rules (pre-specify; do not pick ex post)")
    ap.add_argument("--boot", type=int, default=100, help="bootstrap draws per day for noise correction")
    ap.add_argument("--half-spread", type=float, default=0.005, help="half bid-ask in vol points (decimal)")
    ap.add_argument("--placebo-draws", type=int, default=500, help="0 disables the placebo benchmark")
    ap.add_argument("--out", default="results_v2")
    run(ap.parse_known_args()[0])
