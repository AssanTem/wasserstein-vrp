# Wasserstein distance and the volatility risk premium

Replication code for "Wasserstein Distance and the Volatility Risk Premium".

## Run
```
pip install -r requirements.txt
python vrp_wasserstein_v2.py                 # real data (Yahoo Finance), ~10-20 min
python vrp_wasserstein_v2.py --synthetic     # quick smoke test, no internet
```
Outputs (CSV + `cum_pnl.png`) are written to `results_v2/`.

Main options: `--sample-start 2007-01-01`, `--primary W1`, `--boot 100`, `--half-spread 0.005`, `--placebo-draws 500`.

## What the script does
Variance-swap-ladder P&L; four Wasserstein estimators (W1, W2, KDE-smoothed W1, block-bootstrap bias-corrected W1);
HAC predictive regressions and out-of-sample tests; regime rules; tail-risk metrics; multiple-testing corrections
(Bonferroni, Holm, BH); crisis windows; anatomy of removed trades; placebo benchmark (VIX level, trailing RV, time-shifted W).

## Data
Yahoo Finance tickers: ^GSPC, ^VIX, ^IXIC, ^VXN, ^DJI, ^VXD. Data are not redistributed.

## License
MIT
