# Analysis code

Data: `qqq_dataset/qqq_spy_ff_monthly_crisis.csv`, built by `01_fetch_clean.py` (download -> clean -> Crisis indicator).

Run in this order from the project folder (or open a script in VS Code and run its `# %%` cells):

| Script | What it does | Packages |
|---|---|---|
| `01_fetch_clean.py` | download French factors + Yahoo prices, clean, build Past24 / Crisis (falls back to raw files offline) | pandas, numpy, yfinance |
| `02_eda.py` | Tables 1-3, Figures 1-5 | pandas, numpy, matplotlib |
| `03_models.py` | M1-M4, model comparison, final model summary | + statsmodels |
| `04_diagnostics.py` | residual checks, VIF, influence, Newey-West, orthogonalization, ridge trace | + statsmodels, scipy |
| `05_tests.py` | Q1-Q4 hypothesis tests | + statsmodels |
| `06_backtest.py` | 70/30 split, rolling betas, hedging backtest, all-subsets | pandas, numpy, matplotlib |
| `07_robustness.py` | drop influential months / 2000-02, Huber, Q4 with 3 state definitions (run after 04) | + statsmodels |

`common.py` holds paths, the data loader (long csv names -> short names such as EX_QQQ, MKT, CRISIS) and helpers. Report figures (graphic1-5, from `graphic.py`) go to `output/figures`; exploratory and diagnostic plots from 02/04/06 go to `output/extra_figures`; tables go to `output/tables`.

Install once: `python3 -m pip install pandas numpy scipy statsmodels matplotlib yfinance`
