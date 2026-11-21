"""Step 6 - Backtest: out-of-sample validation, rolling regressions, hedging backtest,
all-subsets comparison. Needs only numpy, pandas, matplotlib.
Rule: every estimate used at month t comes from data before t.
Run:  python code/06_backtest.py
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
from itertools import combinations
import matplotlib.pyplot as plt

df = load()
n = len(df)
MODELS = {"M1 CAPM": ["MKT"], "M2 FF3": ["MKT", "SMB", "HML"],
          "M3 FF5": ["MKT", "SMB", "HML", "RMW", "CMA"],
          "M4 FF5+MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]}

def ols(y, X):
    """OLS with intercept; returns [alpha, betas...]."""
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X)])
    return np.linalg.lstsq(X1, np.asarray(y), rcond=None)[0]

def predict(b, X):
    return b[0] + np.asarray(X) @ b[1:]

# %% 70/30 time split: MSE (train) vs MSPR (test)
cut = int(0.7 * n)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"Train {train.index[0]}-{train.index[-1]} ({len(train)}), test {test.index[0]}-{test.index[-1]} ({len(test)})")
rows = {}
for name, cols in MODELS.items():
    b = ols(train["EX_QQQ"], train[cols]); p = len(b)
    mse = ((train["EX_QQQ"] - predict(b, train[cols])) ** 2).sum() / (len(train) - p)
    mspr = ((test["EX_QQQ"] - predict(b, test[cols])) ** 2).mean()
    rows[name] = {"p": p, "MSE (train)": mse, "MSPR (test)": mspr, "MSPR / MSE": mspr / mse}
split_table = pd.DataFrame(rows).T
print(split_table.round(3))
save_tex(split_table, "tab7_mse_vs_mspr.tex")

# %% 36-month rolling regressions (row t uses months t-35 ... t)
WINDOW = 36
def rolling_ols(cols):
    out = {df.index[e]: ols(df["EX_QQQ"].iloc[e - WINDOW + 1:e + 1], df[cols].iloc[e - WINDOW + 1:e + 1])
           for e in range(WINDOW - 1, n)}
    return pd.DataFrame(out, index=["alpha"] + cols).T

roll_ff5, roll_capm = rolling_ols(MODELS["M3 FF5"]), rolling_ols(["MKT"])
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
t = roll_ff5.index.to_timestamp()
axes[0].plot(t, 12 * roll_ff5["alpha"]); axes[0].axhline(0, color="k", lw=0.6)
axes[0].set(ylabel="Alpha (annual %)", title="36-month rolling FF5 estimates")
axes[1].plot(t, roll_ff5["MKT"], label="MKT beta"); axes[1].plot(t, roll_ff5["HML"], label="HML beta")
axes[1].axhline(1, color="k", lw=0.6, ls="--"); axes[1].legend()
fig.tight_layout()
save_fig(fig, "fig6_rolling.pdf")

# %% hedging backtest: hedge month t with the CAPM beta from months t-36 ... t-1
beta_lag = roll_capm["MKT"].shift(1).reindex(df.index)
bt = pd.DataFrame({"unhedged": df["EX_QQQ"], "hedged": df["EX_QQQ"] - beta_lag * df["MKT"],
                   "BEAR": df["BEAR"], "CRISIS": df["CRISIS"]}).dropna(subset=["hedged"])

def he(d):
    """Hedge effectiveness: share of variance removed."""
    return 1 - d["hedged"].var() / d["unhedged"].var()

hedge = pd.Series({
    "All months": he(bt),
    "Normal state (past 24m market return >= 0)": he(bt[bt["BEAR"] == 0]),
    "Bear state (past 24m market return < 0)": he(bt[bt["BEAR"] == 1]),
    "Most bearish third (Crisis top tercile)": he(bt[bt["CRISIS"] >= bt["CRISIS"].quantile(2 / 3)]),
    "Test period only": he(bt.loc[bt.index >= test.index[0]]),
}, name="Hedge effectiveness")
print(hedge.round(3))
beta_full = ols(df["EX_QQQ"], df[["MKT"]])[1]   # full-sample beta uses future data: not implementable
print("Full-sample (look-ahead) beta, for comparison only:",
      round(1 - (df["EX_QQQ"] - beta_full * df["MKT"]).var() / df["EX_QQQ"].var(), 3))
save_tex(hedge.to_frame(), "tab8_hedge.tex")

# %% all 63 factor subsets, ranked on the training set only
ALL = FACTORS
y = train["EX_QQQ"]; m = len(y)
res = []
for k in range(1, len(ALL) + 1):
    for cols in combinations(ALL, k):
        cols = list(cols); b = ols(y, train[cols])
        sse = ((y - predict(b, train[cols])) ** 2).sum(); p = len(b)
        res.append({"factors": " + ".join(cols), "p": p,
                    "AIC": m * np.log(sse / m) + 2 * p, "BIC": m * np.log(sse / m) + p * np.log(m),
                    "MSPR (test)": ((test["EX_QQQ"] - predict(b, test[cols])) ** 2).mean()})
subsets = pd.DataFrame(res)
print(subsets.nsmallest(5, "BIC").round(2).to_string(index=False))
save_tex(subsets.nsmallest(5, "BIC"), "tab9_all_subsets.tex", float_format="%.2f", index=False)
