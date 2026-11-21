"""Figures for the project description, ordered from motivation to the research questions.

  Fig 1  Phenomenon: growth of $1 and drawdowns, QQQ vs SPY          (why it matters)
  Fig 2  Risk and return: risk-return map + rolling Sharpe ratio      (is it just more risk?)
  Fig 3  Decomposition: correlations of QQQ and the six factors       (Q2, Q3)
  Fig 4  Core question: 36-month rolling alpha, CAPM vs FF5           (Q1)
  Fig 5  Deeper: QQQ vs market colored by the bear-market state       (Q4; three dimensions)

Run:  python code/graphic.py      (or run the # %% cells in VS Code)
Needs only pandas, numpy, matplotlib. Saves PDF + PNG to output/figures/.
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
C_QQQ, C_SPY = "#1f4e9c", "#d1495b"

df = load()
dates = df.index.to_timestamp()
WIN = 36


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    if not INTERACTIVE:
        plt.close(fig)


def shade_bear(ax):
    """Shade months with a negative cumulative market return over the previous 24 months."""
    b = (df["BEAR"] == 1).to_numpy().astype(int)
    for s, e in zip(np.flatnonzero(np.diff(np.r_[0, b]) == 1), np.flatnonzero(np.diff(np.r_[b, 0]) == -1)):
        ax.axvspan(dates[s], dates[e], color="0.88", zorder=0, lw=0)


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X)])
    return np.linalg.lstsq(X1, np.asarray(y), rcond=None)[0]


# %% Figure 1: growth of $1 and drawdowns
START = 1000                                           # initial investment, $
wealth = pd.DataFrame({"QQQ": START * (1 + df["R_QQQ"] / 100).cumprod(),
                       "SPY": START * (1 + df["R_SPY"] / 100).cumprod()})
drawdown = 100 * (wealth / wealth.cummax() - 1)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for ax in (a1, a2):
    shade_bear(ax)
a1.plot(dates, wealth["QQQ"], color=C_QQQ, lw=1.6, label=f"QQQ  (final ${wealth['QQQ'].iloc[-1]:,.0f})")
a1.plot(dates, wealth["SPY"], color=C_SPY, lw=1.6, label=f"SPY  (final ${wealth['SPY'].iloc[-1]:,.0f})")
a1.set_yscale("log")
ticks = [250, 500, 1000, 2000, 5000, 10000, 20000]
a1.set_yticks(ticks)
a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
a1.yaxis.set_minor_formatter(plt.NullFormatter())
a1.set_ylabel("Value of $1,000 invested (log scale)")
a1.legend(loc="upper left", frameon=False)
a1.set_title("Figure 1. Growth of \$1,000 invested in May 1999 and drawdowns: QQQ vs SPY", loc="left", fontweight="bold")
a2.fill_between(dates, drawdown["QQQ"], 0, color=C_QQQ, alpha=0.35, label=f"QQQ (max {drawdown['QQQ'].min():.0f}%)")
a2.plot(dates, drawdown["SPY"], color=C_SPY, lw=1.2, label=f"SPY (max {drawdown['SPY'].min():.0f}%)")
a2.set_ylabel("Drawdown (%)")
a2.legend(loc="lower right", frameon=False)
a2.text(0.01, 0.06, "Shaded: past 24-month market return < 0", transform=a2.transAxes, fontsize=8, color="0.35")
fig.tight_layout()
save(fig, "graphic1_growth_drawdown")

# %% Figure 2: risk-return map and rolling Sharpe ratio
assets = {"QQQ": df["EX_QQQ"], "SPY": df["EX_SPY"], "MKT": df["MKT"], "SMB": df["SMB"],
          "HML": df["HML"], "RMW": df["RMW"], "CMA": df["CMA"], "MOM": df["MOM"]}
vol = {k: np.sqrt(12) * v.std() for k, v in assets.items()}
ret = {k: 12 * v.mean() for k, v in assets.items()}
sharpe = lambda s: np.sqrt(12) * s.rolling(WIN).mean() / s.rolling(WIN).std()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.4]})
for k in assets:
    color = C_QQQ if k == "QQQ" else C_SPY if k == "SPY" else "0.45"
    a1.scatter(vol[k], ret[k], s=70 if k in ("QQQ", "SPY") else 35, color=color, zorder=3)
    offset = {"SPY": (-30, -14), "MKT": (6, 4)}.get(k, (5, 4))
    a1.annotate(k, (vol[k], ret[k]), xytext=offset, textcoords="offset points",
                fontweight="bold" if k in ("QQQ", "SPY") else "normal")
x = np.linspace(0, max(vol.values()) * 1.05, 50)
a1.plot(x, x * ret["SPY"] / vol["SPY"], ls="--", color=C_SPY, lw=1, label="SPY's Sharpe ratio")
a1.set(xlabel="Annualized volatility (%)", ylabel="Annualized excess return (%)", xlim=(0, None))
a1.legend(frameon=False, loc="upper left")
a1.set_title("(a) Risk vs return", loc="left")

shade_bear(a2)
a2.plot(dates, sharpe(df["EX_QQQ"]), color=C_QQQ, lw=1.4, label="QQQ")
a2.plot(dates, sharpe(df["EX_SPY"]), color=C_SPY, lw=1.4, label="SPY")
a2.axhline(0, color="k", lw=0.6)
a2.set(ylabel="Sharpe ratio (36-month rolling)")
a2.legend(frameon=False, loc="upper left")
a2.set_title("(b) Risk-adjusted return over time", loc="left")
fig.suptitle("Figure 2. Risk-return profile and 36-month rolling Sharpe ratio",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "graphic2_risk_return")

# %% Figure 3: correlation of QQQ and the six factors (lower triangle; diagonal in gray)
cols = ["EX_QQQ"] + FACTORS
corr = df[cols].corr()
labels = ["QQQ"] + FACTORS
k = len(cols)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("white")
off_diag = np.ma.masked_where(np.triu(np.ones((k, k), dtype=bool)), corr)
im = ax.imshow(off_diag, cmap=cmap, vmin=-0.9, vmax=0.9)
for i in range(k):                                   # diagonal: neutral gray, value 1.00 kept
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="0.9", zorder=2))
ax.set_xticks(range(k), labels)
ax.set_yticks(range(k), labels)
ax.grid(False)
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)
for i in range(k):
    for j in range(i + 1):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, zorder=3,
                color="0.45" if i == j else ("white" if abs(v) > 0.55 else "black"))
cb = fig.colorbar(im, fraction=0.046, pad=0.04, ticks=[-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9])
cb.set_label("Correlation")
cb.outline.set_visible(False)
ax.set_title("Figure 3. Pairwise correlations between QQQ\nand the Fama-French-Carhart factors",
             loc="left", fontweight="bold")
fig.tight_layout()
save(fig, "graphic3_correlation")

# %% Figure 4: 36-month rolling alpha, CAPM vs five-factor model
F5_COLS = ["MKT", "SMB", "HML", "RMW", "CMA"]
n = len(df)
alpha = {"CAPM": [], "Five-factor": []}
for e in range(WIN - 1, n):
    w = df.iloc[e - WIN + 1:e + 1]
    alpha["CAPM"].append(12 * ols(w["EX_QQQ"], w[["MKT"]])[0])
    alpha["Five-factor"].append(12 * ols(w["EX_QQQ"], w[F5_COLS])[0])
t = dates[WIN - 1:]
full_capm = 12 * ols(df["EX_QQQ"], df[["MKT"]])[0]
full_ff5 = 12 * ols(df["EX_QQQ"], df[F5_COLS])[0]

fig, ax = plt.subplots(figsize=(8.5, 4.2))
shade_bear(ax)
ax.plot(t, alpha["CAPM"], color="0.55", lw=1.4, label=f"CAPM (full sample {full_capm:.1f}%/yr)")
ax.plot(t, alpha["Five-factor"], color=C_QQQ, lw=1.6, label=f"Five-factor (full sample {full_ff5:.1f}%/yr)")
ax.axhline(0, color="k", lw=0.8)
ax.set(ylabel="Annualized alpha (%), 36-month window")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Figure 4. 36-month rolling alpha of QQQ: CAPM vs five-factor model", loc="left", fontweight="bold")
ax.text(0.01, 0.03, "Point estimates only; significance is tested in Q1.", transform=ax.transAxes,
        fontsize=8, color="0.35")
fig.tight_layout()
save(fig, "graphic4_rolling_alpha")

# %% Figure 5: QQQ vs market by bear-market state (three dimensions)
d = df.dropna(subset=["CRISIS"])
fig, ax = plt.subplots(figsize=(7.4, 5.2))
sc = ax.scatter(d["MKT"], d["EX_QQQ"], c=d["CRISIS"], cmap="coolwarm", vmin=-2.5, vmax=2.5,
                s=18, alpha=0.85, edgecolor="none")
fig.colorbar(sc, label="Crisis (z-score; higher = more bearish)")
for mask, label, color in [(d["BEAR"] == 0, "Normal: past 24m market return >= 0", "#2b6cb0"),
                           (d["BEAR"] == 1, "Bear: past 24m market return < 0", "#c53030")]:
    sub = d[mask]
    b = ols(sub["EX_QQQ"], sub[["MKT"]])
    xs = np.linspace(sub["MKT"].min(), sub["MKT"].max(), 50)
    ax.plot(xs, b[0] + b[1] * xs, color=color, lw=2, label=f"{label} (beta {b[1]:.2f}, n = {len(sub)})")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set(xlabel="Market excess return, MKT (%)", ylabel="QQQ excess return (%)")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.set_title("Figure 5. QQQ vs market excess returns by bear-market state",
             loc="left", fontweight="bold")
fig.tight_layout()
save(fig, "graphic5_beta_by_state")

print("Saved graphic1-5 (.pdf and .png) to", FIG)
