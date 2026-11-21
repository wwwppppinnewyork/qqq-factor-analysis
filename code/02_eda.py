"""Step 2 - Descriptive statistics (Tables 1-3) and exploratory plots (saved to output/extra_figures; report figures come from graphic.py).
Run from anywhere:  python code/02_eda.py   (or run the # %% cells in VS Code)
Needs only pandas, numpy, matplotlib.
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
from statistics import NormalDist
import matplotlib.pyplot as plt

df = load()
dates = df.index.to_timestamp()
VARS = ["EX_QQQ", "EX_SPY", "SPREAD"] + FACTORS
print(f"Sample {df.index.min()} to {df.index.max()}, n = {len(df)}")

# %% Table 1: descriptive statistics
def describe(s):
    """Monthly figures in %; annualized mean x 12, volatility x sqrt(12)."""
    return pd.Series({
        "n": s.count(), "Mean": s.mean(), "Mean (ann.)": 12 * s.mean(),
        "SD": s.std(), "Vol (ann.)": np.sqrt(12) * s.std(),
        "Skewness": s.skew(), "Excess kurtosis": s.kurt(),
        "Min": s.min(), "Max": s.max(),
        "Sharpe (ann.)": np.sqrt(12) * s.mean() / s.std(),
    })

table1 = df[VARS].apply(describe).T
print(table1.round(2))
save_tex(table1, "tab1_descriptive.tex", float_format="%.2f")

# %% maximum drawdown
def max_drawdown(r):
    wealth = (1 + r / 100).cumprod()
    return 100 * (1 - wealth / wealth.cummax()).max()

mdd = pd.Series({"QQQ": max_drawdown(df["R_QQQ"]), "SPY": max_drawdown(df["R_SPY"])},
                name="Max drawdown (%)")
print(mdd.round(1))

# %% Table 2: correlations; Table 3: QQQ by bear-market state (terciles of CRISIS)
corr = df[["EX_QQQ"] + FACTORS].corr()
print(corr.round(2))
save_tex(corr, "tab2_correlation.tex", float_format="%.2f")

d = df.dropna(subset=["CRISIS"]).copy()
d["state"] = pd.qcut(d["CRISIS"], 3, labels=["Calm (low)", "Middle", "Bearish (high)"])
table3 = d.groupby("state", observed=True)[["EX_QQQ", "EX_SPY", "MKT"]].agg(["mean", "std"])
print(table3.round(2))
save_tex(table3, "tab3_by_crisis_tercile.tex", float_format="%.2f")

# %% shading helper: months in a Daniel-Moskowitz bear state (PAST24 < 0)
def shade_bear(ax):
    b = (df["BEAR"] == 1).to_numpy().astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, b]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[b, 0]) == -1)
    for s, e in zip(starts, ends):
        ax.axvspan(dates[s], dates[e], color="0.85", zorder=0)

# %% Figure 1: cumulative return
fig, ax = plt.subplots(figsize=(8, 4))
shade_bear(ax)
ax.plot(dates, (1 + df["R_QQQ"] / 100).cumprod(), label="QQQ")
ax.plot(dates, (1 + df["R_SPY"] / 100).cumprod(), label="SPY")
ax.set_yscale("log")
ax.set(ylabel="Growth of $1 (log scale)",
       title="Figure 1. Cumulative return (shaded: past 24-month market return < 0)")
ax.legend()
save_fig(fig, "fig1_cumulative.pdf")

# %% Figure 2: correlation heatmap
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Figure 2. Correlation matrix")
save_fig(fig, "fig2_correlation.pdf")

# %% Figure 3: QQQ vs market, colored by CRISIS, with fitted lines above / below average
fig, ax = plt.subplots(figsize=(6.5, 4.8))
sc = ax.scatter(d["MKT"], d["EX_QQQ"], c=d["CRISIS"], cmap="coolwarm", s=14, alpha=0.8)
fig.colorbar(sc, label="Crisis (z-score, higher = more bearish)")
for mask, label, color in [(d["CRISIS"] <= 0, "Crisis <= 0", "tab:blue"),
                           (d["CRISIS"] > 0, "Crisis > 0", "tab:red")]:
    sub = d[mask]
    slope, intercept = np.polyfit(sub["MKT"], sub["EX_QQQ"], 1)
    xs = np.linspace(sub["MKT"].min(), sub["MKT"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color=color, label=f"{label} (slope {slope:.2f})")
ax.set(xlabel="MKT (%)", ylabel="QQQ excess return (%)",
       title="Figure 3. QQQ vs market by bear-market state")
ax.legend()
save_fig(fig, "fig3_scatter_crisis.pdf")

# %% Figure 4: three dimensions (+ color)
fig = plt.figure(figsize=(6.5, 5))
ax = fig.add_subplot(projection="3d")
p = ax.scatter(d["MKT"], d["HML"], d["EX_QQQ"], c=d["CRISIS"], cmap="coolwarm", s=10, alpha=0.8)
ax.set(xlabel="MKT (%)", ylabel="HML (%)", zlabel="QQQ excess (%)")
ax.set_title("Figure 4. MKT x HML x QQQ (color = Crisis)")
ax.view_init(elev=20, azim=-60)
fig.colorbar(p, shrink=0.6, pad=0.1)
save_fig(fig, "fig4_3d.pdf")

# %% Figure 5: distribution vs normal (Q-Q without scipy)
x = df["EX_QQQ"].dropna().sort_values().to_numpy()
n = len(x)
nd = NormalDist()
theo = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].hist(x, bins=40, density=True, alpha=0.7)
grid = np.linspace(x.min(), x.max(), 200)
mu, sd = x.mean(), x.std(ddof=1)
axes[0].plot(grid, np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color="k")
axes[0].set_title("Histogram vs normal")
axes[1].scatter(theo, x, s=8)
axes[1].plot(theo, mu + sd * theo, color="r")
axes[1].set(xlabel="Theoretical quantiles", ylabel="Sample quantiles", title="Normal Q-Q plot")
fig.suptitle("Figure 5. Distribution of QQQ excess returns")
fig.tight_layout()
save_fig(fig, "fig5_distribution.pdf")
