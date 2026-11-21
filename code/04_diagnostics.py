"""Step 4 - Diagnostics and remedial measures for the final model (needs statsmodels, scipy).
Run:  python code/04_diagnostics.py
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

df = load()
FINAL = f"EX_QQQ ~ {F5}"
res = smf.ols(FINAL, data=df).fit()
n, p = int(res.nobs), len(res.params)
NW = nw_lags(n)

# %% diagnostic plots
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].scatter(res.fittedvalues, res.resid, s=10)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
sm.qqplot(res.resid, line="45", fit=True, ax=axes[1])
axes[1].set_title("Normal Q-Q of residuals")
plot_acf(res.resid, lags=12, ax=axes[2])
axes[2].set_title("Residual ACF")
fig.tight_layout()
save_fig(fig, "figA1_residual_diagnostics.pdf")

fig = sm.graphics.plot_partregress_grid(res, fig=plt.figure(figsize=(10, 7)))
save_fig(fig, "figA2_added_variable.pdf")

# %% tests
bp_lm, bp_p, _, _ = het_breuschpagan(res.resid, res.model.exog)
sw_stat, sw_p = stats.shapiro(res.resid)
print(f"Breusch-Pagan: LM = {bp_lm:.2f}, p = {bp_p:.4f}   (p < 0.05 -> heteroskedasticity)")
print(f"Durbin-Watson: {durbin_watson(res.resid):.3f}   (about 2 -> no AR(1) autocorrelation)")
print(f"Shapiro-Wilk:  W = {sw_stat:.3f}, p = {sw_p:.4f}   (p < 0.05 -> non-normal)")
print("Ljung-Box:\n", acorr_ljungbox(res.resid, lags=[6, 12]))

# %% VIF (> 10 is serious)
X, names = res.model.exog, res.model.exog_names
vif = pd.Series([variance_inflation_factor(X, i) for i in range(1, X.shape[1])],
                index=names[1:], name="VIF")
print(vif.round(2))
save_tex(vif.to_frame(), "tabA1_vif.tex", float_format="%.2f")

# %% influential observations
infl = res.get_influence().summary_frame()
flags = pd.DataFrame({"leverage": infl["hat_diag"] > 2 * p / n,
                      "cooks_d": infl["cooks_d"] > 4 / n,
                      "dffits": infl["dffits"].abs() > 2 * np.sqrt(p / n)})
influential = infl.loc[flags.any(axis=1), ["hat_diag", "cooks_d", "dffits"]]
influential = influential.sort_values("cooks_d", ascending=False)
print(influential.head(10).round(3))
influential.index = influential.index.astype(str)
influential.to_csv(TAB / "influential_months.csv")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.stem(df.index.to_timestamp(), infl["cooks_d"].to_numpy(), markerfmt=" ")
ax.axhline(4 / n, color="r", ls="--", label="4/n")
ax.set(title="Cook's distance by month", ylabel="Cook's D")
ax.legend()
save_fig(fig, "figA3_cooks_distance.pdf")

# %% Newey-West standard errors next to classical ones
res_nw = smf.ols(FINAL, data=df).fit(cov_type="HAC", cov_kwds={"maxlags": NW})
side = pd.DataFrame({"coef": res.params, "se_classical": res.bse, "se_newey_west": res_nw.bse,
                     "p_classical": res.pvalues, "p_newey_west": res_nw.pvalues})
print(side.round(4))
save_tex(side, "tabA2_classical_vs_nw.tex", float_format="%.4f")

# %% orthogonalize CMA on HML (keep the intercept so alpha is unchanged)
aux = smf.ols("CMA ~ HML", data=df).fit()
df["CMA_star"] = df["CMA"] - aux.params["HML"] * df["HML"]
res_orth = smf.ols("EX_QQQ ~ MKT + SMB + HML + RMW + CMA_star", data=df).fit()
print(f"R2 of CMA on HML (overlap share): {aux.rsquared:.3f}")
print(pd.DataFrame({"original": res.params, "orthogonalized": res_orth.params}).round(3))

# %% ridge trace on standardized factors
cols = ["MKT", "SMB", "HML", "RMW", "CMA"]
Z = ((df[cols] - df[cols].mean()) / df[cols].std()).to_numpy()
yc = (df["EX_QQQ"] - df["EX_QQQ"].mean()).to_numpy()
lambdas = np.linspace(0, 200, 81)
path = np.array([np.linalg.solve(Z.T @ Z + lam * np.eye(len(cols)), Z.T @ yc) for lam in lambdas])
fig, ax = plt.subplots(figsize=(6, 4))
for j, c in enumerate(cols):
    ax.plot(lambdas, path[:, j], label=c)
ax.axhline(0, color="k", lw=0.6)
ax.set(xlabel="lambda", ylabel="Standardized coefficient", title="Ridge trace")
ax.legend()
save_fig(fig, "figA4_ridge_trace.pdf")
