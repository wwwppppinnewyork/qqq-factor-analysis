"""Step 5 - Hypothesis tests for Q1-Q4 (needs statsmodels).
Each test is reported with classical and Newey-West (HAC) standard errors.
Run:  python code/05_tests.py
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df = load()
NW = nw_lags(len(df))

def fit(formula, data=df, robust=False):
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": NW}) if robust else model.fit()

def scalar(x):
    return float(np.squeeze(x))

# %% Q1  H0: alpha = 0 in each of M1-M4
q1 = {}
for name, rhs in [("M1 CAPM", "MKT"), ("M2 FF3", F3), ("M3 FF5", F5), ("M4 FF5+MOM", F6)]:
    ols, nw = fit(f"EX_QQQ ~ {rhs}"), fit(f"EX_QQQ ~ {rhs}", robust=True)
    lo, hi = nw.conf_int().loc["Intercept"]
    q1[name] = {"alpha (monthly %)": ols.params["Intercept"],
                "alpha (annual %)": 12 * ols.params["Intercept"],
                "95% CI annual (NW)": f"[{12 * lo:.2f}, {12 * hi:.2f}]",
                "p classical": ols.pvalues["Intercept"], "p Newey-West": nw.pvalues["Intercept"]}
q1 = pd.DataFrame(q1).T
print(q1)
save_tex(q1, "tab5_q1_alpha.tex")

# same test as a general linear test: Reduced = no intercept, Full = M3
print(anova_lm(fit(f"EX_QQQ ~ {F5} - 1"), fit(f"EX_QQQ ~ {F5}")))   # F = t^2

# %% Q2  H0: beta_RMW = beta_CMA = 0  (Reduced = M2, Full = M3); momentum: M4 vs M3
m2, m3 = fit(f"EX_QQQ ~ {F3}"), fit(f"EX_QQQ ~ {F5}")
print(anova_lm(m2, m3))
w = fit(f"EX_QQQ ~ {F5}", robust=True).f_test("RMW = 0, CMA = 0")
print(f"Robust Wald (RMW, CMA): F = {scalar(w.fvalue):.3f}, p = {scalar(w.pvalue):.4f}")
print(anova_lm(m3, fit(f"EX_QQQ ~ {F6}")))

# %% Q3  SPREAD = R_QQQ - R_SPY;  H0: all five slopes = 0
spread, spread_nw = fit(f"SPREAD ~ {F5}"), fit(f"SPREAD ~ {F5}", robust=True)
print(spread_nw.summary())
print(f"Classical overall F = {spread.fvalue:.3f}, p = {spread.f_pvalue:.4g}")
j = spread_nw.f_test("MKT = 0, SMB = 0, HML = 0, RMW = 0, CMA = 0")
print(f"Robust joint Wald: F = {scalar(j.fvalue):.3f}, p = {scalar(j.pvalue):.4g}")
side = pd.DataFrame({"QQQ": fit(f"EX_QQQ ~ {F5}").params, "SPY": fit(f"EX_SPY ~ {F5}").params,
                     "SPREAD (QQQ - SPY)": spread.params, "p (NW)": spread_nw.pvalues})
print(side.round(3))
save_tex(side, "tab6_q3_qqq_vs_spy.tex")

# %% Q4  Does QQQ's market beta rise with the bear-market state CRISIS (continuous z-score)?
# Full:    EX = a + g*CRISIS + b1*MKT_c + d*(CRISIS x MKT_c) + other factors
# Reduced: same without the interaction.   H0: d = 0
d4 = df.dropna(subset=["CRISIS"])
FULL4 = "EX_QQQ ~ CRISIS + MKT_c + CRISIS_x_MKT + SMB + HML + RMW + CMA"
RED4 = "EX_QQQ ~ CRISIS + MKT_c + SMB + HML + RMW + CMA"
print(f"Q4 sample: {d4.index.min()} to {d4.index.max()}, n = {len(d4)}")
print(anova_lm(fit(RED4, d4), fit(FULL4, d4)))                 # classical partial F, df (1, n-8)
q4 = fit(FULL4, d4, robust=True)
delta, p_two = q4.params["CRISIS_x_MKT"], q4.pvalues["CRISIS_x_MKT"]
p_one = p_two / 2 if delta > 0 else 1 - p_two / 2                # H1: delta > 0
b_avg = q4.params["MKT_c"]
print(f"delta = {delta:.3f} per 1 SD of Crisis, two-sided p = {p_two:.4f}, one-sided p = {p_one:.4f}")
print(f"market beta: average state = {b_avg:.3f}, Crisis = +1 SD -> {b_avg + delta:.3f}, "
      f"Crisis = +2 SD -> {b_avg + 2 * delta:.3f}")
