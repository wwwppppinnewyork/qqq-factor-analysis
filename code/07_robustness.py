"""Step 7 - Robustness: do the conclusions survive other choices? (needs statsmodels)
Run after 04_diagnostics.py (needs output/tables/influential_months.csv).
Run:  python code/07_robustness.py
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = load()
FINAL = f"EX_QQQ ~ {F5}"

def nw_fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(data))})

# %% four versions of the final model
drop = pd.PeriodIndex(pd.read_csv(TAB / "influential_months.csv", index_col=0).index, freq="M")
no_dotcom = (df.index < pd.Period("2000-01", "M")) | (df.index > pd.Period("2002-12", "M"))
fits = {
    "Baseline (OLS, NW)": nw_fit(FINAL, df),
    "Drop influential months": nw_fit(FINAL, df.loc[~df.index.isin(drop)]),
    "Drop 2000-2002": nw_fit(FINAL, df.loc[no_dotcom]),
    "Huber robust regression": smf.rlm(FINAL, data=df, M=sm.robust.norms.HuberT()).fit(),
}
table = pd.DataFrame({k: r.params for k, r in fits.items()}).T
table["alpha p-value"] = [r.pvalues["Intercept"] for r in fits.values()]
print(table.round(3))
save_tex(table, "tab10_robustness.tex")

# %% Q4 with three definitions of the bear-market state
d4 = df.dropna(subset=["CRISIS"]).copy()
d4["F_STR"] = 1 / (1 + np.exp(-1.5 * d4["CRISIS"]))   # smooth-transition weight in (0, 1)
rows = {}
for label, state in [("Crisis z-score (baseline)", "CRISIS"),
                     ("Binary: past 24m return < 0 (Daniel-Moskowitz)", "BEAR"),
                     ("Smooth transition F(z), gamma = 1.5", "F_STR")]:
    d4["STATE_x_MKT"] = d4[state] * d4["MKT_c"]
    r = nw_fit(f"EX_QQQ ~ {state} + MKT_c + STATE_x_MKT + SMB + HML + RMW + CMA", d4)
    rows[label] = {"delta": r.params["STATE_x_MKT"], "p (NW)": r.pvalues["STATE_x_MKT"]}
q4_alt = pd.DataFrame(rows).T
print(q4_alt.round(4))
save_tex(q4_alt, "tab11_q4_state_definitions.tex", float_format="%.4f")
