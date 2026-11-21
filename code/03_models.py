"""Step 3 - Fit the nested factor models M1-M4 and compare them (needs statsmodels).
Run:  python code/03_models.py   (or run the # %% cells in VS Code)
"""
# %% setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else "code")
from common import *
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df = load()
FORMULAS = {
    "M1 CAPM":    "EX_QQQ ~ MKT",
    "M2 FF3":     f"EX_QQQ ~ {F3}",
    "M3 FF5":     f"EX_QQQ ~ {F5}",
    "M4 FF5+MOM": f"EX_QQQ ~ {F6}",
}
models = {name: smf.ols(f, data=df).fit() for name, f in FORMULAS.items()}

# %% Table 4: model comparison, "coefficient (standard error)"
TERMS = ["Intercept"] + FACTORS
rows = {}
for name, res in models.items():
    col = {t: f"{res.params[t]:.3f} ({res.bse[t]:.3f})" if t in res.params else "" for t in TERMS}
    col.update({"adj. R2": f"{res.rsquared_adj:.3f}", "AIC": f"{res.aic:.1f}",
                "BIC": f"{res.bic:.1f}", "n": f"{int(res.nobs)}"})
    rows[name] = col
compare = pd.DataFrame(rows)
print(compare)
save_tex(compare, "tab4_model_comparison.tex")

# %% sequential partial F tests: M1 vs M2, M2 vs M3, M3 vs M4
print(anova_lm(*models.values()))

# %% final model (change after reading the tests)
final = models["M3 FF5"]
print(final.summary())
(TAB / "final_model_summary.tex").write_text(final.summary().as_latex())
