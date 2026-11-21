"""Shared setup for all analysis scripts: paths, data loading, helpers.

Every script starts with `from common import *`. It works both when a script is run
from the terminal (python code/03_models.py) and when cells are run one by one in
VS Code, because the working directory is always moved to the project root.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- paths ----------
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:                                   # running cells interactively
    ROOT = Path.cwd() if (Path.cwd() / "qqq_dataset").exists() else Path.cwd().parent
os.chdir(ROOT)

DATA = ROOT / "qqq_dataset" / "qqq_spy_ff_monthly_crisis.csv"
FIG = ROOT / "output" / "figures"
EXTRA = ROOT / "output" / "extra_figures"   # exploratory / diagnostic plots, not in the report body
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

INTERACTIVE = hasattr(sys, "ps1") or "ipykernel" in sys.modules

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# long names in the csv -> short names used in formulas
SHORT = {
    "qqq_excess_return": "EX_QQQ", "spy_excess_return": "EX_SPY",
    "qqq_minus_spy_return": "SPREAD", "qqq_return": "R_QQQ", "spy_return": "R_SPY",
    "risk_free_rate": "RF", "market_excess_return": "MKT", "size_factor_smb": "SMB",
    "value_factor_hml": "HML", "profitability_factor_rmw": "RMW",
    "investment_factor_cma": "CMA", "momentum_factor_mom": "MOM",
    "hitech_industry_return": "R_HITEC", "hitech_industry_excess_return": "TECH",
    "market_past24_return": "PAST24", "Crisis": "CRISIS",
}
FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
F3 = "MKT + SMB + HML"
F5 = "MKT + SMB + HML + RMW + CMA"
F6 = F5 + " + MOM"


def load():
    """Monthly data with a PeriodIndex and short column names.

    CRISIS  continuous bear-market state: z-score of minus the cumulative market return
            over months t-24..t-1 (larger = more bearish). Missing before 2001-03 unless
            the full French history was used to build the dataset.
    BEAR    1{PAST24 < 0}, the binary Daniel-Moskowitz (2016) state. Used only for
            figure shading, descriptive splits and robustness.
    """
    df = pd.read_csv(DATA, dtype={"year_month": str})
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("year_month"), format="%Y%m"), freq="M")
    df.index.name = "month"
    df = df.rename(columns=SHORT)
    df["BEAR"] = (df["PAST24"] < 0).astype(float).where(df["PAST24"].notna())
    df["MKT_c"] = df["MKT"] - df["MKT"].mean()        # centered market factor
    df["CRISIS_x_MKT"] = df["CRISIS"] * df["MKT_c"]    # Q4 interaction
    return df


def nw_lags(n):
    """Newey-West lag rule of thumb, floor(4 (n/100)^(2/9)); 5 for n of about 300."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def save_fig(fig, name):
    """Save to output/figures; keep the figure open when running cells interactively."""
    import matplotlib.pyplot as plt
    EXTRA.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXTRA / name, bbox_inches="tight")
    if not INTERACTIVE:
        plt.close(fig)


def save_tex(table, name, float_format="%.3f", index=True):
    """Save a DataFrame as a booktabs LaTeX table in output/tables.
    Written by hand so it needs no extra package (pandas' to_latex needs jinja2 >= 3.1.2)."""
    t = table.copy()
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [" ".join(str(x) for x in c) for c in t.columns]
    esc = lambda v: str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    def cell(v, col=None):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v): return ""
            return str(int(v)) if col == "n" else float_format % v
        return esc(v)
    cols = ([esc(t.index.name or "")] if index else []) + [esc(c) for c in t.columns]
    lines = [r"\begin{tabular}{" + "l" * index + "r" * t.shape[1] + "}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in t.iterrows():
        vals = ([esc(idx)] if index else []) + [cell(v, c) for c, v in row.items()]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / name).write_text("\n".join(lines) + "\n")


def _add_code_dir_to_path():
    here = Path(__file__).resolve().parent if "__file__" in globals() else ROOT / "code"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_add_code_dir_to_path()
