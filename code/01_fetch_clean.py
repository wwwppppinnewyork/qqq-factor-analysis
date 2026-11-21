"""Step 1 - Fetch, clean, and build indicators. One script, three stages.

  fetch()       download raw data
                  * Fama/French 5 factors (2x3), Momentum, 10 Industry Portfolios (zip, from 1926/1963)
                  * QQQ and SPY daily adjusted close from Yahoo Finance -> month-end prices
                saved to qqq_dataset/raw_qqq/
  clean()       monthly returns, excess returns, spread; merge with factors; checks
  indicators()  Past24 and Crisis (bear-market state)

Output: qqq_dataset/qqq_spy_ff_monthly_crisis.csv  (read by code/common.py)

If the download fails (no internet), the script uses the raw files already in raw_qqq/.
Run:  python code/01_fetch_clean.py            (fetch + clean + indicators)
      python code/01_fetch_clean.py --offline  (skip the download)
Needs: pandas, numpy; yfinance for the download (pip install yfinance)
"""
# %% setup
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "qqq_dataset").exists() else Path.cwd().parent
RAW = ROOT / "qqq_dataset" / "raw_qqq"
OUT = ROOT / "qqq_dataset" / "qqq_spy_ff_monthly_crisis.csv"
RAW.mkdir(parents=True, exist_ok=True)

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
ZIPS = {"ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "mom": "F-F_Momentum_Factor_CSV.zip",
        "ind10": "10_Industry_Portfolios_CSV.zip"}


# %% stage 1: fetch
def read_french_monthly(zip_bytes):
    """First monthly block of a French CSV (for industry files: value-weighted returns).
    Layout: notes -> header row starting with ',' -> YYYYMM rows -> blank -> other blocks."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.lstrip().startswith(","))
    body = []
    for l in lines[hdr + 1:]:
        first = l.split(",")[0].strip()
        if not (len(first) == 6 and first.isdigit()):
            break
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join([lines[hdr]] + body)), skipinitialspace=True)
    df.columns = ["ym"] + [c.strip() for c in df.columns[1:]]
    df["ym"] = df["ym"].astype(str).str.strip()
    df = df.set_index("ym").apply(pd.to_numeric, errors="coerce")
    return df.where(df > -99.99)                       # French codes missing as -99.99


def fetch():
    get = lambda name: urllib.request.urlopen(FRENCH + name, timeout=60).read()
    ff5 = read_french_monthly(get(ZIPS["ff5"]))        # Mkt-RF SMB HML RMW CMA RF
    mom = read_french_monthly(get(ZIPS["mom"]))        # Mom
    mom.columns = ["Mom"]
    ind = read_french_monthly(get(ZIPS["ind10"]))[["HiTec"]]
    ff5.join(mom, how="inner").to_csv(RAW / "ff5_mom_full.csv")
    ind.to_csv(RAW / "ff_industry10_hitec_vw.csv")

    import yfinance as yf
    px = yf.download(["QQQ", "SPY"], start="1999-03-01", auto_adjust=True, progress=False)["Close"]
    m = px.resample("ME").last()                        # month-end adjusted close
    m = m[m.index <= pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)]   # complete months
    m.index = m.index.strftime("%Y%m")
    m.index.name = "ym"
    m[["QQQ", "SPY"]].round(4).to_csv(RAW / "qqq_spy_adjclose_monthly.csv")
    print(f"Fetched: factors to {ff5.index.max()}, prices to {m.index.max()}")


# %% stage 2: clean
def load_raw():
    full = RAW / "ff5_mom_full.csv"
    if full.exists():
        ff = pd.read_csv(full, dtype={"ym": str}).set_index("ym")
    else:                                               # files from the earlier manual download
        ff = pd.concat([pd.read_csv(RAW / f, dtype={"ym": str}) for f in
                        ["ff5_mom_199903_200206.csv", "ff5_mom_200207_202609.csv"]]).set_index("ym")
    px = pd.read_csv(RAW / "qqq_spy_adjclose_monthly.csv", dtype={"ym": str}).set_index("ym")
    hi = pd.read_csv(RAW / "ff_industry10_hitec_vw.csv", dtype={"ym": str}).set_index("ym")
    return ff.sort_index(), px.sort_index(), hi.sort_index()


def clean(ff, px, hi):
    ret = (100 * px.pct_change(fill_method=None)).rename(columns={"QQQ": "R_QQQ", "SPY": "R_SPY"})
    df = (ret.dropna().join(ff, how="inner").join(hi, how="inner")
             .rename(columns={"Mkt-RF": "MKT", "Mom": "MOM", "HiTec": "R_HITEC"}))
    df["EX_QQQ"] = df["R_QQQ"] - df["RF"]
    df["EX_SPY"] = df["R_SPY"] - df["RF"]
    df["SPREAD"] = df["R_QQQ"] - df["R_SPY"]
    df["TECH"] = df["R_HITEC"] - df["RF"]

    months = pd.PeriodIndex(pd.to_datetime(df.index, format="%Y%m"), freq="M")
    assert not df.isna().any().any(), "missing values"
    assert len(pd.period_range(months.min(), months.max(), freq="M")) == len(df), "gap in months"
    print(f"Sample {df.index.min()}-{df.index.max()}, n = {len(df)}")
    print(f"cor(EX_SPY, MKT) = {df['EX_SPY'].corr(df['MKT']):.3f} (expect ~0.99), "
          f"sd ratio = {df['EX_SPY'].std() / df['MKT'].std():.3f} (expect ~1)")
    zero = df.index[(df["R_QQQ"] == 0) | (df["R_SPY"] == 0)].tolist()
    if zero:
        print("Flag: exactly-zero returns (possible stale price):", zero)
    return df


# %% stage 3: indicators
def market_history(ff):
    """Longest available market series (Mkt-RF + RF): the downloaded full file, else a
    French 5-factor zip saved in raw_qqq/ or the project folder, else the factor files."""
    for z in [RAW / ZIPS["ff5"], ROOT / ZIPS["ff5"]]:
        if not (RAW / "ff5_mom_full.csv").exists() and z.exists():
            full = read_french_monthly(z.read_bytes())
            print(f"Market history for Crisis from {z.name}: {full.index.min()}-{full.index.max()}")
            return (full["Mkt-RF"] + full["RF"]).astype(float)
    return (ff["Mkt-RF"] + ff["RF"]).astype(float)


def indicators(df, ff):
    """Past24: cumulative total market return over months t-24..t-1 (Daniel & Moskowitz 2016).
    Crisis: -(Past24 - mean) / sd. Computed on the full factor history, so the window
    is complete at the sample start whenever ff5_mom_full.csv is available."""
    mkt = market_history(ff)
    log_r = np.log1p(mkt / 100)
    n_obs = log_r.rolling(24, min_periods=1).count().shift(1)
    past = np.expm1(log_r.rolling(24, min_periods=1).sum().shift(1)).where(n_obs >= 24)
    past24 = (100 * past).reindex(df.index)
    df["PAST24"] = past24.round(4)
    df["CRISIS"] = (-(past24 - past24.mean()) / past24.std()).round(4)
    print(f"Crisis available for {df['CRISIS'].notna().sum()} of {len(df)} months "
          f"(first {df['CRISIS'].first_valid_index()}); bear months (Past24 < 0): {int((past24 < 0).sum())}")
    return df


# %% run all stages
if __name__ == "__main__":
    if "--offline" not in sys.argv:
        try:
            fetch()
        except Exception as e:
            print(f"Download failed ({type(e).__name__}: {e}); using files in {RAW}")
    ff, px, hi = load_raw()
    df = indicators(clean(ff, px, hi), ff)
    LONG = {"R_QQQ": "qqq_return", "R_SPY": "spy_return", "EX_QQQ": "qqq_excess_return",
            "EX_SPY": "spy_excess_return", "SPREAD": "qqq_minus_spy_return",
            "MKT": "market_excess_return", "SMB": "size_factor_smb", "HML": "value_factor_hml",
            "RMW": "profitability_factor_rmw", "CMA": "investment_factor_cma",
            "MOM": "momentum_factor_mom", "RF": "risk_free_rate", "R_HITEC": "hitech_industry_return",
            "TECH": "hitech_industry_excess_return", "PAST24": "market_past24_return", "CRISIS": "Crisis"}
    out = df[list(LONG)].rename(columns=LONG)
    out.insert(0, "month_end_date", pd.to_datetime(out.index, format="%Y%m") + pd.offsets.MonthEnd(0))
    out.index.name = "year_month"
    out.round(4).to_csv(OUT)
    print("Saved", OUT)
