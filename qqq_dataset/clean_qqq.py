"""Clean QQQ / SPY + Fama-French data following the QQQ workflow checklist.
Usage: python clean_qqq.py raw_qqq out_qqq"""
import sys, pandas as pd, numpy as np
from pathlib import Path
raw, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
qc = []
px = pd.read_csv(raw/"qqq_spy_adjclose_monthly.csv", dtype={"ym": str}).set_index("ym")
qc.append(f"Dropped partial month {px.index.max()} from ETF prices."); px = px.drop(index=px.index.max())
ret = 100*(px/px.shift(1)-1)
ff = pd.concat([pd.read_csv(raw/f, dtype={"ym": str}) for f in ["ff5_mom_199903_200206.csv", "ff5_mom_200207_202609.csv"]]).set_index("ym")
ff.columns = ["MKT","SMB","HML","RMW","CMA","RF","MOM"]
assert ff.index.is_unique
hi = pd.read_csv(raw/"ff_industry10_hitec_vw.csv", dtype={"ym": str}).set_index("ym")
df = ret.join(ff, how="inner").join(hi, how="inner")
df = df[df["QQQ"].notna()]
qc.append(f"Sample: {df.index.min()} to {df.index.max()} ({len(df)} months). First return 1999-05: Yahoo's first QQQ month-end price is 1999-04 (no March 1999 bar).")
df["EX_QQQ"] = df["QQQ"]-df["RF"]; df["EX_SPY"] = df["SPY"]-df["RF"]
df["SPREAD"] = df["QQQ"]-df["SPY"]
df["TECH"] = df["HiTec"]-df["RF"]          # excess return of FF value-weighted high-tech industry portfolio
df = df.rename(columns={"QQQ":"R_QQQ","SPY":"R_SPY","HiTec":"R_HITEC"})
ym = df.index.astype(int)
win = lambda a,b: (ym>=a)&(ym<=b)
df["D_CRISIS"] = (win(200003,200210)|win(200710,200903)|win(202002,202004)|win(202201,202210)).astype(int)
df["D_DOTCOM"]=win(200003,200210).astype(int); df["D_GFC"]=win(200710,200903).astype(int)
df["D_COVID"]=win(202002,202004).astype(int); df["D_HIKE2022"]=win(202201,202210).astype(int)
df.insert(0,"date",pd.to_datetime(df.index,format="%Y%m")+pd.offsets.MonthEnd(0)); df.index.name="ym"
cols=["date","R_QQQ","R_SPY","EX_QQQ","EX_SPY","SPREAD","MKT","SMB","HML","RMW","CMA","MOM","RF","R_HITEC","TECH","D_CRISIS","D_DOTCOM","D_GFC","D_COVID","D_HIKE2022"]
df=df[cols]
# QC
full=pd.period_range(df["date"].min(),df["date"].max(),freq="M")
qc.append(f"Calendar continuity: {len(full)} expected, {len(df)} present -> {'OK' if len(full)==len(df) else 'GAPS'}")
qc.append(f"Missing values: {int(df.drop(columns='date').isna().sum().sum())}")
qc.append(f"Sanity: cor(EX_SPY, MKT) = {df.EX_SPY.corr(df.MKT):.4f}; sd ratio = {df.EX_SPY.std()/df.MKT.std():.3f} (workflow expects ~0.99 / ~1)")
qc.append(f"Sanity: cor(EX_QQQ, TECH) = {df.EX_QQQ.corr(df.TECH):.4f} (QQQ is tech-heavy, expect high)")
for c in ["R_QQQ","R_SPY"]:
    qc.append(f"{c}: mean {df[c].mean():.3f}%  sd {df[c].std():.3f}%  min {df[c].min():.2f}% ({df[c].idxmin()})  max {df[c].max():.2f}% ({df[c].idxmax()})")
zero=[(c,i) for c in ["R_QQQ","R_SPY"] for i in df.index if df.at[i,c]==0]
if zero: qc.append("Flag: exactly-zero monthly returns (likely stale Yahoo price): "+", ".join(f"{c} {i}" for c,i in zero))
qc.append(f"Crisis months: {int(df.D_CRISIS.sum())} of {len(df)}")
qc.append(f"Train/test split (70/30 by time): train {df.index[0]}-{df.index[int(len(df)*0.7)-1]}, test {df.index[int(len(df)*0.7)]}-{df.index[-1]}")
df.round(4).to_csv(out/"qqq_spy_ff_monthly.csv")
(out/"qc_report.txt").write_text("\n".join(qc)+"\n"); print("\n".join(qc))
