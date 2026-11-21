# QQQ multi-factor dataset (follows qqq_workflow.pdf)

**File to model with:** `qqq_spy_ff_monthly.csv`. One row per month, 1999-05 to 2026-09, 329 months. Units are %.

| Column | Meaning | Source |
|---|---|---|
| `R_QQQ`, `R_SPY` | Monthly total return from month-end adjusted close (dividends included) | Yahoo Finance |
| `EX_QQQ`, `EX_SPY` | Excess return = R − RF | computed |
| `SPREAD` | R_QQQ − R_SPY (for Q3) | computed |
| `MKT` `SMB` `HML` `RMW` `CMA` | Fama/French 5 factors (2x3) | Ken French Data Library |
| `MOM` | Momentum factor | Ken French Data Library |
| `RF` | 1-month T-bill return | Ken French Data Library |
| `R_HITEC` | Value-weighted return of the High-Tech industry portfolio (10 Industry Portfolios) | Ken French Data Library |
| `TECH` | R_HITEC − RF. Optional extra factor for the "tech concentration" extension | computed |
| `D_CRISIS` | 1 in 2000.03–2002.10, 2007.10–2009.03, 2020.02–2020.04, 2022.01–2022.10 (workflow windows); 63 months | constructed |
| `D_DOTCOM` `D_GFC` `D_COVID` `D_HIKE2022` | Individual crisis episodes | constructed |

## Cleaning (workflow checklist)
1. Kept only the monthly YYYYMM rows from the French CSVs.
2. Converted ETF returns to %, the same units as the factors.
3. Used adjusted close, so dividends are included.
4. Merged on YYYYMM. The first return is **1999-05**, not 1999-04: Yahoo has no March 1999 month-end price for QQQ, so the April 1999 return cannot be computed.
5. Dropped months after the latest French release (2026-09) and the partial month 2026-11.
6. Checks:
   - No missing values, no gaps.
   - cor(EX_SPY, MKT) = 0.989 and the sd ratio is 0.966, matching the workflow's expectation.
   - cor(EX_QQQ, TECH) = 0.982.
   - **Flag:** SPY's 2010-11 return is exactly 0, most likely a stale Yahoo price. The true value was near 0.

70/30 time split: train 1999-05 to 2018-06, test 2018-07 to 2026-09.

Re-run the cleaning with `python clean_qqq.py raw_qqq out_qqq`.
