from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.config import END_INCLUSIVE, FRED_SERIES, RAW_END_EXCLUSIVE, START, YF_FACTORS
from utilities.utils import ensure_dir, project_calendar, safe_log

OUTPUT = ensure_dir(ROOT / "2 Covariates" / "output")
IMG = ensure_dir(ROOT / "2 Covariates" / "img")


def flatten_yfinance(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.stack(level=1, future_stack=True).rename_axis(["Date", "ticker"]).reset_index()
    else:
        raw = raw.copy()
        raw["ticker"] = ticker
        raw = raw.rename_axis("Date").reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    return raw


def choose_price(df: pd.DataFrame) -> pd.Series:
    adj = pd.to_numeric(df.get("adj_close", pd.Series(index=df.index, dtype=float)), errors="coerce")
    close = pd.to_numeric(df.get("close", pd.Series(index=df.index, dtype=float)), errors="coerce")
    return adj.where(adj > 0).fillna(close.where(close > 0))


def download_yf_factors() -> pd.DataFrame:
    rows = []
    for ticker in YF_FACTORS:
        raw = yf.download(ticker, start=START, end=RAW_END_EXCLUSIVE, auto_adjust=False, progress=False, threads=False)
        flat = flatten_yfinance(raw, ticker)
        rows.append(flat)
        print(f"Yahoo {ticker}: {len(flat)} rows")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(OUTPUT / "yfinance_covariates_raw.csv", index=False)
    return out


def download_fred() -> pd.DataFrame:
    frames = []
    for series_id, label in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}&coed={END_INCLUSIVE}"
        frame = pd.read_csv(url)
        frame = frame.rename(columns={"observation_date": "Date", series_id: label})
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame[label] = pd.to_numeric(frame[label].replace(".", np.nan), errors="coerce")
        frames.append(frame)
        print(f"FRED {series_id}: {len(frame)} rows")
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="Date", how="outer")
    out = out.sort_values("Date")
    out.to_csv(OUTPUT / "fred_covariates_raw.csv", index=False)
    return out


def build_covariates(yf_raw: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({"Date": project_calendar(START, END_INCLUSIVE)})
    if not yf_raw.empty:
        yf_raw["date"] = pd.to_datetime(yf_raw["date"]).dt.tz_localize(None)
        yf_raw["price"] = yf_raw.groupby("ticker", group_keys=False).apply(choose_price, include_groups=False)
        wide = yf_raw.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()
        wide = wide.rename(columns=YF_FACTORS)
        ret = safe_log(wide).diff().add_suffix("_logret")
        yf_features = pd.concat([wide, ret], axis=1).reset_index().rename(columns={"date": "Date"})
        calendar = calendar.merge(yf_features, on="Date", how="left")
    if not fred.empty:
        fred = fred.rename(columns={"date": "Date"})
        fred["Date"] = pd.to_datetime(fred["Date"]).dt.tz_localize(None)
        calendar = calendar.merge(fred, on="Date", how="left")
    calendar = calendar.sort_values("Date")
    for col in calendar.columns:
        if col != "Date":
            calendar[col] = pd.to_numeric(calendar[col], errors="coerce").ffill()
    for col in [c for c in calendar.columns if c.endswith("_yield") or c.endswith("_rate") or c.endswith("_spread")]:
        calendar[f"{col}_delta"] = calendar[col].diff()
    calendar.to_csv(OUTPUT / "covariates_daily.csv", index=False)
    return calendar


def save_correlation_matrix(covariates: pd.DataFrame) -> None:
    numeric = covariates.drop(columns=["Date"], errors="ignore").select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    usable = [
        col
        for col in numeric.columns
        if numeric[col].notna().sum() >= 30 and numeric[col].nunique(dropna=True) > 1
    ]
    if not usable:
        return
    corr = numeric[usable].corr(min_periods=30)
    size = max(10, min(24, 0.28 * len(corr.columns)))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.index, fontsize=5)
    ax.set_title("Covariates correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    fig.savefig(IMG / "correlation_matrix.png", dpi=180)
    plt.close(fig)


def main() -> int:
    yf_path = OUTPUT / "yfinance_covariates_raw.csv"
    fred_path = OUTPUT / "fred_covariates_raw.csv"
    yf_raw = pd.read_csv(yf_path) if yf_path.exists() else download_yf_factors()
    fred = pd.read_csv(fred_path) if fred_path.exists() else download_fred()
    cov = build_covariates(yf_raw, fred)
    save_correlation_matrix(cov)
    print(f"Saved {OUTPUT / 'covariates_daily.csv'} with {len(cov)} rows and {len(cov.columns) - 1} covariates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
