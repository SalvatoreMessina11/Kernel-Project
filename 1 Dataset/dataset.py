from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.config import (  # noqa: E402
    ACTIVE_START,
    CANDIDATE_BANKS,
    CURRENCY_BY_ASSET_CCY,
    CURRENCY_TICKERS,
    END_INCLUSIVE,
    PRICE_START_GRACE_DAYS,
    RAW_END_EXCLUSIVE,
    START,
)
from utilities.utils import project_calendar, read_alternating_csv, write_alternating_csv  # noqa: E402

OUT = ROOT / "1 Dataset" / "intermediate output"
IMG = ROOT / "1 Dataset" / "img"
BANKS_RAW = OUT / "Banks_raw.csv"
CURRENCY_RAW = OUT / "Currency_raw.csv"
BANKS_USD = OUT / "Banks_USD.csv"
RETURN_USD = OUT / "Return_USD.csv"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)


def flatten_yfinance(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["Date", "Close", "Adj Close", "ticker"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.stack(level=1, future_stack=True).rename_axis(["Date", "ticker"]).reset_index()
    else:
        raw = raw.copy()
        raw["ticker"] = ticker
        raw = raw.rename_axis("Date").reset_index()
    return raw


def choose_price(df: pd.DataFrame) -> pd.Series:
    adj = pd.to_numeric(df.get("Adj Close", pd.Series(index=df.index, dtype=float)), errors="coerce")
    close = pd.to_numeric(df.get("Close", pd.Series(index=df.index, dtype=float)), errors="coerce")
    return adj.where(adj > 0).fillna(close.where(close > 0))


def build_banks_raw() -> None:
    if BANKS_RAW.exists():
        print(f"Banks_raw already present: {BANKS_RAW}")
        return

    downloaded = []
    for asset in CANDIDATE_BANKS:
        raw = yf.download(
            asset.ticker,
            start=START,
            end=RAW_END_EXCLUSIVE,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        flat = flatten_yfinance(raw, asset.ticker)
        downloaded.append(flat)
        print(asset.ticker, len(flat))

    raw_all = pd.concat(downloaded, ignore_index=True)
    raw_all["Date"] = pd.to_datetime(raw_all["Date"]).dt.tz_localize(None)
    cutoff = pd.Timestamp(START) + pd.Timedelta(days=PRICE_START_GRACE_DAYS)
    first_dates = {}
    for asset in CANDIDATE_BANKS:
        group = raw_all[raw_all["ticker"].eq(asset.ticker)].sort_values("Date")
        price = choose_price(group)
        valid = group.loc[price.notna(), "Date"]
        if not valid.empty and valid.min() <= cutoff:
            first_dates[asset.ticker] = valid.min()
    common_start = max(first_dates.values())
    calendar = project_calendar(common_start.strftime("%Y-%m-%d"), END_INCLUSIVE)
    values = pd.DataFrame(index=calendar)
    flags = pd.DataFrame(index=calendar)
    audit_rows = []

    for asset in CANDIDATE_BANKS:
        group = raw_all[raw_all["ticker"].eq(asset.ticker)].sort_values("Date")
        price = choose_price(group)
        series = pd.Series(price.to_numpy(), index=group["Date"])
        valid_first = series.dropna().index.min() if series.notna().any() else pd.NaT
        active = pd.notna(valid_first) and valid_first <= cutoff
        audit_rows.append(
            {
                "ticker": asset.ticker,
                "name": asset.name,
                "currency": asset.currency,
                "region": asset.region,
                "first_valid_price_date": valid_first,
                "used_in_active_universe": active,
            }
        )
        if not active:
            continue
        reindexed = series.reindex(calendar)
        values[asset.ticker] = reindexed.ffill()
        flags[asset.ticker] = reindexed.notna()

    write_alternating_csv(values, flags, BANKS_RAW)
    pd.DataFrame(audit_rows).to_csv(OUT / "bank_universe_audit.csv", index=False)
    print(f"Saved {BANKS_RAW} with {len(values.columns)} active banks from {calendar[0].date()}")


def build_currency_raw() -> None:
    if CURRENCY_RAW.exists():
        print(f"Currency_raw already present: {CURRENCY_RAW}")
        return

    bank_values, _ = read_alternating_csv(BANKS_RAW)
    calendar = bank_values.index
    cur_values = pd.DataFrame(index=calendar)
    cur_flags = pd.DataFrame(index=calendar)
    cur_values["USD"] = 1.0
    cur_flags["USD"] = True
    for label, yf_ticker in CURRENCY_TICKERS.items():
        if label == "USD":
            continue
        raw = yf.download(
            yf_ticker,
            start=calendar[0].strftime("%Y-%m-%d"),
            end=RAW_END_EXCLUSIVE,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        flat = flatten_yfinance(raw, yf_ticker)
        flat["Date"] = pd.to_datetime(flat["Date"]).dt.tz_localize(None)
        price = choose_price(flat)
        series = pd.Series(price.to_numpy(), index=flat["Date"]).reindex(calendar)
        if label in {"JPYUSD", "CHFUSD"}:
            series = 1.0 / series
        cur_values[label] = series.ffill()
        cur_flags[label] = series.notna()
        print(label, int(cur_flags[label].sum()), "tradable days")
    write_alternating_csv(cur_values, cur_flags, CURRENCY_RAW)
    print(f"Saved {CURRENCY_RAW}")


def build_banks_usd() -> None:
    bank_values, bank_flags = read_alternating_csv(BANKS_RAW)
    cur_values, cur_flags = read_alternating_csv(CURRENCY_RAW)
    asset_meta = {asset.ticker: asset for asset in CANDIDATE_BANKS}
    usd_values = pd.DataFrame(index=bank_values.index)
    usd_flags = pd.DataFrame(index=bank_values.index)
    for ticker in bank_values.columns:
        ccy = asset_meta[ticker].currency
        rate_col = CURRENCY_BY_ASSET_CCY[ccy]
        usd_values[ticker] = bank_values[ticker] * cur_values[rate_col]
        usd_flags[ticker] = bank_flags[ticker] & cur_flags[rate_col]
    write_alternating_csv(usd_values, usd_flags, BANKS_USD)
    print(f"Saved {BANKS_USD}")


def save_asset_control_plot() -> None:
    usd_values, usd_flags = read_alternating_csv(BANKS_USD)
    plot_start = pd.Timestamp(ACTIVE_START)
    plot_end = pd.Timestamp(END_INCLUSIVE)
    visible_values = usd_values.loc[(usd_values.index >= plot_start) & (usd_values.index <= plot_end)]
    visible_flags = usd_flags.loc[visible_values.index]
    norm = visible_values / visible_values.iloc[0]
    fig, ax = plt.subplots(figsize=(14, 6))
    norm.plot(ax=ax, lw=0.8, alpha=0.8, legend=False, logy=True)
    non_tradable_any = ~visible_flags.all(axis=1)
    for day in visible_flags.index[non_tradable_any]:
        ax.axvspan(day, day + pd.Timedelta(days=1), color="red", alpha=0.035, lw=0)
    ax.set_xlim(plot_start, plot_end)
    ax.set_title("Normalized USD bank prices, 2005-2025; red bands mark at least one non-tradable asset")
    ax.set_ylabel("Normalized price, log scale")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    asset_trend_path = IMG / "asset_price_tradable_bands.png"
    fig.savefig(asset_trend_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {asset_trend_path}")


def build_returns_usd() -> None:
    usd_values, usd_flags = read_alternating_csv(BANKS_USD)
    returns = np.log(usd_values.where(usd_values > 0)).diff()
    return_flags = usd_flags & usd_flags.shift(1).fillna(False)
    write_alternating_csv(returns, return_flags, RETURN_USD)
    print(f"Saved {RETURN_USD}")


def main() -> int:
    ensure_dirs()
    build_banks_raw()
    build_currency_raw()
    build_banks_usd()
    save_asset_control_plot()
    build_returns_usd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
