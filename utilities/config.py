from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = "2000-01-01"
END_INCLUSIVE = "2025-12-31"
RAW_END_EXCLUSIVE = "2026-01-01"
ACTIVE_START = "2005-01-01"
PRICE_START_GRACE_DAYS = 7
BASE_CURRENCY = "USD"

CALIBRATION_MONTHS = 24
TRAIN_MONTHS = 18
VALIDATION_MONTHS = 6
ACTIVE_MONTHS = 6

MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.07
ANNUALIZATION = 252
RISK_AVERSION = 3.0

KERNEL_MODELS = ["linear", "polynomial", "gaussian"]
STRATEGY_ORDER = ["equal_weight", *KERNEL_MODELS]
STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "linear": "Linear kernel constrained MVE",
    "polynomial": "Polynomial degree-2 kernel constrained MVE",
    "gaussian": "Gaussian RBF kernel constrained MVE",
}
STRATEGY_COLORS = {
    "equal_weight": "#4D4D4D",
    "linear": "#0072B2",
    "polynomial": "#D55E00",
    "gaussian": "#009E73",
}

REGION_ORDER = ["US", "UK", "EU", "CH", "JP"]
REGION_COLORS = {
    "US": "#0072B2",
    "UK": "#E69F00",
    "EU": "#009E73",
    "CH": "#D55E00",
    "JP": "#CC79A7",
}


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    country: str
    region: str
    currency: str


CANDIDATE_BANKS: list[Asset] = [
    Asset("JPM", "JPMorgan Chase", "United States", "US", "USD"),
    Asset("BAC", "Bank of America", "United States", "US", "USD"),
    Asset("WFC", "Wells Fargo", "United States", "US", "USD"),
    Asset("C", "Citigroup", "United States", "US", "USD"),
    Asset("GS", "Goldman Sachs", "United States", "US", "USD"),
    Asset("MS", "Morgan Stanley", "United States", "US", "USD"),
    Asset("USB", "U.S. Bancorp", "United States", "US", "USD"),
    Asset("PNC", "PNC Financial Services", "United States", "US", "USD"),
    Asset("HSBA.L", "HSBC Holdings", "United Kingdom", "UK", "GBP"),
    Asset("BARC.L", "Barclays", "United Kingdom", "UK", "GBP"),
    Asset("LLOY.L", "Lloyds Banking Group", "United Kingdom", "UK", "GBP"),
    Asset("NWG.L", "NatWest Group", "United Kingdom", "UK", "GBP"),
    Asset("STAN.L", "Standard Chartered", "United Kingdom", "UK", "GBP"),
    Asset("BNP.PA", "BNP Paribas", "France", "EU", "EUR"),
    Asset("ACA.PA", "Credit Agricole", "France", "EU", "EUR"),
    Asset("GLE.PA", "Societe Generale", "France", "EU", "EUR"),
    Asset("SAN.MC", "Banco Santander", "Spain", "EU", "EUR"),
    Asset("BBVA.MC", "BBVA", "Spain", "EU", "EUR"),
    Asset("ISP.MI", "Intesa Sanpaolo", "Italy", "EU", "EUR"),
    Asset("UCG.MI", "UniCredit", "Italy", "EU", "EUR"),
    Asset("DBK.DE", "Deutsche Bank", "Germany", "EU", "EUR"),
    Asset("CBK.DE", "Commerzbank", "Germany", "EU", "EUR"),
    Asset("INGA.AS", "ING Groep", "Netherlands", "EU", "EUR"),
    Asset("UBSG.SW", "UBS Group", "Switzerland", "CH", "CHF"),
    Asset("8306.T", "Mitsubishi UFJ Financial Group", "Japan", "JP", "JPY"),
    Asset("8316.T", "Sumitomo Mitsui Financial Group", "Japan", "JP", "JPY"),
    Asset("8411.T", "Mizuho Financial Group", "Japan", "JP", "JPY"),
    Asset("8308.T", "Resona Holdings", "Japan", "JP", "JPY"),
    Asset("8309.T", "Sumitomo Mitsui Trust Holdings", "Japan", "JP", "JPY"),
]


CURRENCY_TICKERS = {
    "USD": None,
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "JPYUSD": "JPY=X",
    "CHFUSD": "CHF=X",
}

CURRENCY_BY_ASSET_CCY = {
    "USD": "USD",
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
    "JPY": "JPYUSD",
    "CHF": "CHFUSD",
}


YF_FACTORS = {
    "^VIX": "vix",
    "^GSPC": "us_broad_equity",
    "KBE": "us_bank_etf",
    "^STOXX50E": "eu_broad_equity",
    "^FCHI": "france_broad_equity",
    "^GDAXI": "germany_broad_equity",
    "FTSEMIB.MI": "italy_broad_equity",
    "^IBEX": "spain_broad_equity",
    "^AEX": "netherlands_broad_equity",
    "EUFN": "eu_financials_etf",
    "^FTSE": "uk_broad_equity",
    "BNKS.L": "uk_bank_etf_candidate",
    "^SSMI": "swiss_broad_equity",
    "^N225": "jp_broad_equity",
    "1615.T": "jp_bank_etf_candidate",
    "IXG": "global_financials_etf_proxy",
    "EURUSD=X": "eurusd",
    "GBPUSD=X": "gbpusd",
    "JPY=X": "jpy_per_usd",
    "CHF=X": "chf_per_usd",
    "^IRX": "us_13w_tbill_yield",
    "^FVX": "us_5y_yield",
    "^TNX": "us_10y_yield",
}


FRED_SERIES = {
    "DGS2": "us_2y_yield",
    "DGS10": "us_10y_yield_fred",
    "DFII10": "us_10y_real_yield",
    "T10YIE": "us_10y_breakeven_inflation",
    "BAA10Y": "us_baa_10y_credit_spread",
    "WALCL": "fed_total_assets",
    "DEXUSEU": "usd_per_eur_fred",
    "DEXUSUK": "usd_per_gbp_fred",
    "DEXJPUS": "jpy_per_usd_fred",
    "ECBDFR": "ecb_deposit_rate",
    "ECBMRRFR": "ecb_main_refinancing_rate",
    "IRLTLT01EZM156N": "euro_area_10y_yield",
    "IUDSOIA": "uk_sonia_rate",
    "IRLTLT01GBM156N": "uk_10y_yield",
    "IRSTCI01JPM156N": "jp_call_money_rate",
    "IRLTLT01JPM156N": "jp_10y_yield",
    "IRLTLT01CHM156N": "ch_10y_yield",
}


CRISIS_WINDOWS = {
    "2008_global_financial_crisis": ("2008-01-01", "2009-03-31"),
    "2012_euro_stress": ("2011-07-01", "2012-12-31"),
    "2020_covid_shock": ("2020-02-01", "2020-06-30"),
}


def metadata_frame():
    import pandas as pd

    return pd.DataFrame([asset.__dict__ for asset in CANDIDATE_BANKS])
