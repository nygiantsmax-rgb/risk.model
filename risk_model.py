"""Equity portfolio risk model.

Covariance:   Ledoit-Wolf shrinkage (sklearn) and RiskMetrics EWMA (lambda = 0.94).
Tail risk:    Cornish-Fisher parametric VaR/ES, Gaussian VaR/ES, historical-simulation VaR/ES.
Attribution:  Euler marginal / component VaR per position, plus exact historical component ES.

Conventions (stated once, used everywhere):
  * All VaR / ES figures are 1-day, expressed as POSITIVE fractions of portfolio value (0.031 = 3.1% loss).
  * Covariance is estimated on daily LOG returns (as specified).  Portfolio P&L for historical
    simulation and for the skew/kurtosis fed to Cornish-Fisher uses daily SIMPLE returns
    aggregated with the weights (log returns do not aggregate across assets).  Both parametric
    and historical paths read the same portfolio series.
  * Weights that do not sum to 1 leave the residual in cash (zero return, zero risk) unless
    normalize=True.  Component risk shares are therefore "% of total portfolio risk", all of
    which is carried by the equity sleeve.
  * Mean return is set to zero by default (RiskMetrics convention); include_mean=True uses the
    sample mean of the simple returns.
  * Historical quantile: inverted-CDF order statistic (no interpolation); ES is the mean of all
    returns at or below the VaR return, so ES and VaR are read off the same tail days.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252
CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)
CovMethod = Literal["ledoit_wolf", "ewma"]
METHOD_LABEL: dict[str, str] = {"ledoit_wolf": "LedoitWolf", "ewma": "EWMA"}

# Sector buckets used to order the correlation heatmap. Inferred from the brief (AI
# infrastructure / semiconductors / defensive); PANW does not fit any of the three and gets its own.
SECTORS: dict[str, str] = {
    "MU": "Semis", "AMD": "Semis", "TSM": "Semis",
    "VRT": "AI Infra/Power", "ENS": "AI Infra/Power", "POWL": "AI Infra/Power",
    "PWR": "AI Infra/Power", "GEV": "AI Infra/Power", "CCJ": "AI Infra/Power",
    "LITE": "Semis",  # optical components for AI data centres; trades with the semis complex
    "PANW": "Software/Cyber",
    "AAPL": "Mega-cap Tech", "GOOGL": "Mega-cap Tech",
    "MCD": "Defensive", "WMT": "Defensive",
}
SECTOR_ORDER: tuple[str, ...] = ("Semis", "AI Infra/Power", "Mega-cap Tech", "Software/Cyber", "Defensive")


# --------------------------------------------------------------------------------------------
# 1. Data ingestion & cleaning
# --------------------------------------------------------------------------------------------
def fetch_prices(
    tickers: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int = 3,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Download split/dividend-adjusted closes. `end` is exclusive (yfinance semantics).

    Raises if any ticker is still entirely missing after per-ticker retries: forward fill must
    never be allowed to paper over a ticker that failed to download.
    """
    import yfinance as yf

    # yfinance's shared sqlite tz cache throws "database is locked" under concurrent access.
    yf.set_tz_cache_location(cache_dir or tempfile.mkdtemp(prefix="yf_cache_"))

    def _download(symbols: Sequence[str]) -> pd.DataFrame:
        raw = yf.download(list(symbols), start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
        return close

    prices = _download(tickers).reindex(columns=list(tickers))
    for attempt in range(retries):
        missing = [t for t in tickers if prices[t].isna().all()]
        if not missing:
            break
        time.sleep(1.0 + attempt)
        for t in missing:
            single = _download([t])
            if t in single.columns and not single[t].isna().all():
                prices = prices.join(single[[t]], how="outer", rsuffix="_new")
                prices[t] = prices.pop(f"{t}_new")

    dead = [t for t in tickers if prices[t].isna().all()]
    if dead:
        raise RuntimeError(f"No price data after {retries} retries for: {dead}")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices[list(tickers)].sort_index()


@dataclass(frozen=True)
class CleanData:
    prices: pd.DataFrame
    log_returns: pd.DataFrame
    simple_returns: pd.DataFrame
    ffilled_cells: int


def clean_prices(prices: pd.DataFrame) -> CleanData:
    """Forward-fill interior gaps, drop leading rows that cannot be filled, compute returns."""
    first_valid = prices.apply(pd.Series.first_valid_index)
    interior_nans = sum(int(prices.loc[first_valid[c]:, c].isna().sum()) for c in prices.columns)
    filled = prices.ffill().dropna(how="any")
    log_ret = np.log(filled).diff().iloc[1:]
    return CleanData(filled, log_ret, np.expm1(log_ret), interior_nans)


# --------------------------------------------------------------------------------------------
# 2-3. Covariance estimators
# --------------------------------------------------------------------------------------------
def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage toward a scaled identity. Returns (daily covariance, shrinkage)."""
    lw = LedoitWolf(assume_centered=False).fit(returns.to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns), float(lw.shrinkage_)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics zero-mean EWMA covariance as of the last observation.

    Equivalent to the recursion S_t = lam*S_{t-1} + (1-lam) r_t r_t' started from S_0 = 0,
    with the truncated weights renormalised to sum to one.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must lie in (0, 1)")
    x = returns.to_numpy()
    w = (1.0 - lam) * lam ** np.arange(len(x) - 1, -1, -1)
    w /= w.sum()
    cov = (x * w[:, None]).T @ x
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ewma_vol_series(x: pd.Series, lam: float = 0.94) -> pd.Series:
    """Daily zero-mean EWMA volatility path of a single return series.

    Same weighting as ewma_cov: the value on the last date equals sqrt(w' ewma_cov w)
    when x = log_returns @ w.
    """
    v = x.to_numpy(dtype=float)
    out = np.empty_like(v)
    s = 0.0
    for t, r in enumerate(v):
        s = lam * s + (1.0 - lam) * r * r
        out[t] = s / (1.0 - lam ** (t + 1))
    return pd.Series(np.sqrt(out), index=x.index, name="ewma_vol")


# Reference portfolios for volatility comparison: investable ETF proxies (iShares allocation funds
# AOA = 80/20 equity/bond, AOK = 30/70).
DEFAULT_REFERENCES: dict[str, str] = {"Benchmark": "SPY", "Aggressive": "AOA", "Conservative": "AOK"}
VOL_HORIZONS: dict[str, int] = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}


@dataclass(frozen=True)
class VolComparison:
    paths: pd.DataFrame     # annualised EWMA vol through time, one column per series
    horizons: pd.DataFrame  # annualised realised vol over trailing windows (rows) per series (cols)
    vs_benchmark: pd.Series  # portfolio vol / benchmark vol per horizon
    returns: pd.DataFrame    # date-aligned daily simple returns (Portfolio + references); the one aligned frame


def volatility_comparison(port_returns: pd.Series, port_ewma_vol: pd.Series, ref_prices: pd.DataFrame,
                          references: Mapping[str, str], lam: float = 0.94) -> VolComparison:
    """Compare portfolio volatility with reference ETFs on the same dates.

    The portfolio EWMA path is passed in (report.ewma_vol) rather than recomputed, so the
    dashboard's two volatility charts can never disagree about the portfolio line.
    """
    ref_log = np.log(ref_prices.ffill()).diff().reindex(port_returns.index)
    labels = {f"{name} ({sym})": sym for name, sym in references.items()}
    simple = pd.DataFrame({"Portfolio": port_returns} | {lab: np.expm1(ref_log[s]) for lab, s in labels.items()})
    simple = simple.dropna()
    ann = np.sqrt(TRADING_DAYS)
    paths = pd.DataFrame({"Portfolio": port_ewma_vol.reindex(simple.index)}
                         | {lab: ewma_vol_series(ref_log[s].reindex(simple.index), lam) for lab, s in labels.items()})
    paths = paths * ann
    # A calendar year holds ~250 sessions, not 252: accept a window up to 5% short of its nominal length.
    rows = {h: (simple.iloc[-n:].std(ddof=1) * ann if len(simple) >= int(0.95 * n)
                else pd.Series(np.nan, index=simple.columns))
            for h, n in VOL_HORIZONS.items()}
    horizons = pd.DataFrame(rows).T
    horizons.index.name = "Trailing window"
    bench = next(iter(labels))  # first reference is the benchmark
    return VolComparison(paths, horizons, horizons["Portfolio"] / horizons[bench], simple)


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.to_numpy()))
    return cov / np.outer(sd, sd)


# --------------------------------------------------------------------------------------------
# 4. Tail risk: Cornish-Fisher, Gaussian, historical
# --------------------------------------------------------------------------------------------
def _cf_coefficients(skew: float, exkurt: float) -> np.ndarray:
    """Coefficients c0..c3 of the Cornish-Fisher quantile written as a cubic in z:
    z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36."""
    s, k = skew, exkurt
    return np.array([
        -s / 6.0,
        1.0 - k / 8.0 + 5.0 * s * s / 36.0,
        s / 6.0,
        k / 24.0 - s * s / 18.0,
    ])


def cornish_fisher_quantile(z: float, skew: float, exkurt: float) -> float:
    c = _cf_coefficients(skew, exkurt)
    return float(c[0] + c[1] * z + c[2] * z**2 + c[3] * z**3)


def _normal_partial_moments(z: float, kmax: int) -> np.ndarray:
    """M_k = E[Z^k ; Z <= z] for standard normal Z, k = 0..kmax.
    M_0 = Phi(z), M_1 = -phi(z), M_k = -z^(k-1) phi(z) + (k-1) M_(k-2)."""
    phi, Phi = stats.norm.pdf(z), stats.norm.cdf(z)
    m = np.empty(kmax + 1)
    m[0] = Phi
    if kmax >= 1:
        m[1] = -phi
    for k in range(2, kmax + 1):
        m[k] = -(z ** (k - 1)) * phi + (k - 1) * m[k - 2]
    return m


def cornish_fisher_tail_mean(z: float, skew: float, exkurt: float) -> float:
    """E[q_CF(U) | U <= Phi(z)]: closed-form average of the CF quantile over the tail."""
    c = _cf_coefficients(skew, exkurt)
    m = _normal_partial_moments(z, 3)
    return float(c @ m / m[0])


def cornish_fisher_is_valid(skew: float, exkurt: float) -> bool:
    """Domain of validity (cf. Maillard 2012): the CF quantile is monotone increasing in z.

    cf'(z) = c1 + 2 c2 z + 3 c3 z^2 > 0 for all z  <=>  c3 > 0 and c2^2 < 3 c1 c3
    (or the degenerate linear case c3 = c2 = 0, c1 > 0).
    """
    _, c1, c2, c3 = _cf_coefficients(skew, exkurt)
    if c3 == 0.0:
        return c2 == 0.0 and c1 > 0.0
    return bool(c3 > 0.0 and c2 * c2 < 3.0 * c1 * c3)


@dataclass(frozen=True)
class ParametricRisk:
    var: float
    es: float
    quantile: float  # standardised (negative) return quantile used for VaR


def parametric_var_es(sigma: float, mu: float, conf: float,
                      skew: float = 0.0, exkurt: float = 0.0) -> ParametricRisk:
    """Cornish-Fisher VaR/ES. With skew = exkurt = 0 this is exactly the Gaussian result."""
    z = float(stats.norm.ppf(1.0 - conf))
    q = cornish_fisher_quantile(z, skew, exkurt)
    tail = cornish_fisher_tail_mean(z, skew, exkurt)
    return ParametricRisk(var=-(mu + sigma * q), es=-(mu + sigma * tail), quantile=q)


@dataclass(frozen=True)
class HistoricalRisk:
    var: float
    es: float
    tail_mask: np.ndarray
    n_tail: int


def historical_var_es(port_returns: np.ndarray, conf: float) -> HistoricalRisk:
    r = np.asarray(port_returns, dtype=float)
    # k-th worst return, k = ceil(n * alpha). Rounded first: 1 - 0.95 is 0.05000000000000004 in
    # floating point, which would push k past an exact integer (e.g. n = 260 -> 14 instead of 13).
    k = max(1, int(np.ceil(round(len(r) * (1.0 - conf), 9))))
    var_ret = float(np.partition(r, k - 1)[k - 1])
    mask = r <= var_ret
    return HistoricalRisk(var=-var_ret, es=-float(r[mask].mean()), tail_mask=mask, n_tail=int(mask.sum()))


# --------------------------------------------------------------------------------------------
# 5. Risk attribution
# --------------------------------------------------------------------------------------------
def euler_var_components(weights: np.ndarray, cov: np.ndarray, mu_vec: np.ndarray,
                         q: float) -> tuple[np.ndarray, np.ndarray]:
    """Marginal and component VaR for VaR(w) = -(w'mu + q * sqrt(w'Cw)).

    The standardised quantile q (Cornish-Fisher or Gaussian) is held fixed, i.e. portfolio
    skew/kurtosis are treated as constant under a marginal weight change. VaR(w) is then
    homogeneous of degree 1, so components sum exactly to total VaR (Euler).
    """
    cw = cov @ weights
    sigma_p = float(np.sqrt(weights @ cw))
    marginal = -(mu_vec + q * cw / sigma_p)
    return marginal, weights * marginal


def historical_component_es(weights: np.ndarray, simple_returns: np.ndarray,
                            tail_mask: np.ndarray) -> np.ndarray:
    """Exact Euler decomposition of historical ES: -w_i * E[r_i | portfolio in tail]."""
    return -weights * simple_returns[tail_mask].mean(axis=0)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
@dataclass
class RiskModelConfig:
    tickers: list[str] = field(default_factory=list)   # no built-in portfolio: callers supply holdings
    weights: list[float] = field(default_factory=list)
    end: pd.Timestamp | None = None           # exclusive; default = today (drops any partial bar)
    lookback_years: int = 1                   # used only when `start` is None
    start: pd.Timestamp | None = None         # inclusive window start; overrides lookback_years
    ewma_lambda: float = 0.94
    primary_method: CovMethod = "ledoit_wolf"
    normalize_weights: bool = False
    include_mean: bool = False
    notional: float | None = None             # portfolio value in $, for dollar VaR/ES


MIN_WINDOW_OBS = 100  # below this the 99% historical tail rests on a single day


def window_bounds(cfg: RiskModelConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start inclusive, end exclusive) of the estimation window. The one place it is computed."""
    end = (cfg.end or pd.Timestamp.today()).normalize()
    start = (pd.Timestamp(cfg.start).normalize() if cfg.start is not None
             else end - pd.DateOffset(years=cfg.lookback_years))
    if start >= end:
        raise ValueError(f"window start {start.date()} must be before end {end.date()}")
    return start, end


@dataclass
class RiskReport:
    summary: pd.DataFrame
    components: pd.DataFrame
    corr: dict[str, pd.DataFrame]
    meta: dict[str, object]
    port_returns: pd.Series          # daily simple portfolio returns (cash earns zero)
    ewma_vol: pd.Series              # daily EWMA vol path of the weighted log-return series


class PortfolioRiskModel:
    def __init__(self, config: RiskModelConfig, prices: pd.DataFrame | None = None) -> None:
        if not config.tickers:
            raise ValueError("no holdings: pass tickers and weights")
        if len(config.tickers) != len(config.weights):
            raise ValueError("tickers and weights differ in length")
        self.cfg = config
        w = np.asarray(config.weights, dtype=float)
        self.raw_weight_sum = float(w.sum())
        self.weights = w / w.sum() if config.normalize_weights else w
        self._prices = prices

    @property
    def prices(self) -> pd.DataFrame | None:
        """Raw prices used by the last run (downloaded on first run if not supplied)."""
        return self._prices

    def _load(self) -> CleanData:
        if self._prices is None:
            start, end = window_bounds(self.cfg)
            self._prices = fetch_prices(self.cfg.tickers, start, end)
        return clean_prices(self._prices[self.cfg.tickers])

    def run(self) -> RiskReport:
        data = self._load()
        w = self.weights
        simple = data.simple_returns.to_numpy()
        port = simple @ w

        mu_vec = simple.mean(axis=0) if self.cfg.include_mean else np.zeros(len(w))
        mu_p = float(w @ mu_vec)
        skew = float(stats.skew(port, bias=False))
        exkurt = float(stats.kurtosis(port, fisher=True, bias=False))

        lw_cov, shrinkage = ledoit_wolf_cov(data.log_returns)
        covs: dict[str, pd.DataFrame] = {
            "ledoit_wolf": lw_cov,
            "ewma": ewma_cov(data.log_returns, self.cfg.ewma_lambda),
        }

        rows: dict[str, dict[str, float]] = {}
        param: dict[tuple[str, float], ParametricRisk] = {}
        for m, cov in covs.items():
            sigma = float(np.sqrt(w @ cov.to_numpy() @ w))
            col = rows.setdefault(METHOD_LABEL[m], {})
            col["Ann. Volatility"] = sigma * np.sqrt(TRADING_DAYS)
            for c in CONFIDENCE_LEVELS:
                cf = parametric_var_es(sigma, mu_p, c, skew, exkurt)
                gs = parametric_var_es(sigma, mu_p, c)
                param[(m, c)] = cf
                pct = int(round(c * 100))
                col[f"VaR {pct}% (Cornish-Fisher)"] = cf.var
                col[f"ES {pct}% (Cornish-Fisher)"] = cf.es
                col[f"VaR {pct}% (Gaussian)"] = gs.var
                col[f"ES {pct}% (Gaussian)"] = gs.es

        hist_col = rows.setdefault("Historical", {})
        hist_col["Ann. Volatility"] = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS))
        hist: dict[float, HistoricalRisk] = {}
        for c in CONFIDENCE_LEVELS:
            h = historical_var_es(port, c)
            hist[c] = h
            pct = int(round(c * 100))
            hist_col[f"VaR {pct}% (Historical)"] = h.var
            hist_col[f"ES {pct}% (Historical)"] = h.es

        order = ["Ann. Volatility"] + [
            f"{k} {int(round(c * 100))}% ({src})"
            for src in ("Cornish-Fisher", "Gaussian", "Historical")
            for c in CONFIDENCE_LEVELS for k in ("VaR", "ES")
        ]
        summary = pd.DataFrame(rows).reindex(order)
        summary.index.name = "Metric (1-day, fraction of NAV)"

        components = self._components(covs, param, mu_vec, simple, hist)
        # A holding listed after the requested start pushes the common window later; say which one.
        first_valid = self._prices[self.cfg.tickers].apply(pd.Series.first_valid_index)
        latest = first_valid.idxmax()
        late_start = ((str(latest), pd.Timestamp(first_valid[latest]).date())
                      if pd.Timestamp(first_valid[latest]) > window_bounds(self.cfg)[0] + pd.Timedelta(days=7)
                      else None)
        meta = {
            "late_start": late_start,
            "window_start": data.log_returns.index[0].date(),
            "window_end": data.log_returns.index[-1].date(),
            "n_obs": len(port),
            "window_years": len(port) / TRADING_DAYS,
            "short_window": len(port) < MIN_WINDOW_OBS,
            "raw_weight_sum": self.raw_weight_sum,
            "cash_weight": 1.0 - float(w.sum()),
            "normalized": self.cfg.normalize_weights,
            "include_mean": self.cfg.include_mean,
            "lw_shrinkage": shrinkage,
            "port_skew": skew,
            "port_excess_kurtosis": exkurt,
            "cf_valid": cornish_fisher_is_valid(skew, exkurt),
            "ffilled_cells": data.ffilled_cells,
            "hist_tail_days": {c: hist[c].n_tail for c in CONFIDENCE_LEVELS},
            "primary_method": self.cfg.primary_method,
            "ewma_lambda": self.cfg.ewma_lambda,
            "notional": self.cfg.notional,
        }
        port_series = pd.Series(port, index=data.simple_returns.index, name="portfolio")
        vol_path = ewma_vol_series(data.log_returns @ w, self.cfg.ewma_lambda)
        return RiskReport(summary, components, {m: cov_to_corr(c) for m, c in covs.items()}, meta,
                          port_series, vol_path)

    def _components(self, covs: Mapping[str, pd.DataFrame], param: Mapping[tuple[str, float], ParametricRisk],
                    mu_vec: np.ndarray, simple: np.ndarray, hist: Mapping[float, HistoricalRisk]) -> pd.DataFrame:
        w = self.weights
        primary = self.cfg.primary_method
        alt = "ewma" if primary == "ledoit_wolf" else "ledoit_wolf"
        out = pd.DataFrame(index=self.cfg.tickers)
        out["Sector"] = [SECTORS.get(t, "Other") for t in self.cfg.tickers]
        out["Weight"] = w

        for c in CONFIDENCE_LEVELS:
            pct = int(round(c * 100))
            risk = param[(primary, c)]
            marginal, comp = euler_var_components(w, covs[primary].to_numpy(), mu_vec, risk.quantile)
            if not np.isclose(comp.sum(), risk.var, rtol=1e-10, atol=1e-14):
                raise AssertionError(f"Euler components {comp.sum()} != VaR {risk.var}")
            out[f"MVaR_{pct}"] = marginal
            out[f"CompVaR_{pct}"] = comp
            out[f"CompVaR_{pct}_pct"] = comp / risk.var

        _, comp_alt = euler_var_components(w, covs[alt].to_numpy(), mu_vec, param[(alt, 0.99)].quantile)
        out[f"CompVaR_99_pct_{METHOD_LABEL[alt]}"] = comp_alt / param[(alt, 0.99)].var

        h95 = hist[0.95]
        ces = historical_component_es(w, simple, h95.tail_mask)
        if not np.isclose(ces.sum(), h95.es, rtol=1e-10, atol=1e-14):
            raise AssertionError(f"historical component ES {ces.sum()} != ES {h95.es}")
        out["HistCompES_95_pct"] = ces / h95.es
        out["RiskToWeight_99"] = out["CompVaR_99_pct"] / (w / w.sum())

        total = out.drop(columns=["Sector"]).sum()
        total[[c for c in total.index if c.startswith("MVaR")]] = np.nan
        total["RiskToWeight_99"] = np.nan
        out.loc["TOTAL"] = total
        out.loc["TOTAL", "Sector"] = ""
        return out


# --------------------------------------------------------------------------------------------
# Correlation heatmap: ANSI terminal + matplotlib/seaborn
# --------------------------------------------------------------------------------------------
def _in_notebook() -> bool:
    return "ipykernel" in sys.modules


def sector_order(tickers: Sequence[str], weights: Sequence[float]) -> list[str]:
    rank = {s: i for i, s in enumerate(SECTOR_ORDER)}
    wmap = dict(zip(tickers, weights))
    return sorted(tickers, key=lambda t: (rank.get(SECTORS.get(t, ""), len(rank)), -wmap[t]))


def _rgb(c: float) -> tuple[int, int, int]:
    """Diverging blue - light grey - red scale on [-1, 1]."""
    neg, mid, pos = np.array([59, 76, 192]), np.array([235, 235, 235]), np.array([180, 4, 38])
    c = float(np.clip(c, -1.0, 1.0))
    rgb = mid + (pos - mid) * c if c >= 0 else mid + (neg - mid) * (-c)
    return tuple(int(round(v)) for v in rgb)  # type: ignore[return-value]


def render_terminal_heatmap(corr: pd.DataFrame, order: Sequence[str], color: bool = True) -> str:
    """ASCII-only heatmap; colour is carried by ANSI 24-bit background escapes."""
    c = corr.loc[order, order]
    groups = [SECTORS.get(t, "Other") for t in order]
    cell = 6
    lines = []
    header = " " * 7 + "".join(
        ("|" if i and groups[i] != groups[i - 1] else "") + f"{t:>{cell}}" for i, t in enumerate(order))
    lines.append(header)
    for i, row in enumerate(order):
        if i and groups[i] != groups[i - 1]:
            lines.append(" " * 7 + "-" * (len(header) - 7))
        parts = [f"{row:<6} "]
        for j, col in enumerate(order):
            if j and groups[j] != groups[j - 1]:
                parts.append("|")
            v = float(c.iloc[i, j])
            txt = f"{v:>{cell}.2f}"
            if color:
                r, g, b = _rgb(v)
                fg = "97" if abs(v) > 0.6 else "30"
                txt = f"\x1b[48;2;{r};{g};{b}m\x1b[{fg}m{txt}\x1b[0m"
            parts.append(txt)
        parts.append(f"  {groups[i]}")
        lines.append("".join(parts))
    return "\n".join(lines)


def plot_heatmap(corr: pd.DataFrame, order: Sequence[str], title: str, path: Path | None) -> None:
    import matplotlib
    if not _in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    try:
        import seaborn as sns
        sns.heatmap(c, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
                    square=True, linewidths=0.5, cbar_kws={"label": "correlation"})
    except ImportError:
        im = ax.imshow(c.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label="correlation")
        ax.set_xticks(range(len(order)), order, rotation=90)
        ax.set_yticks(range(len(order)), order)
        for i in range(len(order)):
            for j in range(len(order)):
                ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    groups = [SECTORS.get(t, "Other") for t in order]
    for k in range(1, len(order)):
        if groups[k] != groups[k - 1]:
            ax.axhline(k, color="black", lw=1.5)
            ax.axvline(k, color="black", lw=1.5)
    ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    if _in_notebook():
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------------------------
def format_report(report: RiskReport) -> str:
    m = report.meta
    pct = lambda x: f"{x:.2%}"  # noqa: E731
    summ = report.summary.map(lambda x: "" if pd.isna(x) else pct(x))
    comp = report.components.copy()
    for col in comp.columns:
        if col == "Sector":
            continue
        if col.startswith("RiskToWeight"):
            comp[col] = comp[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}x")
        elif col.startswith("MVaR"):
            comp[col] = comp[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            comp[col] = comp[col].map(lambda x: "" if pd.isna(x) else pct(x))

    hist_days = ", ".join(f"{int(round(c * 100))}%: {n} days" for c, n in m["hist_tail_days"].items())
    cash_line = (f"weights normalised to 1.0 (input summed to {m['raw_weight_sum']:.4f})" if m["normalized"]
                 else f"input weights sum to {m['raw_weight_sum']:.4f}; residual {m['cash_weight']:.2%} held as cash (zero risk)")
    lines = [
        "=" * 100,
        "PORTFOLIO RISK REPORT",
        "=" * 100,
        f"Window          : {m['window_start']} -> {m['window_end']}  ({m['n_obs']} daily returns, "
        f"{m['window_years']:.1f} years)",
        *([f"                  starts later than requested: {m['late_start'][0]} has no prices before "
           f"{m['late_start'][1]}"] if m.get("late_start") else []),
        f"Weights         : {cash_line}",
        f"Mean return     : {'sample mean' if m['include_mean'] else 'zero (RiskMetrics convention)'}",
        f"LW shrinkage    : {m['lw_shrinkage']:.4f}   EWMA lambda: {m['ewma_lambda']}",
        f"Portfolio skew  : {m['port_skew']:+.3f}   excess kurtosis: {m['port_excess_kurtosis']:+.3f}   "
        f"Cornish-Fisher valid domain: {'yes' if m['cf_valid'] else 'NO - CF quantile non-monotone, treat CF figures with caution'}",
        f"Forward-filled  : {m['ffilled_cells']} price cells",
        *([f"WARNING         : only {m['n_obs']} daily returns; tail estimates are unreliable below {MIN_WINDOW_OBS}"]
          if m["short_window"] else []),
        "",
        "RISK SUMMARY (1-day horizon, % of NAV; VaR and ES reported as positive losses)",
        summ.to_string(),
        "",
        f"  Historical tail sample sizes -> {hist_days}. 99% historical figures rest on a handful of",
        "  days and carry much wider error bands than the 95% figures.",
        "",
    ]
    if m.get("notional"):
        dollars = (report.summary.drop(index="Ann. Volatility") * float(m["notional"]))  # type: ignore[arg-type]
        lines += [f"RISK IN DOLLARS (portfolio value ${float(m['notional']):,.2f})",  # type: ignore[arg-type]
                  dollars.map(lambda x: "" if pd.isna(x) else f"${x:,.2f}").to_string(), ""]
    lines += [
        f"COMPONENT RISK ({METHOD_LABEL[str(m['primary_method'])]} covariance, Cornish-Fisher quantile)",
        comp.to_string(),
        "",
        "  MVaR_xx         : dVaR/dw_i, change in portfolio VaR (fraction of NAV) per unit of weight.",
        "  CompVaR_xx      : w_i * MVaR_i; sums to total VaR. CompVaR_xx_pct = share of total VaR.",
        "                    Euler decomposition with portfolio skew/kurtosis held fixed (an approximation",
        "                    for the CF quantile; exact for Gaussian VaR). Shares are identical at 95%/99%.",
        "  CompVaR_99_pct_<alt>: same share under the other covariance estimator.",
        "  HistCompES_95_pct: exact share of 95% historical ES (asset mean return on tail days).",
        "  RiskToWeight_99 : risk share / capital share of the equity sleeve. >1 = risk concentrator.",
    ]
    return "\n".join(lines)


def _enable_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        os.system("")  # flips the console into VT-processing mode on Windows 10+
    return True


def resolve_holdings(positions: Path | None, transactions: Path | None, end: pd.Timestamp | None,
                     start: pd.Timestamp | None = None, lookback_years: int = 1,
                     tickers: str | None = None,
                     weights: str | None = None) -> tuple[list[str], list[float], float | None, pd.DataFrame | None]:
    """Pick the holdings source: positions file > transactions file > --tickers/--weights.

    Returns (tickers, weights, notional, prices). Prices are returned only when they had to be
    downloaded to value transaction-rebuilt holdings, so the model reuses them.
    """
    import portfolio_import as pi

    if positions is not None:
        pos = pi.parse_schwab_positions(positions)
        print(f"Imported {len(pos.holdings)} positions from {positions.name} "
              f"(account {pos.account}, as of {pos.as_of.date() if pos.as_of is not None else '?'}, "
              f"value ${pos.total_value:,.2f}, cash {pos.cash / pos.total_value:.2%})")
        for w in pos.warnings:
            print(f"  WARNING: {w}")
        if transactions is not None:
            rec = pi.reconcile(pos, pi.parse_schwab_transactions(transactions))
            bad = rec[rec["Status"] != "ok"]
            print(f"  Reconciliation vs {transactions.name}: {len(rec) - len(bad)}/{len(rec)} symbols match"
                  + ("" if bad.empty else "\n" + bad.to_string()))
        return pos.tickers, pos.weights, pos.total_value, None

    if transactions is not None:
        tx = pi.parse_schwab_transactions(transactions)
        if len(tx.unhandled):
            print(f"  WARNING: {len(tx.unhandled)} rows with unknown share effect ignored: "
                  f"{sorted(tx.unhandled['Action'].unique())}")
        start_ts, end_ts = window_bounds(RiskModelConfig(end=end, start=start, lookback_years=lookback_years))
        qty = tx.holdings_as_of(end_ts - pd.Timedelta(days=1))
        prices = fetch_prices(qty.index.tolist(), start_ts, end_ts)
        last = prices.ffill().iloc[-1]
        tickers, weights, total = pi.weights_from_quantities(qty, last)
        print(f"Rebuilt {len(tickers)} holdings from {transactions.name} as of {prices.index[-1].date()} "
              f"close; invested value ${total:,.2f} (cash balance unknown, weights sum to 100%)")
        return tickers, weights, total, prices

    if not tickers:
        raise SystemExit("No holdings given: pass --positions, --transactions, or --tickers with --weights.")
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if weights:
        wts = [float(x) for x in weights.split(",")]
        if len(wts) != len(syms):
            raise SystemExit(f"--weights has {len(wts)} values for {len(syms)} tickers")
        if sum(wts) > 1.5:                                            # typed as percentages
            wts = [x / 100.0 for x in wts]
    else:
        wts = [1.0 / len(syms)] * len(syms)                           # equal weight
    return syms, wts, None, None


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Equity portfolio risk & covariance model")
    p.add_argument("--method", choices=["ledoit_wolf", "ewma"], default="ledoit_wolf",
                   help="covariance used for the component-risk table (both are always reported)")
    p.add_argument("--lam", type=float, default=0.94, help="EWMA decay factor")
    p.add_argument("--end", type=str, default=None, help="exclusive end date YYYY-MM-DD (default: today)")
    p.add_argument("--start", type=str, default=None,
                   help="inclusive window start YYYY-MM-DD, or 'inception' for the first trade in --transactions "
                        "(default: one year before the end)")
    p.add_argument("--normalize", action="store_true", help="rescale weights to sum to 1 instead of holding cash")
    p.add_argument("--include-mean", action="store_true", help="use sample mean instead of zero mean")
    p.add_argument("--no-color", action="store_true", help="plain ASCII heatmap")
    p.add_argument("--no-plot", action="store_true", help="skip PNG heatmaps")
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "output")
    p.add_argument("--positions", type=Path, help="Schwab positions CSV: holdings, weights and account value")
    p.add_argument("--transactions", type=Path,
                   help="Schwab transactions CSV: reconciled against --positions, or (alone) used to rebuild "
                        "holdings as of the window end, priced at the last close")
    p.add_argument("--tickers", help="comma-separated tickers, e.g. AAPL,MSFT,NVDA (when no Schwab file is given)")
    p.add_argument("--weights", help="comma-separated weights matching --tickers, as fractions or percentages "
                                     "(default: equal weight)")
    p.add_argument("--references", default=",".join(DEFAULT_REFERENCES.values()),
                   help="benchmark,aggressive,conservative ETFs for the volatility comparison "
                        "(default SPY,AOA,AOK; empty string to skip)")
    p.add_argument("--rf", type=float, default=None, help="risk-free rate, e.g. 0.04 (default: ^IRX average)")
    p.add_argument("--erp", type=float, default=0.05, help="equity risk premium for CAPM projections")
    p.add_argument("--horizon", type=int, default=5, help="projection horizon in years")
    p.add_argument("--cap", type=float, default=0.20, help="single-name cap for the rebalancing scenario")
    # Inside Jupyter the kernel injects its own argv (-f <connection file>); ignore it.
    a = p.parse_known_args(argv)[0] if _in_notebook() else p.parse_args(argv)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    end = pd.Timestamp(a.end) if a.end else None
    start = None
    if a.start == "inception":
        if not a.transactions:
            p.error("--start inception needs --transactions")
        import portfolio_import as pi
        start = pi.parse_schwab_transactions(a.transactions).first_date
        print(f"Window starts at account inception: {start.date()}")
    elif a.start:
        start = pd.Timestamp(a.start)
    tickers, weights, notional, prices = resolve_holdings(a.positions, a.transactions, end, start,
                                                          tickers=a.tickers, weights=a.weights)

    cfg = RiskModelConfig(tickers=tickers, weights=weights, end=end, start=start, ewma_lambda=a.lam,
                          primary_method=a.method, normalize_weights=a.normalize,
                          include_mean=a.include_mean, notional=notional)
    model = PortfolioRiskModel(cfg, prices=prices)
    report = model.run()

    print(format_report(report))

    if a.references:
        refs = dict(zip(DEFAULT_REFERENCES, [to_sym.strip().upper() for to_sym in a.references.split(",")]))
        start_ts, end_ts = window_bounds(cfg)
        ref_px = fetch_prices(list(refs.values()), start_ts, end_ts)
        vc = volatility_comparison(report.port_returns, report.ewma_vol, ref_px, refs, cfg.ewma_lambda)
        tbl = vc.horizons.map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
        tbl["Portfolio / benchmark"] = vc.vs_benchmark.map(lambda x: "" if pd.isna(x) else f"{x:.2f}x")
        latest = vc.paths.iloc[-1].map(lambda x: f"{x:.1%}")
        print("\nVOLATILITY VS REFERENCE PORTFOLIOS (annualized, realized over trailing windows)")
        print(tbl.to_string())
        print("  Current EWMA vol: " + ", ".join(f"{k} {v}" for k, v in latest.items()))

        import analysis as an  # lazy: analysis imports this module
        import portfolio_import as pi
        rf, rf_src = (a.rf, "user input") if a.rf is not None else an.fetch_risk_free(start_ts, end_ts)
        top_name = report.components.drop(index="TOTAL")["CompVaR_99_pct"].idxmax()
        positions = pi.parse_schwab_positions(a.positions) if a.positions else None
        tx = pi.parse_schwab_transactions(a.transactions) if a.transactions else None
        lots = pi.fifo_lots(tx, top_name)[0] if tx is not None else None
        result = an.build_analysis(report, cfg, model.prices, vc, rf, rf_src, erp=a.erp, horizon=a.horizon,
                                   cap=a.cap, notional=notional, positions=positions, lots=lots)
        perf = result.performance
        pct_rows = {"Total return", "Annualized return", "Annualized volatility", "Max drawdown", "Worst day"}
        print("\nPERFORMANCE VS REFERENCES (backtest of current weights held constant; not the account's actual return)")
        print(perf.apply(lambda col: [f"{v:.1%}" if i in pct_rows else f"{v:.2f}" for i, v in col.items()]).to_string())
        print(f"\nLONG-TERM UPSIDE/DOWNSIDE ({a.horizon}y; CAPM rf {rf:.2%} [{rf_src}] + beta x ERP {a.erp:.1%})")
        h = result.projections.xs(a.horizon, level="Years")
        print(h[["Beta", "Expected return (CAPM)", "Volatility (log)", "P5", "P50", "P95", "Prob. of loss",
                 "Median annual growth"]].round(3).to_string())
        print("\nREBALANCING SCENARIOS (each re-run through the full model)")
        sc = result.scenarios.copy().astype(object)
        for col in sc.columns:
            if col == "Top risk name":
                continue
            f = ("{:.1f}" if col == "Effective N" else "{:.2f}" if col == "Beta" else
                 "${:,.0f}" if col.endswith("($)") else "{:.2%}" if col.startswith(("VaR", "ES")) else "{:.1%}")
            sc[col] = [("" if pd.isna(v) else f.format(float(v))) for v in result.scenarios[col]]
        print(sc.T.to_string())
        print("\n" + result.headline + "\n")
        print(an.render_text(result))
        a.outdir.mkdir(parents=True, exist_ok=True)
        (a.outdir / "risk_report.md").write_text("# Portfolio risk report\n\n" + an.render_text(result), encoding="utf-8")
        (a.outdir / "ai_analyst_brief.md").write_text(an.ai_brief(result, report, vc, notional, transactions=tx, positions=positions),
                                                         encoding="utf-8")
        print(f"\nWrote risk_report.md and ai_analyst_brief.md to {a.outdir}")

    order = sector_order(cfg.tickers, cfg.weights)
    color = not a.no_color and _enable_ansi()
    for m in ("ewma", "ledoit_wolf"):
        label = f"EWMA lambda={cfg.ewma_lambda} (current regime)" if m == "ewma" else "Ledoit-Wolf (full window)"
        print(f"\nCORRELATION - {label}")
        print(render_terminal_heatmap(report.corr[m], order, color=color))
    print("\n  Sector buckets are inferred: " + " / ".join(SECTOR_ORDER) + " (unlisted tickers: Other).")

    a.outdir.mkdir(parents=True, exist_ok=True)
    report.summary.to_csv(a.outdir / "risk_summary.csv")
    report.components.to_csv(a.outdir / "component_risk.csv")
    if not a.no_plot:
        for m in ("ewma", "ledoit_wolf"):
            plot_heatmap(report.corr[m], order, f"Correlation - {METHOD_LABEL[m]} ({report.meta['window_end']})",
                         a.outdir / f"corr_{m}.png")
    print(f"\nSaved CSVs{'' if a.no_plot else ' and PNG heatmaps'} to {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
