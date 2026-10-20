"""Performance comparison, long-term projections, rebalancing scenarios and a rules-based
recommendation, built on risk_model.py.

Every narrative string is built here once (ASCII only) and consumed by both the CLI and the
dashboard, together with the AI-analyst brief that external assistants read from the page.

Conventions:
  * Historical performance is a BACKTEST of today's weights held constant over the window. It is
    not the account's actual return, and it flatters positions that grew into large weights.
  * Forward projections are lognormal with CAPM expected returns: mu = rf + beta * ERP, drift
    ln(1 + mu), sigma = annualised std of daily log returns. E[W_T] = (1 + mu)^T and the median is
    exp((ln(1 + mu) - sigma^2 / 2) T), so volatility drag is explicit.
  * Rebalancing scenarios are re-run through PortfolioRiskModel, so each has its own moments,
    Cornish-Fisher/historical VaR and component shares.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

import risk_model as rm
from portfolio_import import basis_matches as pi_basis_matches

RF_FALLBACK = 0.04
ERP_DEFAULT = 0.05
HORIZONS_YEARS: tuple[int, ...] = (1, 3, 5, 10)
PERCENTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
PORT = "Portfolio"


@dataclass(frozen=True)
class Thresholds:
    single_weight: float = 0.20        # one position above this share of capital
    single_risk_share: float = 0.35    # ... or above this share of VaR
    sector_risk_share: float = 0.50
    vol_ratio: float = 2.0             # portfolio vol / benchmark vol (longest window)
    max_drawdown: float = 0.25
    vol_drag: float = 0.05             # sigma^2 / 2 per year
    corr_rise: float = 0.05            # EWMA minus Ledoit-Wolf mean pairwise correlation
    scenario_var_cut: float = 0.15     # cap scenario must cut 99% VaR by at least this much to lead


# ------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------
def fetch_risk_free(start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, str]:
    """Average 13-week T-bill yield over the window. ^IRX is quoted in percent, not as a price."""
    try:
        irx = rm.fetch_prices(["^IRX"], start, end)["^IRX"].dropna()
        if len(irx):
            return float(irx.mean()) / 100.0, "13-week T-bill yield (^IRX), window average"
    except Exception:  # network or empty series: fall back, and say so
        pass
    return RF_FALLBACK, f"fallback constant {RF_FALLBACK:.1%} (^IRX unavailable)"


# ------------------------------------------------------------------------------------------
# Historical performance
# ------------------------------------------------------------------------------------------
def wealth_drawdown(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Growth of 1 and drawdown from peak. The starting value 1.0 counts as the first peak."""
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth, wealth / peak - 1.0


def sharpe_standard_error(sharpe: float, years: float) -> float:
    """Lo (2002) iid approximation for an annualised Sharpe ratio."""
    return float(np.sqrt((1.0 + 0.5 * sharpe * sharpe) / years))


def performance_metrics(returns: pd.DataFrame, rf: float, benchmark: str) -> pd.DataFrame:
    """Rows = metrics, columns = series. `returns` are aligned daily simple returns."""
    ann = rm.TRADING_DAYS
    n = len(returns)
    b = returns[benchmark]
    up, down = b > 0, b < 0
    out: dict[str, dict[str, float]] = {}
    for col in returns.columns:
        r = returns[col]
        wealth, dd = wealth_drawdown(r)
        cagr = float(wealth.iloc[-1] ** (ann / n) - 1.0)
        vol = float(r.std(ddof=1) * np.sqrt(ann))
        ex = r - rf / ann
        sharpe = float(ex.mean() * ann / vol)
        downside = float(np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)) * np.sqrt(ann))
        mdd = float(dd.min())
        out[col] = {
            "Total return": float(wealth.iloc[-1] - 1.0),
            "Annualized return": cagr,
            "Annualized volatility": vol,
            "Sharpe ratio": sharpe,
            "Sharpe std. error": sharpe_standard_error(sharpe, n / ann),
            "Sortino ratio": float(ex.mean() * ann / downside) if downside > 0 else np.nan,
            "Max drawdown": mdd,
            "Calmar ratio": cagr / abs(mdd) if mdd < 0 else np.nan,
            "Worst day": float(r.min()),
            "Beta vs benchmark": float(r.cov(b) / b.var()),
            "Correlation vs benchmark": float(r.corr(b)),
            "Upside capture": float(r[up].mean() / b[up].mean()),
            "Downside capture": float(r[down].mean() / b[down].mean()),
        }
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------
# Forward projections
# ------------------------------------------------------------------------------------------
def lognormal_outcomes(mu: float, sigma: float, years: float) -> dict[str, float]:
    """Distribution of the value of 1 after `years`, with E[W] = (1 + mu)^years."""
    drift = np.log1p(mu)
    m = (drift - 0.5 * sigma * sigma) * years
    s = sigma * np.sqrt(years)
    out = {f"P{int(round(p * 100))}": float(np.exp(m + s * stats.norm.ppf(p))) for p in PERCENTILES}
    out["Expected"] = float((1.0 + mu) ** years)
    out["Prob. of loss"] = float(stats.norm.cdf(-m / s))
    out["Median annual growth"] = float(np.exp(drift - 0.5 * sigma * sigma) - 1.0)
    return out


def projection_inputs(r: pd.Series, bench: pd.Series, rf: float, erp: float) -> dict[str, float]:
    beta = float(r.cov(bench) / bench.var())
    sigma = float(np.log1p(r).std(ddof=1) * np.sqrt(rm.TRADING_DAYS))
    return {"Beta": beta, "Expected return (CAPM)": rf + beta * erp, "Volatility (log)": sigma}


def projection_table(returns: pd.DataFrame, benchmark: str, rf: float, erp: float,
                     horizons: tuple[int, ...] = HORIZONS_YEARS) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        inp = projection_inputs(returns[col], returns[benchmark], rf, erp)
        for t in horizons:
            rows.append({"Series": col, "Years": t, **inp,
                         **lognormal_outcomes(inp["Expected return (CAPM)"], inp["Volatility (log)"], t)})
    return pd.DataFrame(rows).set_index(["Series", "Years"])


# ------------------------------------------------------------------------------------------
# Rebalancing scenarios
# ------------------------------------------------------------------------------------------
def cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Cap each weight at `cap`; excess goes to uncapped names pro rata. Total is preserved."""
    w = w.astype(float).copy()
    if (w < 0).any():
        raise ValueError("cap_weights expects long-only weights")
    if cap * len(w) < w.sum() - 1e-12:
        raise ValueError(f"cap {cap:.0%} is infeasible for {len(w)} names summing to {w.sum():.2%}")
    capped = pd.Series(False, index=w.index)
    for _ in range(len(w)):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        capped |= over
        free = ~capped
        w[free] += excess * w[free] / w[free].sum()
    return w


def run_scenarios(cfg: rm.RiskModelConfig, prices: pd.DataFrame, cap: float) -> dict[str, rm.RiskReport]:
    w = pd.Series(cfg.weights, index=cfg.tickers, dtype=float)
    if cfg.normalize_weights:
        w = w / w.sum()
    variants = {
        "Current": w,
        f"Cap {cap:.0%} per name": cap_weights(w, cap),
        "Equal weight": pd.Series(w.sum() / len(w), index=w.index),
    }
    return {name: rm.PortfolioRiskModel(replace(cfg, tickers=list(v.index), weights=v.tolist(),
                                                normalize_weights=False), prices=prices).run()
            for name, v in variants.items()}


def scenario_table(scenarios: Mapping[str, rm.RiskReport], bench: pd.Series, rf: float, erp: float,
                   horizon: int, notional: float | None) -> pd.DataFrame:
    rows = {}
    for name, rep in scenarios.items():
        comp = rep.components.drop(index="TOTAL")
        w = comp["Weight"] / comp["Weight"].sum()
        r = rep.port_returns.reindex(bench.index).dropna()
        b = bench.reindex(r.index)
        inp = projection_inputs(r, b, rf, erp)
        out = lognormal_outcomes(inp["Expected return (CAPM)"], inp["Volatility (log)"], horizon)
        _, dd = wealth_drawdown(r)
        var99 = float(rep.summary.loc["VaR 99% (Cornish-Fisher)", "LedoitWolf"])
        top = comp["CompVaR_99_pct"].idxmax()
        rows[name] = {
            "Max weight": float(comp["Weight"].max()),
            "Effective N": float(1.0 / (w ** 2).sum()),
            "Top risk name": top,
            "Top risk share": float(comp.loc[top, "CompVaR_99_pct"]),
            "Ann. volatility": float(rep.summary.loc["Ann. Volatility", "LedoitWolf"]),
            "VaR 99%": var99,
            "ES 99%": float(rep.summary.loc["ES 99% (Cornish-Fisher)", "LedoitWolf"]),
            "VaR 99% ($)": var99 * notional if notional else np.nan,
            "Beta": inp["Beta"],
            "Expected return (CAPM)": inp["Expected return (CAPM)"],
            "Median annual growth": out["Median annual growth"],
            f"Prob. of loss ({horizon}y)": out["Prob. of loss"],
            "Max drawdown (backtest)": float(dd.min()),
        }
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------
# Markowitz efficient frontier
# ------------------------------------------------------------------------------------------
MU_SOURCES = ("capm", "historical")


@dataclass
class Frontier:
    curves: dict[str, pd.DataFrame]   # name -> rows of Volatility, Expected return, Sharpe + weights
    assets: pd.DataFrame              # per holding: Volatility, Expected return, Beta
    points: pd.DataFrame              # named portfolios: Volatility, Expected return, Sharpe
    weights: pd.DataFrame             # weights of the named portfolios (rows) per holding (cols)
    mu_source: str
    rf: float
    invested: float                   # equity weight held fixed; the rest is cash earning rf
    gaps: dict[str, float]            # current vs long-only frontier: return gap at same vol, vol gap at same return


def frontier_inputs(prices: pd.DataFrame, tickers: list[str], bench: pd.Series, rf: float, erp: float,
                    mu_source: str = "capm") -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Annualised expected returns, Ledoit-Wolf covariance (same estimator as the model) and betas.

    CAPM: mu_i = rf + beta_i * ERP, beta from daily simple returns on the dates `bench` covers, so a
    portfolio's w'mu + cash*rf equals rf + beta_p * ERP exactly. Historical: annualised mean of the
    same simple returns (noisy over a short window; shown for comparison only).
    """
    if mu_source not in MU_SOURCES:
        raise ValueError(f"mu_source must be one of {MU_SOURCES}")
    data = rm.clean_prices(prices[tickers])
    cov = rm.ledoit_wolf_cov(data.log_returns)[0] * rm.TRADING_DAYS
    simple = data.simple_returns.reindex(bench.index).dropna()
    b = bench.reindex(simple.index)
    beta = simple.apply(lambda c: c.cov(b) / b.var())
    mu = rf + beta * erp if mu_source == "capm" else simple.mean() * rm.TRADING_DAYS
    return mu.rename("Expected return"), cov, beta.rename("Beta")


def _portfolio_point(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, cash: float, rf: float) -> tuple[float, float, float]:
    vol = float(np.sqrt(w @ cov @ w))
    ret = float(w @ mu + cash * rf)
    return vol, ret, (ret - rf) / vol if vol > 0 else np.nan


def _solve(objective, w0: np.ndarray, bounds, constraints) -> np.ndarray:
    from scipy.optimize import minimize
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    return res.x


def frontier_curve(mu: np.ndarray, cov: np.ndarray, invested: float, rf: float, upper: float,
                   n_points: int = 40) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Long-only frontier with sum(w) = invested and 0 <= w <= upper.

    Returns (curve rows, min-variance weights, max-Sharpe weights).
    """
    n = len(mu)
    if upper * n < invested - 1e-12:
        raise ValueError(f"per-name limit {upper:.0%} cannot hold {invested:.0%} across {n} names")
    cash = 1.0 - invested
    bounds = [(0.0, upper)] * n
    budget = {"type": "eq", "fun": lambda w: w.sum() - invested}
    w0 = np.full(n, invested / n)
    w_min = _solve(lambda w: w @ cov @ w, w0, bounds, [budget])
    # Highest attainable return: fill the highest-mu names up to the limit (exact for a linear objective).
    w_max, left = np.zeros(n), invested
    for i in np.argsort(-mu):
        w_max[i] = min(upper, left)
        left -= w_max[i]
    r_lo, r_hi = float(w_min @ mu), float(w_max @ mu)
    rows, w_prev = [], w_min
    for target in np.linspace(r_lo, r_hi, n_points):
        tgt = {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
        try:
            w = _solve(lambda w: w @ cov @ w, w_prev, bounds, [budget, tgt])
        except RuntimeError:
            continue
        w_prev = w
        vol, ret, sh = _portfolio_point(w, mu, cov, cash, rf)
        rows.append({"Volatility": vol, "Expected return": ret, "Sharpe": sh, "_w": w})
    w_sh = _solve(lambda w: -_portfolio_point(w, mu, cov, cash, rf)[2], w_min, bounds, [budget])
    curve = pd.DataFrame(rows)
    return curve, w_min, w_sh


def efficient_frontier(prices: pd.DataFrame, cfg: rm.RiskModelConfig, bench: pd.Series, rf: float,
                       erp: float, cap: float, mu_source: str = "capm",
                       extra: Mapping[str, pd.Series] | None = None) -> Frontier:
    """Frontier over the current holdings, cash held at its current weight.

    `extra` adds named weight vectors (e.g. the rebalancing scenarios) to the points table.
    """
    tickers = list(cfg.tickers)
    w_cur = pd.Series(cfg.weights, index=tickers, dtype=float)
    if cfg.normalize_weights:
        w_cur = w_cur / w_cur.sum()
    invested = float(w_cur.sum())
    mu_s, cov_df, beta = frontier_inputs(prices, tickers, bench, rf, erp, mu_source)
    mu, cov = mu_s.reindex(tickers).to_numpy(), cov_df.loc[tickers, tickers].to_numpy()
    cash = 1.0 - invested

    curves, named = {}, {}
    for label, upper in (("Long-only", invested), (f"Long-only, max {cap:.0%} per name", cap)):
        if label != "Long-only" and upper * len(tickers) <= invested + 1e-9:
            continue                                                  # the cap leaves only equal weight: no curve
        curve, w_min, w_sh = frontier_curve(mu, cov, invested, rf, upper)
        weights = pd.DataFrame(list(curve.pop("_w")), columns=tickers)
        curves[label] = pd.concat([curve, weights], axis=1)
        if label == "Long-only":
            named["Min variance"], named["Max Sharpe"] = w_min, w_sh
        else:
            named[f"Max Sharpe (max {cap:.0%})"] = w_sh
    named = {"Current": w_cur.to_numpy(), **{k: pd.Series(v).reindex(tickers).to_numpy()
                                             for k, v in (extra or {}).items()}, **named}
    pts = {k: dict(zip(("Volatility", "Expected return", "Sharpe"), _portfolio_point(w, mu, cov, cash, rf)))
           for k, w in named.items()}
    assets = pd.DataFrame({"Volatility": np.sqrt(np.diag(cov)), "Expected return": mu, "Beta": beta.reindex(tickers)},
                          index=tickers)

    # Distance of the current portfolio from the unconstrained long-only frontier.
    lo = curves["Long-only"].sort_values("Volatility")
    cur = pts["Current"]
    upper_branch = lo[lo["Expected return"] >= lo.loc[lo["Volatility"].idxmin(), "Expected return"]]
    ret_at_vol = float(np.interp(cur["Volatility"], upper_branch["Volatility"], upper_branch["Expected return"]))
    by_ret = upper_branch.sort_values("Expected return")
    vol_at_ret = float(np.interp(cur["Expected return"], by_ret["Expected return"], by_ret["Volatility"]))
    gaps = {"return_gap_same_vol": ret_at_vol - cur["Expected return"],
            "vol_saving_same_return": cur["Volatility"] - vol_at_ret,
            "frontier_return_at_current_vol": ret_at_vol, "frontier_vol_at_current_return": vol_at_ret}
    return Frontier(curves, assets, pd.DataFrame(pts).T, pd.DataFrame(named, index=tickers).T,
                    mu_source, rf, invested, gaps)


# ------------------------------------------------------------------------------------------
# Findings, narrative, recommendation
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Finding:
    severity: str   # "High" | "Medium" | "Low" | "Info"
    title: str
    detail: str
    rule: str       # the threshold that fired, shown to the reader


@dataclass
class Analysis:
    headline: str
    summary: list[str]
    analysis: list[str]
    findings: list[Finding]
    recommendation: list[str]
    performance: pd.DataFrame
    projections: pd.DataFrame
    scenarios: pd.DataFrame
    scenario_reports: dict[str, rm.RiskReport]
    assumptions: dict[str, object]
    facts: dict[str, object] = field(default_factory=dict)  # numbers behind the text, for the AI brief
    frontier: "Frontier | None" = None


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _mean_offdiag(c: pd.DataFrame) -> float:
    a = c.to_numpy()
    n = len(a)
    return float((a.sum() - np.trace(a)) / (n * n - n)) if n > 1 else np.nan


def _frontier_sentence(fr: Frontier, cap: float) -> str:
    g, pts = fr.gaps, fr.points
    cur, ms = pts.loc["Current"], pts.loc["Max Sharpe"]
    src = "CAPM expected returns" if fr.mu_source == "capm" else "historical mean returns over the window"
    s = (f"Efficient frontier (current holdings, long-only, {src}, Ledoit-Wolf risk): at today's volatility of "
         f"{cur['Volatility']:.1%} the frontier offers {g['frontier_return_at_current_vol']:.1%} expected return vs "
         f"{cur['Expected return']:.1%} now; the same {cur['Expected return']:.1%} is reachable at "
         f"{g['frontier_vol_at_current_return']:.1%} volatility ({g['vol_saving_same_return']:.1%} less). "
         f"The maximum-Sharpe mix has Sharpe {ms['Sharpe']:.2f} vs {cur['Sharpe']:.2f} today")
    capped = f"Max Sharpe (max {cap:.0%})"
    if capped in pts.index:
        s += f" ({pts.loc[capped, 'Sharpe']:.2f} with a {cap:.0%} per-name limit)"
    return s + "."


def build_analysis(report: rm.RiskReport, cfg: rm.RiskModelConfig, prices: pd.DataFrame,
                   vc: rm.VolComparison, rf: float, rf_source: str, erp: float = ERP_DEFAULT,
                   horizon: int = 5, cap: float = 0.20, notional: float | None = None,
                   positions=None, lots: pd.DataFrame | None = None, mu_source: str = "capm",
                   th: Thresholds = Thresholds()) -> Analysis:
    # A cap below equal weight cannot hold the invested total (e.g. 20% across 3 names); fall back
    # to equal weight so small portfolios still get every section.
    invested = 1.0 if cfg.normalize_weights else float(np.sum(cfg.weights))
    cap = max(cap, invested / len(cfg.tickers))
    bench = vc.returns.columns[1]
    bench_sym = bench.split("(")[-1].rstrip(")")
    perf = performance_metrics(vc.returns, rf, bench)
    proj = projection_table(vc.returns, bench, rf, erp, tuple(sorted(set(HORIZONS_YEARS) | {horizon})))
    scen_reports = run_scenarios(cfg, prices, cap)
    scen = scenario_table(scen_reports, vc.returns[bench], rf, erp, horizon, notional)
    cap_name = f"Cap {cap:.0%} per name"
    scen_w = {k: r.components.drop(index="TOTAL")["Weight"] for k, r in scen_reports.items() if k != "Current"}
    fr = efficient_frontier(prices, cfg, vc.returns[bench], rf, erp, cap, mu_source, extra=scen_w)

    s, m = report.summary, report.meta
    yrs = float(m.get("window_years", m["n_obs"] / rm.TRADING_DAYS))
    yrs_txt = "one year" if abs(yrs - 1.0) < 0.05 else f"{yrs:.1f} years"
    comp = report.components.drop(index="TOTAL")
    share = comp["CompVaR_99_pct"]
    top = share.idxmax()
    top3 = share.sort_values(ascending=False).head(3)
    sectors = comp.groupby("Sector")[["Weight", "CompVaR_99_pct"]].sum().sort_values("CompVaR_99_pct", ascending=False)
    hedges = share[share < 0].sort_values().index.tolist()
    var99, es99 = float(s.loc["VaR 99% (Cornish-Fisher)", "LedoitWolf"]), float(s.loc["ES 99% (Cornish-Fisher)", "LedoitWolf"])
    hvar99 = float(s.loc["VaR 99% (Historical)", "Historical"])
    lw_vol, ew_vol = float(s.loc["Ann. Volatility", "LedoitWolf"]), float(s.loc["Ann. Volatility", "EWMA"])
    ratio = vc.vs_benchmark.dropna()
    ratio_h, ratio_v = (ratio.index[-1], float(ratio.iloc[-1])) if len(ratio) else ("-", np.nan)
    p, b = perf[PORT], perf[bench]
    pj, bj = proj.loc[(PORT, horizon)], proj.loc[(bench, horizon)]
    corr_rise = _mean_offdiag(report.corr["ewma"]) - _mean_offdiag(report.corr["ledoit_wolf"])
    drag = 0.5 * float(pj["Volatility (log)"]) ** 2
    cur, capped = scen.loc["Current"], scen.loc[cap_name]
    var_cut = 1.0 - float(capped["VaR 99%"]) / float(cur["VaR 99%"])
    value = notional or 0.0
    money = (lambda frac: f" ({_usd(frac * value)})") if value else (lambda frac: "")

    # ---- risk summary --------------------------------------------------------------------
    summary = [
        (f"Portfolio value {_usd(value)}. " if value else "")
        + f"1-day 99% VaR {var99:.2%}{money(var99)} and expected shortfall {es99:.2%}{money(es99)} "
          f"(Ledoit-Wolf covariance, Cornish-Fisher tails); historical 99% VaR {hvar99:.2%}.",
        f"Annualized volatility {lw_vol:.1%} over the full window ({yrs_txt}) and {ew_vol:.1%} on recent (EWMA) data, "
        f"versus {perf.loc['Annualized volatility', bench]:.1%} for the {bench_sym} benchmark: "
        f"{ratio_v:.1f}x the benchmark over {ratio_h}.",
        f"Risk is concentrated: {top} is {comp.loc[top, 'Weight']:.0%} of capital but {share[top]:.0%} of VaR. "
        f"The top three ({', '.join(top3.index)}) carry {top3.sum():.0%} of portfolio risk.",
        "By sector: " + "; ".join(f"{sec} {row['CompVaR_99_pct']:.0%} of risk on {row['Weight']:.0%} of capital"
                                  for sec, row in sectors.iterrows()) + ".",
        f"Beta to {bench_sym} {p['Beta vs benchmark']:.2f}, correlation {p['Correlation vs benchmark']:.2f}. "
        f"Largest peak-to-trough fall in the window {p['Max drawdown']:.1%} vs {b['Max drawdown']:.1%} for {bench_sym}.",
    ]
    if hedges:
        summary.append(f"Diversifiers (negative contribution to VaR): {', '.join(hedges)}.")

    # ---- investment analysis -------------------------------------------------------------
    se = float(p["Sharpe std. error"])
    distinguishable = abs(float(p["Sharpe ratio"]) - float(b["Sharpe ratio"])) > 2 * se
    analysis_text = [
        f"Backtest of today's weights held constant from {m['window_start']} to {m['window_end']} "
        f"(this is not the account's actual return): total return {p['Total return']:.0%} vs "
        f"{b['Total return']:.0%} for {bench_sym}; Sharpe {p['Sharpe ratio']:.2f} (+/- {se:.2f} std. error) vs "
        f"{b['Sharpe ratio']:.2f}. With {yrs_txt} of data the difference is "
        f"{'statistically distinguishable' if distinguishable else 'not statistically distinguishable'} at about 95%, "
        f"and today's weights partly reflect which names already rose, which flatters this backtest.",
        f"Upside/downside capture vs {bench_sym} (daily): {p['Upside capture']:.1f}x on up days and "
        f"{p['Downside capture']:.1f}x on down days. "
        + ("Gains have outpaced losses relative to the market, but both are amplified."
           if p["Upside capture"] > p["Downside capture"] else
           "Losses are amplified more than gains, an unfavourable asymmetry."),
        f"Long-term view ({horizon} years; CAPM expected return = risk-free {rf:.2%} + beta x {erp:.1%} equity "
        f"premium; lognormal): expected return {pj['Expected return (CAPM)']:.1%}/yr at beta {pj['Beta']:.2f}, "
        f"but volatility drag of {drag:.1%}/yr leaves median growth at {pj['Median annual growth']:.1%}/yr vs "
        f"{bj['Median annual growth']:.1%}/yr for {bench_sym}. $1 after {horizon} years: {pj['P5']:.2f} at the "
        f"5th percentile, {pj['P50']:.2f} median, {pj['P95']:.2f} at the 95th; chance of ending below the "
        f"starting value {pj['Prob. of loss']:.0%} vs {bj['Prob. of loss']:.0%} for {bench_sym}.",
        f"Rebalancing: capping each name at {cap:.0%} cuts 99% VaR from {cur['VaR 99%']:.2%} to "
        f"{capped['VaR 99%']:.2%} ({var_cut:.0%} lower) and volatility from {cur['Ann. volatility']:.1%} to "
        f"{capped['Ann. volatility']:.1%}; projected median growth moves from {cur['Median annual growth']:.1%} to "
        f"{capped['Median annual growth']:.1%}/yr. Equal weight: VaR {scen.loc['Equal weight', 'VaR 99%']:.2%}, "
        f"volatility {scen.loc['Equal weight', 'Ann. volatility']:.1%}.",
        _frontier_sentence(fr, cap),
        "The CAPM projection uses an equity-only premium, so bond-heavy references (AOK, AOA) are understated "
        "on return; projections depend entirely on the stated risk-free rate and premium.",
    ]

    # ---- findings (rules, thresholds printed) --------------------------------------------
    findings: list[Finding] = []
    w_top = float(comp.loc[top, "Weight"])
    if w_top > th.single_weight or share[top] > th.single_risk_share:
        findings.append(Finding("High", f"Single-name concentration in {top}",
                                f"{top} is {w_top:.0%} of capital and {share[top]:.0%} of VaR "
                                f"({comp.loc[top, 'RiskToWeight_99']:.2f}x its capital share).",
                                f"weight > {th.single_weight:.0%} or risk share > {th.single_risk_share:.0%}"))
    lead_sector = sectors.index[0]
    if sectors.iloc[0]["CompVaR_99_pct"] > th.sector_risk_share:
        findings.append(Finding("Medium", f"Sector concentration in {lead_sector}",
                                f"{lead_sector} carries {sectors.iloc[0]['CompVaR_99_pct']:.0%} of risk on "
                                f"{sectors.iloc[0]['Weight']:.0%} of capital.",
                                f"sector risk share > {th.sector_risk_share:.0%}"))
    if ratio_v > th.vol_ratio:
        findings.append(Finding("Medium", "Volatility well above the benchmark",
                                f"{ratio_v:.1f}x {bench_sym} over {ratio_h}; a one-standard-deviation year is "
                                f"roughly +/-{lw_vol:.0%}{money(lw_vol)}.",
                                f"vol ratio > {th.vol_ratio:.1f}x"))
    if -p["Max drawdown"] > th.max_drawdown:
        findings.append(Finding("Medium", "Deep drawdown in the backtest",
                                f"Peak-to-trough {p['Max drawdown']:.1%}{money(-float(p['Max drawdown']))} vs "
                                f"{b['Max drawdown']:.1%} for {bench_sym}.",
                                f"max drawdown worse than -{th.max_drawdown:.0%}"))
    if drag > th.vol_drag:
        findings.append(Finding("Medium", "Volatility drag on long-term growth",
                                f"sigma^2/2 = {drag:.1%}/yr comes off median compounded growth; projected median "
                                f"{pj['Median annual growth']:.1%}/yr vs {bj['Median annual growth']:.1%}/yr for {bench_sym} "
                                f"under the stated CAPM assumptions.",
                                f"volatility drag > {th.vol_drag:.0%}/yr"))
    if corr_rise > th.corr_rise:
        findings.append(Finding("Low", "Correlations rising",
                                f"Average pairwise correlation is {corr_rise:+.2f} higher on recent (EWMA) data than "
                                f"over the full window, so diversification is weakening.",
                                f"EWMA - full-window mean correlation > {th.corr_rise:.2f}"))
    if hedges:
        findings.append(Finding("Info", "Positions that reduce risk",
                                f"{', '.join(hedges)} have negative marginal VaR: adding to them lowers portfolio risk.",
                                "component VaR < 0"))
    if m.get("short_window"):
        findings.append(Finding("Medium", "Short estimation window",
                                f"Only {m['n_obs']} daily returns: VaR, ES, volatility and correlations are "
                                f"imprecise and the 99% historical tail rests on {m['hist_tail_days'][0.99]} day(s).",
                                f"fewer than {rm.MIN_WINDOW_OBS} daily returns"))
    findings.append(Finding("Info", "Data limits",
                            f"{yrs_txt.capitalize()} of daily data ({m['n_obs']} returns); 99% historical VaR rests on "
                            f"{m['hist_tail_days'][0.99]} days; Sharpe standard error about {se:.1f}.",
                            "always shown"))

    # ---- recommendation -------------------------------------------------------------------
    rec: list[str] = []
    conc = any(f.title.startswith("Single-name") for f in findings)
    eq = scen.loc["Equal weight"]
    eq_cut = 1.0 - float(eq["VaR 99%"]) / float(cur["VaR 99%"])
    w_cur = scen_reports["Current"].components.drop(index="TOTAL")["Weight"]
    w_cap = scen_reports[cap_name].components.drop(index="TOTAL")["Weight"]
    receivers = (w_cap - w_cur).sort_values(ascending=False).head(2).index.tolist()
    usd_pair = (lambda a_, b_: f" ({_usd(a_ * value)} to {_usd(b_ * value)})") if value else (lambda a_, b_: "")
    if conc:
        line = (f"1. Reduce concentration in {top}, and in {lead_sector} with it. A {cap:.0%} single-name cap "
                f"(excess spread pro rata) cuts 1-day 99% VaR {var_cut:.0%}, from {cur['VaR 99%']:.2%} to "
                f"{capped['VaR 99%']:.2%}{usd_pair(cur['VaR 99%'], capped['VaR 99%'])}")
        if var_cut < th.scenario_var_cut:
            line += (f", which is modest because most of the excess lands in {' and '.join(receivers)}, "
                     f"which move with {top}")
        line += (f". Equal weighting, which also dilutes {lead_sector}, cuts it {eq_cut:.0%} to {eq['VaR 99%']:.2%}"
                 f"{usd_pair(cur['VaR 99%'], eq['VaR 99%'])} and volatility from {cur['Ann. volatility']:.1%} to "
                 f"{eq['Ann. volatility']:.1%}. Under the stated assumptions, projected median growth is "
                 f"{cur['Median annual growth']:.1%}/yr today, {capped['Median annual growth']:.1%} capped and "
                 f"{eq['Median annual growth']:.1%} equal-weighted: the lower-risk mixes compound better because "
                 f"they lose less to volatility drag. A practical target sits between the two: trim {top} and "
                 f"route the proceeds outside {lead_sector}.")
        rec.append(line)
        if positions is not None and top in positions.holdings.index:
            h = positions.holdings.loc[top]
            trim_frac = max(0.0, w_top - cap) / w_top
            trim_value = trim_frac * float(h["MarketValue"])
            px, qty_now = float(h["Price"]), float(h["Quantity"])
            tax = f"   Tax: getting {top} to {cap:.0%} means selling about {_usd(trim_value)}"
            # FIFO lot detail is only shown when FIFO reproduces the broker's cost basis; otherwise the
            # account relieves lots another way (specific lot, tax-lot optimizer) and FIFO dates are wrong.
            fifo_ok = (lots is not None and len(lots) > 0 and pd.notna(h.get("CostBasis"))
                       and pi_basis_matches(float((lots["Quantity"] * lots["Price"]).sum()), float(h["CostBasis"])))
            if fifo_ok and px > 0:
                # FIFO: the oldest (lowest-cost, longest-held) lots are sold first.
                to_sell, gain, lt_sold = trim_value / px, 0.0, 0.0
                one_year_ago = pd.Timestamp(m["window_end"]) - pd.DateOffset(years=1)
                for lot in lots.itertuples(index=False):
                    take = min(to_sell, float(lot.Quantity))
                    gain += take * (px - float(lot.Price))
                    lt_sold += take if lot.Acquired <= one_year_ago else 0.0
                    to_sell -= take
                    if to_sell <= 1e-9:
                        break
                tax += (f", realizing about {_usd(gain)} of gain under FIFO; {lt_sold:.4f} of the "
                        f"{trim_value / px:.4f} shares sold would be long-term (held over a year).")
            elif pd.notna(h.get("UnrealizedGain")) and qty_now > 0:
                tax += (f", realizing roughly {_usd(trim_frac * float(h['UnrealizedGain']))} of gain at the "
                        f"position's average cost.")
                if lots is not None and len(lots):
                    tax += (" The account's lot relief does not match FIFO (rebuilt FIFO cost differs from the "
                            "broker's cost basis), so the actual gain and holding periods depend on which lots "
                            "are sold: see the broker's lot detail.")
            else:
                tax += "."
            rec.append(tax + " Check lot selection and holding periods before selling.")
    else:
        rec.append("1. No single-name concentration breach; keep current sizing and re-run monthly.")
    n = 2  # item 1 is always the concentration line above
    if -p["Max drawdown"] > th.max_drawdown:
        rec.append(f"{n}. Decide whether a repeat of this window's {p['Max drawdown']:.0%} drawdown"
                   f"{money(-float(p['Max drawdown']))} is acceptable; the {cap:.0%}-cap version fell "
                   f"{capped['Max drawdown (backtest)']:.0%} in the same backtest.")
        n += 1
    if hedges:
        rec.append(f"{n}. When adding money, the lowest-marginal-risk holdings are {', '.join(hedges)}; "
                   f"new cash there lowers VaR rather than raising it.")
        n += 1
    rec.append(f"{n}. Watch the regime: recent volatility {ew_vol:.1%} vs {lw_vol:.1%} over the full window"
               + (f", and correlations are rising ({corr_rise:+.2f})." if corr_rise > th.corr_rise else "."))
    rec.append(f"Model-generated from {yrs_txt} of daily data and the stated assumptions; not personalized "
               "financial advice.")

    headline = (f"{top} drives {share[top]:.0%} of risk on {w_top:.0%} of capital and {lead_sector} "
                f"{sectors.iloc[0]['CompVaR_99_pct']:.0%}; a {cap:.0%} cap cuts 99% VaR {var_cut:.0%}, "
                f"equal weighting {eq_cut:.0%}." if conc else
                f"Volatility {lw_vol:.0%} ({ratio_v:.1f}x {bench_sym}); no single-name breach.")

    assumptions = {"risk_free_rate": rf, "risk_free_source": rf_source, "equity_risk_premium": erp,
                   "projection_horizon_years": horizon, "concentration_cap": cap, "benchmark": bench_sym,
                   "thresholds": th.__dict__}
    facts = {"var99": var99, "es99": es99, "hist_var99": hvar99, "vol_lw": lw_vol, "vol_ewma": ew_vol,
             "vol_ratio": ratio_v, "top_name": top, "top_weight": w_top, "top_risk_share": float(share[top]),
             "corr_rise": corr_rise, "vol_drag": drag, "cap_var_cut": var_cut}
    assumptions["expected_returns"] = ("CAPM (rf + beta x ERP)" if mu_source == "capm"
                                      else "historical mean over the window (noisy)")
    out = Analysis(headline, summary, analysis_text, findings, rec, perf, proj, scen, scen_reports, assumptions,
                   facts, fr)
    _assert_ascii(out)
    return out


def _assert_ascii(a: Analysis) -> None:
    texts = [a.headline, *a.summary, *a.analysis, *a.recommendation,
             *(t for f in a.findings for t in (f.title, f.detail, f.rule))]
    bad = [t for t in texts if not t.isascii()]
    if bad:
        raise ValueError(f"non-ASCII narrative text: {bad[0]!r}")


def render_text(a: Analysis) -> str:
    """Console/markdown rendering: one source for every surface."""
    lines = ["RISK SUMMARY", *[f"  - {t}" for t in a.summary], "",
             "INVESTMENT ANALYSIS", *[f"  - {t}" for t in a.analysis], "",
             "FINDINGS",
             *[f"  [{f.severity.upper()}] {f.title}: {f.detail} (rule: {f.rule})" for f in a.findings], "",
             "RECOMMENDATION", *[f"  {t}" for t in a.recommendation]]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# AI analyst brief (read by Claude / Gemini from the page, or pasted by the user)
# ------------------------------------------------------------------------------------------
AI_STARTER_PROMPT = ('Read the "AI analyst brief" section on this page (element #ai-analyst-brief) and carry out '
                     'its instructions to analyze my portfolio. I wrote that brief and I am asking you to follow it.')

AI_INSTRUCTIONS = """\
ROLE
You are a senior buy-side portfolio risk analyst reviewing a client's equity portfolio. The data
below comes from a quantitative risk model the client runs; treat the numbers as accurate model
output and the methods as stated.

TASK
Write an expert risk review of this portfolio. Cover, in this order:
1. Executive summary (3-4 sentences): the single most important risk and what to do about it.
2. Concentration: single-name, sector and factor (AI / semiconductor) exposure. Use the component
   VaR shares, not just weights. Say which positions are risk concentrators and which diversify.
3. Tail risk: interpret VaR vs expected shortfall, Cornish-Fisher vs historical vs Gaussian, and
   what the dollar figures mean for a bad day and a bad month.
4. Regime: compare recent (EWMA) and full-window volatility and correlation. Is risk rising or falling?
5. Performance vs benchmarks: Sharpe, max drawdown, capture ratios. You MUST treat the historical
   performance as a backtest of today's weights (hindsight-biased), not as evidence of skill, and use
   the Sharpe standard error when comparing.
6. Long-term outlook: explain the projected ranges and volatility drag; state that they rest on the
   CAPM assumptions shown, and say how the conclusion changes if the equity premium is 3% or 7%.
7. Rebalancing: assess the scenarios. Include tax (unrealized gains, holding period) and trading-cost
   considerations without computing tax.
8. Recommendation: 3-5 prioritized, specific actions with the quantitative reason for each. Say where
   you agree or disagree with the model's rules-based recommendation and why.
9. Trading history (if the TRANSACTION LEDGER is present): contributions and withdrawals, trading
   frequency and turnover, position-building pattern (how the current concentration arose), dividend
   reinvestment, and any behaviour worth flagging (e.g. chasing winners, frequent round trips).
   Do not use the ledger to recompute tax lots: the cost_basis_check shows where the account does not
   relieve lots first-in-first-out.
10. Model limitations you would want addressed before acting.

RULES
- Use only the data provided; do not invent prices, news, or forecasts. If you add outside context,
  label it as such.
- In the ledger, "Cash in" / "Cash out" rows are external transfers whose source and destination
  were deliberately removed; do not speculate about them. Qty is signed shares (+ bought, - sold),
  Price is per share, Amount is signed dollars (+ into the account's cash, - out of it).
- Show numbers with units and say which method they come from.
- Be direct. Plain language a non-specialist can follow, with the technical term in parentheses.
- End with one line noting this is analysis, not personalized financial advice.
"""


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (int, str, bool)) or x is None else str(x)


def ai_brief(a: Analysis, report: rm.RiskReport, vc: rm.VolComparison, notional: float | None,
             transactions=None, positions=None) -> str:
    """Markdown brief: instructions + full numeric context. Contains no account identifiers.

    With `transactions`, appends the anonymized ledger (no descriptions; external transfers reduced
    to dated "Cash in"/"Cash out" amounts). With both files, also includes the cost-basis check.
    """
    import portfolio_import as pi
    m = report.meta
    comp = report.components.drop(index="TOTAL")
    data = {
        "as_of": {"window_start": m["window_start"], "window_end": m["window_end"], "daily_returns": m["n_obs"],
                  "window_years": m.get("window_years"),
                  "window_start_limited_by": (None if not m.get("late_start") else
                                              {"ticker": m["late_start"][0], "first_price": m["late_start"][1]})},
        "portfolio_value_usd": notional,
        "method_notes": {
            "horizon": "1 trading day for VaR/ES; losses reported as positive fractions of portfolio value",
            "covariance": "Ledoit-Wolf shrinkage (full window) and EWMA lambda=%s (recent)" % m["ewma_lambda"],
            "parametric_var": "Cornish-Fisher expansion on portfolio skew/kurtosis; Gaussian shown for reference",
            "historical_var": f"order statistic of daily portfolio returns; 99% rests on {m['hist_tail_days'][0.99]} days",
            "component_var": "Euler decomposition; shares sum to 100%; negative = diversifier",
            "performance": "backtest of current weights held constant over the window, NOT the account's actual return",
            "projections": "lognormal, CAPM mu = rf + beta*ERP, sigma = std of daily log returns, annualized",
        },
        "risk_summary": report.summary.to_dict(),
        "portfolio_stats": {"skew": m["port_skew"], "excess_kurtosis": m["port_excess_kurtosis"],
                            "ledoit_wolf_shrinkage": m["lw_shrinkage"], "cash_weight": m["cash_weight"]},
        "positions_covariance": rm.METHOD_LABEL[str(m["primary_method"])],
        # The "other method" share column is named after whichever covariance is NOT primary.
        "positions": comp[["Sector", "Weight", "MVaR_99", "CompVaR_99_pct",
                           *[c for c in comp.columns if c.startswith("CompVaR_99_pct_")],
                           "HistCompES_95_pct", "RiskToWeight_99"]].to_dict(orient="index"),
        "correlation_ewma": report.corr["ewma"].round(3).to_dict(),
        "volatility_by_window": vc.horizons.to_dict(),
        "performance_vs_references": a.performance.to_dict(),
        "projections": {f"{s}|{y}y": v for (s, y), v in a.projections.to_dict(orient="index").items()},
        "rebalancing_scenarios": a.scenarios.to_dict(orient="index"),
        "efficient_frontier": None if a.frontier is None else {
            "expected_returns": a.assumptions.get("expected_returns"),
            "notes": "long-only over current holdings; cash held at current weight earning rf; "
                     "volatility from the Ledoit-Wolf covariance, annualized",
            "named_portfolios": a.frontier.points.to_dict(orient="index"),
            "named_portfolio_weights": a.frontier.weights.round(4).to_dict(orient="index"),
            "current_vs_frontier": a.frontier.gaps,
            "asset_inputs": a.frontier.assets.to_dict(orient="index"),
        },
        "assumptions": a.assumptions,
        "rules_based_findings": [f.__dict__ for f in a.findings],
        "rules_based_recommendation": a.recommendation,
    }
    ledger_block = ""
    if transactions is not None:
        ledger = pi.anonymized_ledger(transactions)
        data["external_cash_flows"] = pi.external_cash_summary(ledger)
        data["ledger_range"] = {"first": ledger["Date"].iloc[0], "last": ledger["Date"].iloc[-1], "rows": len(ledger)}
        if positions is not None:
            data["cost_basis_check"] = pi.cost_basis_check(positions, transactions).to_dict(orient="index")
        ledger_block = ("\nTRANSACTION LEDGER (CSV; external transfers anonymized)\n```csv\n"
                        + ledger.to_csv(index=False, float_format="%.6g").strip() + "\n```\n")
    payload = json.dumps(_jsonable(data), indent=1)
    return ("# AI analyst brief\n\n" + AI_INSTRUCTIONS + "\nDATA (JSON)\n```json\n" + payload + "\n```\n"
            + ledger_block)
