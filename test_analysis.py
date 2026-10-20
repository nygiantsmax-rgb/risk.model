"""Tests for analysis.py. Oracles are hand calculations, Monte Carlo, or a separate code path.

Run with `py -m pytest test_analysis.py` or `py test_analysis.py` (non-zero exit on failure).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

import analysis as an
import portfolio_import as pi
import risk_model as rm

RNG = np.random.default_rng(11)


def _prices(n: int, vols: dict[str, float], start: str = "2025-01-02", common: float = 0.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n + 1)
    f = RNG.normal(0, 0.01, n)
    cols = {}
    for k, v in vols.items():
        r = common * f * v / 0.01 + RNG.normal(0, v, n) + 0.0004
        cols[k] = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.DataFrame(cols, index=idx)


def test_wealth_drawdown_counts_starting_value_as_peak() -> None:
    w, dd = an.wealth_drawdown(pd.Series([-0.2, 0.1, 0.5, -0.5]))
    assert np.isclose(dd.iloc[0], -0.2)                       # first-day loss is a drawdown from 1.0
    assert np.isclose(dd.iloc[1], 0.8 * 1.1 - 1)
    assert np.isclose(dd.min(), -0.5)                          # 1.32 -> 0.66
    assert np.isclose(w.iloc[-1], 0.8 * 1.1 * 1.5 * 0.5)


def test_performance_metrics_against_hand_calculation() -> None:
    b = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    p = pd.Series([0.02, -0.03, 0.05, -0.02, 0.01])  # up capture 1.33 != down capture 1.67
    perf = an.performance_metrics(pd.DataFrame({"P": p, "B": b}), rf=0.0252, benchmark="B")["P"]
    ex = p.to_numpy() - 0.0001
    vol = np.std(p.to_numpy(), ddof=1) * np.sqrt(252)
    assert np.isclose(perf["Sharpe ratio"], ex.mean() * 252 / vol)
    assert np.isclose(perf["Upside capture"], np.mean([0.02, 0.05, 0.01]) / np.mean([0.01, 0.03, 0.02]))
    assert np.isclose(perf["Downside capture"], np.mean([-0.03, -0.02]) / np.mean([-0.02, -0.01]))
    assert not np.isclose(perf["Upside capture"], perf["Downside capture"])  # data must separate them
    assert np.isclose(perf["Beta vs benchmark"], np.cov(p, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    assert np.isclose(perf["Total return"], np.prod(1 + p.to_numpy()) - 1)
    assert np.isclose(perf["Sharpe std. error"], np.sqrt((1 + 0.5 * perf["Sharpe ratio"] ** 2) / (5 / 252)))


def test_lognormal_outcomes_match_monte_carlo() -> None:
    mu, sigma, t = 0.12, 0.45, 5
    o = an.lognormal_outcomes(mu, sigma, t)
    rng = np.random.default_rng(3)
    # Independent path: simulate daily log returns with the same annual drift and vol.
    daily = rng.normal((np.log1p(mu) - 0.5 * sigma ** 2) / 252, sigma / np.sqrt(252), size=(40_000, 252 * t)).sum(1)
    w = np.exp(daily)
    for p in an.PERCENTILES:
        assert np.isclose(o[f"P{int(round(p * 100))}"], np.quantile(w, p), rtol=0.03), p
    assert np.isclose(o["Prob. of loss"], (w < 1).mean(), atol=0.01)
    assert np.isclose(o["Expected"], w.mean(), rtol=0.05)
    assert np.isclose(o["Expected"], (1 + mu) ** t)


def test_cap_weights() -> None:
    w = pd.Series({"A": 0.40, "B": 0.25, "C": 0.15, "D": 0.10, "E": 0.05})  # sums to 0.95
    c = an.cap_weights(w, 0.20)
    assert np.isclose(c.sum(), w.sum())
    assert (c <= 0.20 + 1e-12).all()
    free = c[c < 0.20 - 1e-12].index
    assert np.allclose(c[free] / c[free].sum(), w[free] / w[free].sum())  # uncapped keep proportions
    try:
        an.cap_weights(w, 0.15)
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible cap accepted")


def _setup():
    tick = {"AAA": 0.03, "BBB": 0.025, "CCC": 0.02, "DDD": 0.01, "EEE": 0.012}
    prices = _prices(270, tick, common=0.8)
    refs = _prices(270, {"SPY": 0.01, "AOA": 0.008, "AOK": 0.004}, common=0.9)
    cfg = rm.RiskModelConfig(tickers=list(tick), weights=[0.45, 0.2, 0.15, 0.1, 0.05], notional=10_000.0)  # 5% cash
    rep = rm.PortfolioRiskModel(cfg, prices=prices).run()
    vc = rm.volatility_comparison(rep.port_returns, rep.ewma_vol, refs, rm.DEFAULT_REFERENCES)
    a = an.build_analysis(rep, cfg, prices, vc, 0.04, "test", notional=10_000.0)
    return cfg, prices, rep, vc, a


def test_scenarios_reuse_the_model() -> None:
    cfg, prices, rep, vc, a = _setup()
    cur = a.scenario_reports["Current"]
    pd.testing.assert_frame_equal(cur.summary, rep.summary)   # "Current" is the main report, re-derived
    assert np.isclose(a.scenarios.loc["Cap 20% per name", "Max weight"], 0.20)
    for name, r in a.scenario_reports.items():  # every scenario keeps the same invested total (cash unchanged)
        assert np.isclose(r.components.loc["TOTAL", "Weight"], 0.95), name
    assert np.isclose(a.scenarios.loc["Current", "VaR 99%"], rep.summary.loc["VaR 99% (Cornish-Fisher)", "LedoitWolf"])


def test_single_source_for_historical_volatility() -> None:
    _, _, rep, vc, a = _setup()
    assert np.isclose(a.performance.loc["Annualized volatility", "Portfolio"],
                      rep.summary.loc["Ann. Volatility", "Historical"], rtol=1e-12)


def test_analysis_text_findings_and_brief() -> None:
    _, _, rep, vc, a = _setup()
    assert any(f.severity == "High" and "AAA" in f.title for f in a.findings)   # 45% weight breaches 20%
    assert a.recommendation[0].startswith("1. Reduce concentration in AAA")
    assert "not personalized financial advice" in a.recommendation[-1]
    assert an.render_text(a).isascii()
    brief = an.ai_brief(a, rep, vc, 10_000.0)
    payload = brief.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    data = json.loads(payload)                                  # strict JSON: no NaN tokens
    assert data["rules_based_recommendation"] == a.recommendation
    assert "backtest" in data["method_notes"]["performance"]


def test_tax_line_uses_fifo_only_when_it_matches_broker_basis() -> None:
    import types
    _, prices, rep, vc, _ = _setup()
    cfg = rm.RiskModelConfig(tickers=list(prices.columns), weights=[0.45, 0.2, 0.15, 0.1, 0.05], notional=10_000.0)
    lots = pd.DataFrame({"Acquired": [pd.Timestamp("2024-01-02"), pd.Timestamp("2026-01-02")],
                         "Quantity": [10.0, 30.0], "Price": [50.0, 100.0]})        # FIFO cost $3,500
    hold = pd.DataFrame({"MarketValue": [4500.0], "Price": [112.5], "Quantity": [40.0],
                         "CostBasis": [3500.0], "UnrealizedGain": [1000.0]}, index=["AAA"])
    match = types.SimpleNamespace(holdings=hold)
    a = an.build_analysis(rep, cfg, prices, vc, 0.04, "test", notional=10_000.0, positions=match, lots=lots)
    tax = next(t for t in a.recommendation if t.lstrip().startswith("Tax:"))
    assert "under FIFO" in tax and "does not match FIFO" not in tax
    # Broker reports a different basis (e.g. specific-lot relief): FIFO detail must not be presented.
    differ = types.SimpleNamespace(holdings=hold.assign(CostBasis=[3000.0], UnrealizedGain=[1500.0]))
    b = an.build_analysis(rep, cfg, prices, vc, 0.04, "test", notional=10_000.0, positions=differ, lots=lots)
    tax_b = next(t for t in b.recommendation if t.lstrip().startswith("Tax:"))
    assert "does not match FIFO" in tax_b and "under FIFO" not in tax_b and "average cost" in tax_b


def test_brief_includes_ledger_without_private_text() -> None:
    import test_portfolio_import as tpi
    _, _, rep, vc, a = _setup()
    tx = pi.parse_schwab_transactions(tpi.CASH_TX.encode())
    brief = an.ai_brief(a, rep, vc, 10_000.0, transactions=tx)
    assert "TRANSACTION LEDGER" in brief and "Cash out" in brief and "Cash in" in brief
    assert not [w for w in tpi.PRIVATE if w in brief]
    payload = json.loads(brief.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert payload["external_cash_flows"]["net"] == 250.0


def test_analysis_and_brief_with_ewma_primary() -> None:
    # Regression: the brief hard-coded the EWMA "other method" column and crashed when EWMA was primary.
    from dataclasses import replace
    cfg, prices, _, _, _ = _setup()
    for method, other in (("ewma", "CompVaR_99_pct_LedoitWolf"), ("ledoit_wolf", "CompVaR_99_pct_EWMA")):
        c = replace(cfg, primary_method=method)
        rep = rm.PortfolioRiskModel(c, prices=prices).run()
        refs = _prices(270, {"SPY": 0.01, "AOA": 0.008, "AOK": 0.004}, common=0.9)
        vc = rm.volatility_comparison(rep.port_returns, rep.ewma_vol, refs, rm.DEFAULT_REFERENCES)
        a = an.build_analysis(rep, c, prices, vc, 0.04, "test", notional=10_000.0)
        data = json.loads(an.ai_brief(a, rep, vc, 10_000.0).split("```json\n", 1)[1].split("\n```", 1)[0])
        first = next(iter(data["positions"].values()))
        assert other in first, (method, list(first))
        assert data["positions_covariance"] == rm.METHOD_LABEL[method]


def test_min_variance_matches_closed_form() -> None:
    # Oracle: unconstrained minimum variance w = c * inv(S) 1, valid when every weight is positive.
    cov = np.array([[0.04, 0.006, 0.004], [0.006, 0.09, 0.01], [0.004, 0.01, 0.0625]])
    mu = np.array([0.06, 0.09, 0.08])
    inv1 = np.linalg.solve(cov, np.ones(3))
    closed = 0.95 * inv1 / inv1.sum()
    assert (closed > 0).all()
    _, w_min, _ = an.frontier_curve(mu, cov, 0.95, 0.03, upper=0.95)
    assert np.allclose(w_min, closed, atol=1e-6)


def test_frontier_invariants() -> None:
    cfg, prices, rep, vc, a = _setup()
    fr = a.frontier
    tickers = list(cfg.tickers)
    for name, curve in fr.curves.items():
        w = curve[tickers].to_numpy()
        assert np.allclose(w.sum(1), 0.95, atol=1e-7), name                   # cash weight held fixed
        upper = 0.95 if name == "Long-only" else 0.20
        assert (w >= -1e-9).all() and (w <= upper + 1e-7).all(), name
        assert (np.diff(curve["Expected return"]) > 0).all(), name
        assert (np.diff(curve["Volatility"]) > -1e-9).all(), name             # efficient branch only
        assert fr.points.loc["Max Sharpe" if name == "Long-only" else "Max Sharpe (max 20%)", "Sharpe"] >= \
            curve["Sharpe"].max() - 1e-6, name
    # Nothing feasible beats the frontier: the current mix sits on or below it.
    assert fr.gaps["return_gap_same_vol"] >= -1e-6 and fr.gaps["vol_saving_same_return"] >= -1e-6
    assert fr.points.loc["Max Sharpe", "Sharpe"] >= fr.points["Sharpe"].drop("Max Sharpe (max 20%)").max() - 1e-6


def test_frontier_single_source_with_model_and_projection() -> None:
    _, _, rep, vc, a = _setup()
    cur = a.frontier.points.loc["Current"]
    assert np.isclose(cur["Volatility"], rep.summary.loc["Ann. Volatility", "LedoitWolf"], rtol=1e-10)
    assert np.isclose(cur["Expected return"], a.projections.loc[("Portfolio", 5), "Expected return (CAPM)"], rtol=1e-10)
    cap_pt = a.frontier.points.loc["Cap 20% per name"]
    assert np.isclose(cap_pt["Volatility"], a.scenarios.loc["Cap 20% per name", "Ann. volatility"], rtol=1e-10)


def test_narrative_reports_actual_window_length() -> None:
    _, _, rep, vc, a = _setup()                                       # 270 returns = 1.1 years
    text = " ".join(a.analysis + a.recommendation + [f.detail for f in a.findings])
    assert "1.1 years of data" in text and "one year" not in text.lower().replace("over a year", "")
    assert not any(f.title == "Short estimation window" for f in a.findings)
    cfg, prices, _, _, _ = _setup()
    short_px = prices.iloc[-81:]
    rep_s = rm.PortfolioRiskModel(cfg, prices=short_px).run()
    refs = _prices(270, {"SPY": 0.01, "AOA": 0.008, "AOK": 0.004}, common=0.9).reindex(short_px.index)
    vc_s = rm.volatility_comparison(rep_s.port_returns, rep_s.ewma_vol, refs, rm.DEFAULT_REFERENCES)
    a_s = an.build_analysis(rep_s, cfg, short_px, vc_s, 0.04, "test", notional=10_000.0)
    assert any(f.title == "Short estimation window" and f.severity == "Medium" for f in a_s.findings)


def test_no_fixed_one_year_wording_on_a_long_window() -> None:
    import re
    tick = {"AAA": 0.03, "BBB": 0.025, "CCC": 0.02, "DDD": 0.01, "EEE": 0.012}
    prices = _prices(630, tick, common=0.8)                           # 2.5 years
    refs = _prices(630, {"SPY": 0.01, "AOA": 0.008, "AOK": 0.004}, common=0.9)
    cfg = rm.RiskModelConfig(tickers=list(tick), weights=[0.45, 0.2, 0.15, 0.1, 0.05], notional=10_000.0)
    rep = rm.PortfolioRiskModel(cfg, prices=prices).run()
    vc = rm.volatility_comparison(rep.port_returns, rep.ewma_vol, refs, rm.DEFAULT_REFERENCES)
    fire_all = an.Thresholds(single_weight=0.0, single_risk_share=0.0, sector_risk_share=0.0, vol_ratio=0.0,
                             max_drawdown=0.0, vol_drag=0.0, corr_rise=-1.0)   # every wording path runs
    a = an.build_analysis(rep, cfg, prices, vc, 0.04, "test", notional=10_000.0, th=fire_all)
    texts = [a.headline, *a.summary, *a.analysis, *a.recommendation,
             *(t for f in a.findings for t in (f.title, f.detail, f.rule)), an.AI_INSTRUCTIONS]
    fixed_period = re.compile(r"\b(the|full|past|one) year\b|full-year|year's|1-year", re.IGNORECASE)
    hits = [t for t in texts if fixed_period.search(t)]
    assert not hits, hits[:2]
    assert any("2.5 years" in t for t in texts)
    assert {"Correlations rising", "Deep drawdown in the backtest"} <= {f.title for f in a.findings}


def test_small_portfolio_below_cap_feasibility() -> None:
    # Regression: 3 names cannot each be capped at 20%; analysis used to raise and lose every section.
    tick = {"AAA": 0.03, "BBB": 0.02, "CCC": 0.01}
    prices = _prices(270, tick, common=0.8)
    refs = _prices(270, {"SPY": 0.01, "AOA": 0.008, "AOK": 0.004}, common=0.9)
    cfg = rm.RiskModelConfig(tickers=list(tick), weights=[0.4, 0.35, 0.25])
    rep = rm.PortfolioRiskModel(cfg, prices=prices).run()
    vc = rm.volatility_comparison(rep.port_returns, rep.ewma_vol, refs, rm.DEFAULT_REFERENCES)
    a = an.build_analysis(rep, cfg, prices, vc, 0.04, "test", cap=0.20)
    assert "Cap 33% per name" in a.scenarios.index                   # effective cap = equal weight
    assert np.allclose(a.scenario_reports["Cap 33% per name"].components.drop(index="TOTAL")["Weight"], 1 / 3)
    assert list(a.frontier.curves) == ["Long-only"]                   # no degenerate capped curve
    assert an.render_text(a).isascii()


def test_fifo_lots() -> None:
    tx = pi.parse_schwab_transactions(b'''"Date","Action","Symbol","Description","Quantity","Price","Fees & Comm","Amount"
"03/01/2026","Sell","AAA","A","1.5","$30.00","",""
"02/01/2026","Journal","AAA","A","-2","","",""
"02/01/2026","Journal","AAA","A","2","","",""
"01/15/2026","Buy","AAA","A","2","$20.00","",""
"01/01/2025","Buy","AAA","A","1","$10.00","",""
''')
    lots, unmatched = pi.fifo_lots(tx, "AAA")
    assert unmatched == 0.0
    assert len(lots) == 1 and np.isclose(lots.iloc[0]["Quantity"], 1.5) and lots.iloc[0]["Price"] == 20.0
    assert np.isclose(lots["Quantity"].sum(), tx.holdings_as_of()["AAA"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
