"""Verification suite for risk_model.py. Each check uses an oracle on a different code path.

Run with `py -m pytest test_risk_model.py` or `py test_risk_model.py` (exits non-zero on failure).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import integrate, stats
from sklearn.covariance import LedoitWolf

import risk_model as rm

RNG = np.random.default_rng(7)


def _synthetic_prices(n: int = 260, k: int = 5) -> pd.DataFrame:
    a = RNG.normal(size=(k, k)) * 0.01
    shocks = stats.t.rvs(df=4, size=(n, k), random_state=RNG) @ a.T * 0.6
    idx = pd.bdate_range("2025-01-02", periods=n + 1)
    lp = np.vstack([np.zeros(k), np.cumsum(shocks, axis=0)]) + np.log(100.0)
    return pd.DataFrame(np.exp(lp), index=idx, columns=[f"T{i}" for i in range(k)])


def test_cf_tail_mean_matches_numerical_integral() -> None:
    # Oracle: integrate the CF quantile function over u in (0, alpha) directly.
    for s, k in [(-0.6, 2.5), (0.3, 1.0), (-0.2, 4.0)]:
        for conf in (0.95, 0.99):
            alpha = 1 - conf
            z = stats.norm.ppf(alpha)
            f = lambda u: rm.cornish_fisher_quantile(stats.norm.ppf(u), s, k)  # noqa: E731
            num, _ = integrate.quad(f, 0.0, alpha, limit=200, epsabs=1e-13)
            assert np.isclose(rm.cornish_fisher_tail_mean(z, s, k), num / alpha, rtol=1e-7), (s, k, conf)


def test_cf_quantile_matches_textbook_formula() -> None:
    for s, k in [(-0.6, 2.5), (0.4, 0.7)]:
        for z in (-1.645, -2.326):
            ref = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
            assert np.isclose(rm.cornish_fisher_quantile(z, s, k), ref, rtol=1e-12)


def test_cf_reduces_to_gaussian() -> None:
    sigma, mu = 0.02, 0.0005
    for conf in (0.95, 0.99):
        z = stats.norm.ppf(1 - conf)
        r = rm.parametric_var_es(sigma, mu, conf)
        assert np.isclose(r.var, -(mu + sigma * z), rtol=1e-12)
        assert np.isclose(r.es, -mu + sigma * stats.norm.pdf(z) / (1 - conf), rtol=1e-12)


def test_ewma_matches_recursion() -> None:
    x = pd.DataFrame(RNG.normal(scale=0.02, size=(150, 4)))
    lam = 0.94
    s = np.zeros((4, 4))
    for row in x.to_numpy():
        s = lam * s + (1 - lam) * np.outer(row, row)
    s /= 1 - lam ** len(x)
    assert np.allclose(rm.ewma_cov(x, lam).to_numpy(), s, rtol=1e-12, atol=0)


def test_ewma_vol_path_ends_at_portfolio_ewma_vol() -> None:
    x = pd.DataFrame(RNG.normal(scale=0.02, size=(200, 4)))
    w = np.array([0.4, 0.3, 0.2, 0.05])
    path = rm.ewma_vol_series(x @ w, 0.94)
    expected = np.sqrt(w @ rm.ewma_cov(x, 0.94).to_numpy() @ w)
    assert np.isclose(path.iloc[-1], expected, rtol=1e-12)
    assert np.isclose(path.iloc[0], abs(float((x @ w).iloc[0])), rtol=1e-12)  # t=0: normalised to r0^2


def test_volatility_comparison_horizons() -> None:
    idx = pd.bdate_range("2025-01-02", periods=301)
    ref = pd.DataFrame({"SPY": 100 * np.exp(np.cumsum(RNG.normal(0, 0.01, 301))),
                        "AOK": 50 * np.exp(np.cumsum(RNG.normal(0, 0.004, 301)))}, index=idx)
    port = pd.Series(RNG.normal(0, 0.03, 300), index=idx[1:])
    path = rm.ewma_vol_series(np.log1p(port), 0.94)
    vc = rm.volatility_comparison(port, path, ref, {"Benchmark": "SPY", "Conservative": "AOK"})
    spy = ref["SPY"].iloc[-22:].to_numpy()
    oracle = np.std(spy[1:] / spy[:-1] - 1, ddof=1) * np.sqrt(252)   # 21 simple returns from raw prices
    assert np.isclose(vc.horizons.loc["1M", "Benchmark (SPY)"], oracle, rtol=1e-12)
    assert np.isclose(vc.horizons.loc["12M", "Portfolio"], port.iloc[-252:].std(ddof=1) * np.sqrt(252))
    assert np.isclose(vc.vs_benchmark["3M"], vc.horizons.loc["3M", "Portfolio"] / vc.horizons.loc["3M", "Benchmark (SPY)"])
    assert np.allclose(vc.paths["Portfolio"], path * np.sqrt(252))  # passed through, not recomputed
    assert list(vc.horizons.index) == ["1M", "3M", "6M", "12M"]
    # A 250-session year (what one calendar year of data holds) must still fill the 12M row.
    short = rm.volatility_comparison(port.iloc[-250:], path.iloc[-250:], ref, {"Benchmark": "SPY"})
    assert np.isclose(short.horizons.loc["12M", "Portfolio"], port.iloc[-250:].std(ddof=1) * np.sqrt(252))
    tiny = rm.volatility_comparison(port.iloc[-200:], path.iloc[-200:], ref, {"Benchmark": "SPY"})
    assert tiny.horizons.loc["12M"].isna().all()


def test_ledoit_wolf_matches_formula() -> None:
    # Oracle: Ledoit-Wolf (2004) shrinkage written out from the paper, not via sklearn.
    x = RNG.normal(size=(120, 6)) @ RNG.normal(size=(6, 6)) * 0.01
    n, p = x.shape
    xc = x - x.mean(0)
    s = xc.T @ xc / n
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2)
    b2 = min(d2, sum(np.sum((np.outer(r, r) - s) ** 2) for r in xc) / n**2)
    delta = b2 / d2
    target = (1 - delta) * s + delta * mu * np.eye(p)
    cov, shrink = rm.ledoit_wolf_cov(pd.DataFrame(x))
    assert np.isclose(shrink, delta, rtol=1e-8)
    assert np.allclose(cov.to_numpy(), target, rtol=1e-8)


def test_historical_var_is_order_statistic() -> None:
    # Hard-coded k per (n, conf) so float rounding in 1 - conf cannot leak into the oracle.
    for n, conf, k in [(251, 0.95, 13), (251, 0.99, 3), (260, 0.95, 13), (300, 0.99, 3), (200, 0.99, 2)]:
        r = RNG.normal(size=n) * 0.02
        srt = np.sort(r)
        h = rm.historical_var_es(r, conf)
        assert np.isclose(h.var, -srt[k - 1])
        assert np.isclose(h.es, -srt[:k].mean())
        assert h.n_tail == k


def test_euler_components_sum_and_match_finite_difference() -> None:
    k = 5
    a = RNG.normal(size=(k, k)) * 0.01
    cov = a @ a.T
    w = RNG.uniform(0.05, 0.3, size=k)
    mu = RNG.normal(scale=1e-4, size=k)
    q = rm.cornish_fisher_quantile(stats.norm.ppf(0.01), -0.5, 3.0)
    var = lambda ww: -(ww @ mu + q * np.sqrt(ww @ cov @ ww))  # noqa: E731
    marg, comp = rm.euler_var_components(w, cov, mu, q)
    assert np.isclose(comp.sum(), var(w), rtol=1e-12)
    h = 1e-7
    fd = np.array([(var(w + h * e) - var(w - h * e)) / (2 * h) for e in np.eye(k)])
    assert np.allclose(marg, fd, rtol=1e-5)


def test_end_to_end_on_synthetic_prices() -> None:
    prices = _synthetic_prices()
    prices.iloc[50:53, 2] = np.nan  # interior gap -> forward filled
    tickers = list(prices.columns)
    w = [0.4, 0.25, 0.15, 0.1, 0.05]  # sums to 0.95 -> 5% cash
    cfg = rm.RiskModelConfig(tickers=tickers, weights=w)
    rep = rm.PortfolioRiskModel(cfg, prices=prices).run()

    assert rep.meta["ffilled_cells"] == 3
    assert np.isclose(rep.meta["cash_weight"], 0.05)
    comp = rep.components
    assert np.isclose(comp.loc["TOTAL", "CompVaR_99_pct"], 1.0)
    assert np.isclose(comp.loc["TOTAL", "HistCompES_95_pct"], 1.0)
    assert np.isclose(comp.loc["TOTAL", "CompVaR_99"], rep.summary.loc["VaR 99% (Cornish-Fisher)", "LedoitWolf"])
    # Independent recomputation of historical VaR from raw prices.
    lr = np.log(prices.ffill()).diff().dropna()
    port = np.expm1(lr).to_numpy() @ np.array(w)
    assert len(port) == 260
    assert np.isclose(rep.summary.loc["VaR 95% (Historical)", "Historical"], -np.sort(port)[12])  # 13th worst
    for m in ("ledoit_wolf", "ewma"):
        c = rep.corr[m].to_numpy()
        assert np.allclose(np.diag(c), 1.0) and np.all(np.linalg.eigvalsh(c) > 0)
    # ES must dominate VaR at every level and source.
    s = rep.summary
    for src, col in [("Cornish-Fisher", "LedoitWolf"), ("Cornish-Fisher", "EWMA"), ("Historical", "Historical")]:
        for p in (95, 99):
            assert s.loc[f"ES {p}% ({src})", col] >= s.loc[f"VaR {p}% ({src})", col]


def test_cf_validity_matches_numerical_monotonicity() -> None:
    # Oracle: finite-difference slope of the CF quantile over a wide z grid.
    z = np.linspace(-8, 8, 4001)
    grid = [(0.0, 0.0), (-0.04, 0.58), (-0.6, 2.5), (-1.5, 0.3), (0.0, 12.0), (1.0, 1.0),
            (0.5, 5.0), (-2.0, 6.0), (0.3, -0.5), (0.0, 0.5), (-1.0, 4.0)]
    verdicts = set()
    for s, k in grid:
        q = np.array([rm.cornish_fisher_quantile(v, s, k) for v in z])
        monotone = bool(np.all(np.diff(q) > 0))
        verdicts.add(monotone)
        assert rm.cornish_fisher_is_valid(s, k) == monotone, (s, k)
    assert verdicts == {True, False}  # grid must exercise both outcomes


def _fake_yf(responses):
    """yfinance stand-in; `responses(symbols, call_no)` returns the Close frame for each call."""
    import types
    fake = types.ModuleType("yfinance")
    fake.set_tz_cache_location = lambda _p: None
    calls = {"n": 0}

    def download(symbols, **_kw):
        calls["n"] += 1
        close = responses(list(symbols), calls["n"])
        close.columns = pd.MultiIndex.from_product([["Close"], close.columns])
        return close

    fake.download = download
    return fake


def _with_fake_yf(fake, fn):
    real = sys.modules.get("yfinance")
    orig_sleep = rm.time.sleep
    sys.modules["yfinance"] = fake
    rm.time.sleep = lambda _s: None
    try:
        return fn()
    finally:
        rm.time.sleep = orig_sleep
        if real is not None:
            sys.modules["yfinance"] = real
        else:
            del sys.modules["yfinance"]


def test_retry_recovers_ticker_missing_on_batch_call() -> None:
    idx = pd.bdate_range("2025-01-02", periods=6)

    def responses(symbols, call_no):
        vals = {s: np.arange(1.0, 7.0) * (i + 1) for i, s in enumerate(symbols)}
        if call_no == 1 and "PWR" in symbols:  # batch call: PWR fails like the sqlite lock
            vals["PWR"] = np.full(6, np.nan)
        return pd.DataFrame(vals, index=idx)

    out = _with_fake_yf(_fake_yf(responses), lambda: rm.fetch_prices(["MU", "PWR"], idx[0], idx[-1]))
    assert list(out.columns) == ["MU", "PWR"]
    assert len(out) == 6 and not out.isna().any().any()
    assert np.allclose(out["PWR"], np.arange(1.0, 7.0))  # single-ticker retry value, not batch


def test_color_heatmap_is_ascii_and_matches_plain() -> None:
    import re
    tickers = ["MU", "AMD", "VRT", "MCD", "WMT"]
    a = RNG.uniform(-1, 1, size=(5, 5))
    c = np.corrcoef(a @ a.T + np.eye(5))
    corr = pd.DataFrame(c, index=tickers, columns=tickers)
    order = rm.sector_order(tickers, [0.4, 0.2, 0.2, 0.1, 0.1])
    colored = rm.render_terminal_heatmap(corr, order, color=True)
    plain = rm.render_terminal_heatmap(corr, order, color=False)
    assert "\x1b[" in colored
    assert colored.count("\x1b[0m") == colored.count("\x1b[48;2;") == len(tickers) ** 2  # every cell reset
    assert colored.isascii() and plain.isascii()
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain


def test_window_bounds_and_fetch_use_explicit_start() -> None:
    end = pd.Timestamp("2026-10-02")
    assert rm.window_bounds(rm.RiskModelConfig(end=end)) == (pd.Timestamp("2025-10-02"), end)
    assert rm.window_bounds(rm.RiskModelConfig(end=end, lookback_years=2))[0] == pd.Timestamp("2024-10-02")
    assert rm.window_bounds(rm.RiskModelConfig(end=end, start=pd.Timestamp("2025-07-11"), lookback_years=3))[0] == \
        pd.Timestamp("2025-07-11")                                    # explicit start overrides the lookback
    try:
        rm.window_bounds(rm.RiskModelConfig(end=end, start=end))
    except ValueError:
        pass
    else:
        raise AssertionError("start == end accepted")
    # The model must fetch exactly the window it reports.
    seen = {}
    prices = _synthetic_prices(n=300)
    prices.index = pd.bdate_range("2025-07-11", periods=len(prices))

    def fake_fetch(tickers, start, end_):
        seen["window"] = (start, end_)
        return prices
    orig = rm.fetch_prices
    rm.fetch_prices = fake_fetch
    try:
        cfg = rm.RiskModelConfig(tickers=list(prices.columns), weights=[0.2] * 5, end=end,
                                 start=pd.Timestamp("2025-07-11"))
        rep = rm.PortfolioRiskModel(cfg).run()
    finally:
        rm.fetch_prices = orig
    assert seen["window"] == (pd.Timestamp("2025-07-11"), end)
    assert rep.meta["late_start"] is None
    assert np.isclose(rep.meta["window_years"], 300 / 252)


def test_late_start_and_short_window_flags() -> None:
    prices = _synthetic_prices(n=260)
    prices.index = pd.bdate_range("2025-01-02", periods=len(prices))
    prices.iloc[:120, 3] = np.nan                                     # T3 "lists" 120 days into the window
    cfg = rm.RiskModelConfig(tickers=list(prices.columns), weights=[0.2] * 5, end=prices.index[-1] + pd.Timedelta(days=1),
                             start=pd.Timestamp("2025-01-02"))
    rep = rm.PortfolioRiskModel(cfg, prices=prices).run()
    assert rep.meta["late_start"] == ("T3", prices.index[120].date())
    assert rep.meta["window_start"] > prices.index[120].date()        # the common window really starts later
    assert not rep.meta["short_window"]
    short = rm.PortfolioRiskModel(cfg, prices=prices.iloc[-61:]).run()
    assert short.meta["n_obs"] == 60 and short.meta["short_window"]


def test_dead_ticker_raises() -> None:
    import types
    fake = types.ModuleType("yfinance")
    fake.set_tz_cache_location = lambda _p: None
    idx = pd.bdate_range("2025-01-02", periods=5)

    def download(symbols, **_kw):
        cols = pd.MultiIndex.from_product([["Close"], symbols])
        data = np.where(np.array(symbols) == "DEAD", np.nan, 1.0)
        return pd.DataFrame(np.tile(data, (5, 1)), index=idx, columns=cols)

    fake.download = download
    real = sys.modules.get("yfinance")
    sys.modules["yfinance"] = fake
    try:
        rm.time.sleep, orig = (lambda _s: None), rm.time.sleep
        try:
            rm.fetch_prices(["OK", "DEAD"], idx[0], idx[-1], retries=2)
        except RuntimeError as e:
            assert "DEAD" in str(e)
        else:
            raise AssertionError("dead ticker did not raise")
    finally:
        rm.time.sleep = orig
        if real is not None:
            sys.modules["yfinance"] = real
        else:
            del sys.modules["yfinance"]


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
