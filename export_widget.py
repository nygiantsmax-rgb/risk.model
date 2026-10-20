"""Build wix_risk_widget.html: a self-contained, client-side risk widget for a Wix "Embed HTML" element.

Demo portfolio: SPY's largest holdings at their index weights, rescaled to 100%, benchmarked to SPY.
Every number is computed here by the tested model and embedded as data; the page's JavaScript only
re-weights it, and checks itself against the model's baseline numbers on load.

Usage:  py export_widget.py [--top 10] [--years 1] [--out wix_risk_widget.html]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analysis as an
import risk_model as rm

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "/*__WIDGET_DATA__*/null"
BENCH = "SPY"

# Display groups for the demo (GICS-style); anything unlisted falls under "Other".
GROUPS = {
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "MU": "Semiconductors", "AMD": "Semiconductors",
    "TSM": "Semiconductors",
    "AAPL": "Software & hardware", "MSFT": "Software & hardware", "ORCL": "Software & hardware",
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication", "NFLX": "Communication",
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer", "COST": "Consumer",
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "LLY": "Health care", "UNH": "Health care", "JNJ": "Health care",
}
GROUP_ORDER = ["Semiconductors", "Software & hardware", "Communication", "Consumer", "Financials",
               "Health care", "Other"]


def spy_top_holdings(top: int) -> pd.Series:
    import yfinance as yf
    yf.set_tz_cache_location(tempfile.mkdtemp(prefix="yf_cache_"))
    th = yf.Ticker(BENCH).funds_data.top_holdings
    w = th["Holding Percent"].astype(float).head(top)
    w.index = [s.replace(".", "-").upper() for s in w.index]
    return w


def build(top: int, years: int) -> tuple[dict, str]:
    index_w = spy_top_holdings(top)
    covered = float(index_w.sum())
    w = index_w / covered                                             # rescaled to 100%
    order = sorted(w.index, key=lambda t: (GROUP_ORDER.index(GROUPS.get(t, "Other")), -w[t]))
    w = w.reindex(order)
    tickers = list(w.index)

    end = pd.Timestamp.today().normalize()                            # exclusive: drops today's partial bar
    start = end - pd.DateOffset(years=years)
    prices = rm.fetch_prices(tickers, start, end)
    cfg = rm.RiskModelConfig(tickers=tickers, weights=w.tolist(), end=end, start=start)
    report = rm.PortfolioRiskModel(cfg, prices=prices).run()
    vc = rm.volatility_comparison(report.port_returns, report.ewma_vol, rm.fetch_prices([BENCH], start, end),
                                  {"Benchmark": BENCH}, cfg.ewma_lambda)
    bench_col = vc.returns.columns[1]
    rf, rf_src = an.fetch_risk_free(start, end)
    perf = an.performance_metrics(vc.returns, rf, bench_col)

    clean = rm.clean_prices(prices[tickers])
    cov_daily = rm.ledoit_wolf_cov(clean.log_returns)[0].loc[tickers, tickers]
    simple = clean.simple_returns.reindex(vc.returns.index)[tickers]
    # The page recomputes skew/kurtosis from these rows, so they must be exactly the model's dates.
    assert len(vc.returns) == len(report.port_returns), "benchmark alignment dropped dates"
    assert not simple.isna().any().any()
    assert np.allclose(simple.to_numpy() @ w.to_numpy(), vc.returns["Portfolio"].to_numpy(), atol=1e-12)

    comp = report.components.drop(index="TOTAL")
    baseline = {
        "vol": float(report.summary.loc["Ann. Volatility", "LedoitWolf"]),
        "var95_1d": float(report.summary.loc["VaR 95% (Cornish-Fisher)", "LedoitWolf"]),
        "sharpe": float(perf.loc["Sharpe ratio", "Portfolio"]),
        "sharpe_bench": float(perf.loc["Sharpe ratio", bench_col]),
        "total": float(perf.loc["Total return", "Portfolio"]),
        "mdd": float(perf.loc["Max drawdown", "Portfolio"]),
        "risk_share": [float(comp.loc[t, "CompVaR_95_pct"]) for t in tickers],
    }
    m = report.meta
    data = {
        "label": f"Demo: SPY's top {len(tickers)} holdings ({covered:.0%} of the fund), rescaled to 100%",
        "resetLabel": "Reset to SPY weights",
        "asOf": str(m["window_end"]),
        "window": {"start": str(m["window_start"]), "end": str(m["window_end"]), "days": int(m["n_obs"])},
        "tickers": tickers,
        "groups": [GROUPS.get(t, "Other") for t in tickers],
        "weights": [float(x) for x in w],
        "covDaily": [[float(f"{v:.12g}") for v in row] for row in cov_daily.to_numpy()],
        "returns": np.round(simple.to_numpy(), 9).tolist(),
        "benchmark": {"symbol": BENCH, "returns": np.round(vc.returns[bench_col].to_numpy(), 9).tolist()},
        "rf": rf, "rfSource": rf_src,
        "z95": float(stats.norm.ppf(0.05)),
        "irr": None,
        "baseline": baseline,
    }
    template = (HERE / "widget_template.html").read_text(encoding="utf-8")
    assert template.count(PLACEHOLDER) == 1
    html = template.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))
    return data, html


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--top", type=int, default=10, help="number of SPY holdings (yfinance lists the top 10)")
    p.add_argument("--years", type=int, default=1, help="years of daily history")
    p.add_argument("--out", type=Path, default=HERE / "wix_risk_widget.html")
    a = p.parse_args()
    data, html = build(a.top, a.years)
    a.out.write_text(html, encoding="utf-8")
    b = data["baseline"]
    print(f"Wrote {a.out} ({len(html) / 1024:.0f} KB): {', '.join(data['tickers'])}")
    print(f"  window {data['window']['start']} -> {data['window']['end']} ({data['window']['days']} days), rf {data['rf']:.2%}")
    print(f"  vol {b['vol']:.1%}  VaR95 1d {b['var95_1d']:.2%}  Sharpe {b['sharpe']:.2f} vs SPY {b['sharpe_bench']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
