"""Interactive risk dashboard on top of risk_model.py.

Run:  py -m streamlit run dashboard.py
All inputs live in the sidebar and scope every chart; each chart has a table view underneath.
"""
from __future__ import annotations

import datetime as dt
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import portfolio_import as pi
import risk_model as rm

st.set_page_config(page_title="Portfolio Risk", layout="wide")

# Reference palette (dataviz skill, pre-validated). Three categorical slots for bars/heatmaps
# (all-pairs safe); slot 4 only on line/bar charts, which validate on the adjacent pairlist.
PALETTES: dict[str, dict[str, str]] = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", s1_soft="#b7d3f6",
                  div_neg="#1c5cab", div_mid="#f0efec", div_pos="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500", s1_soft="#184f95",
                 div_neg="#3987e5", div_mid="#383835", div_pos="#e66767"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
PLOT_CONFIG = {"displayModeBar": False}


def _palette() -> dict[str, str]:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return PALETTES["dark" if theme == "dark" else "light"]


P = _palette()


def styled(fig: go.Figure, height: int = 340, pct_y: bool = False, pct_x: bool = False) -> go.Figure:
    fig.update_layout(
        template="none", height=height, paper_bgcolor=P["surface"], plot_bgcolor=P["surface"],
        font=dict(family=FONT, color=P["ink2"], size=13),
        margin=dict(l=8, r=16, t=40, b=8),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(color=P["ink2"])),
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["axis"], font=dict(family=FONT, color=P["ink"])),
        bargap=0.3, bargroupgap=0.06,
    )
    axis = dict(gridcolor=P["grid"], linecolor=P["axis"], zerolinecolor=P["axis"], zerolinewidth=1,
                tickfont=dict(color=P["muted"]), showline=True, automargin=True)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    if pct_x:
        fig.update_xaxes(tickformat=".0%")
    return fig


def chart(fig: go.Figure, table: pd.DataFrame, fmt: str | dict[str, str] = "{:.2%}") -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with st.expander("Table view"):
        st.dataframe(table.style.format(fmt, na_rep=""), width="stretch")


def end_labels(fig: go.Figure, ends: pd.Series, fmt: str, data: pd.DataFrame, label_px: float = 18.0) -> None:
    """Direct-label line ends, nudged at least `label_px` apart so close lines don't overprint.

    The gap is converted from pixels to data units using the plotted y-range (`data`, plus zero when
    the axis starts there) and the plot-area height.
    """
    ends = ends.sort_values()
    lo, hi = min(float(np.nanmin(data.to_numpy())), 0.0), float(np.nanmax(data.to_numpy()))
    plot_px = max((fig.layout.height or 340) - 110, 100)  # height minus title/legend/axis bands
    gap, prev = label_px / plot_px * ((hi - lo) or 1.0), -np.inf
    for name, v in ends.items():
        prev = max(float(v), prev + gap)
        short = "Portfolio" if name == "Portfolio" else str(name).split("(")[-1].rstrip(")")
        fig.add_annotation(xref="paper", x=1.0, y=prev, text=f"{short} {fmt.format(v)}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=P["ink"], size=12))


# ------------------------------------------------------------------------------------------
# Inputs (sidebar scopes everything)
# ------------------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(symbols: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return rm.fetch_prices(list(symbols), start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_risk_free(start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, str]:
    return an.fetch_risk_free(start, end)


# Uploaded files are parsed per run and never cached: st.cache_data is shared server memory.
def parse_positions(data: bytes) -> pi.PositionsImport:
    return pi.parse_schwab_positions(data)


def parse_transactions(data: bytes) -> pi.TransactionsImport:
    return pi.parse_schwab_transactions(data)


with st.sidebar:
    st.header("Import")
    pos_file = st.file_uploader("Schwab positions CSV", type="csv",
                                help="Sets holdings, weights (market value / account value) and portfolio value.")
    tx_file = st.file_uploader("Schwab transactions CSV (optional)", type="csv",
                               help="With a positions file: reconciles share counts. On its own: rebuilds "
                                    "holdings as of the window end and values them at the last close.")
positions: pi.PositionsImport | None = None
transactions: pi.TransactionsImport | None = None
try:
    if pos_file is not None:
        positions = parse_positions(pos_file.getvalue())
    if tx_file is not None:
        transactions = parse_transactions(tx_file.getvalue())
except ValueError as exc:
    st.sidebar.error(f"Import failed: {exc}")

with st.sidebar:
    st.header("Window")
    end_date = st.date_input("Window end (exclusive)", value=dt.date.today(),
                             help="Prices up to the day before this date. Today's partial bar is excluded.")
    inception = transactions.first_date.date() if transactions is not None else None
    use_inception = inception is not None and st.checkbox(
        f"Start at account inception ({inception})", value=False,
        help="Off by default. The risk model applies today's weights to how these stocks actually moved, so "
             "price history from before you owned them is still valid evidence; starting at inception only "
             "shortens the sample. Turn it on to see risk over the account's own lifetime.")
    start_date = st.date_input("Window start (inclusive)",
                               value=inception if use_inception else (pd.Timestamp(end_date) - pd.DateOffset(years=1)).date(),
                               min_value=dt.date(2000, 1, 1), max_value=end_date - dt.timedelta(days=1),
                               disabled=use_inception,
                               help="An earlier start gives more data. The effective start is the first day every "
                                    "holding has prices (a recently listed holding can push it later).")
    if use_inception:
        start_date = inception

end_ts = pd.Timestamp(end_date)
start_ts = pd.Timestamp(start_date)
if start_ts >= end_ts:
    st.error(f"Window start {start_ts.date()} must be before the window end {end_ts.date()}.")
    st.stop()

source_note = None
default_value = 0.0
if positions is not None:
    default_holdings = pd.DataFrame({"Ticker": positions.tickers, "Weight": positions.weights})
    default_value = positions.total_value
    as_of = positions.as_of.date() if positions.as_of is not None else "?"
    source_note = f"From positions file: account {positions.account}, as of {as_of}"
    editor_key = f"pos-{pos_file.file_id}"
elif transactions is not None:
    qty = transactions.holdings_as_of(end_ts - pd.Timedelta(days=1))
    try:
        last_px = load_prices(tuple(sorted(qty.index)), start_ts, end_ts).ffill().iloc[-1]
        t_list, w_list, default_value = pi.weights_from_quantities(qty, last_px)
    except Exception as exc:
        st.error(f"Could not value the rebuilt holdings: {exc}")
        st.stop()
    default_holdings = pd.DataFrame({"Ticker": t_list, "Weight": w_list})
    source_note = f"Rebuilt from transactions as of the last close before {end_ts.date()} (cash balance unknown)"
    editor_key = f"tx-{tx_file.file_id}-{end_ts.date()}"
else:
    # No built-in portfolio: the app starts blank until a file is uploaded or tickers are typed in.
    default_holdings = pd.DataFrame({"Ticker": pd.Series(dtype="str"), "Weight": pd.Series(dtype="float")})
    editor_key = "blank"

with st.sidebar:
    st.header("Holdings")
    if source_note:
        st.caption(source_note)
    holdings = st.data_editor(
        default_holdings, key=f"holdings-{editor_key}",
        num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"Ticker": st.column_config.TextColumn(help="Yahoo Finance symbol, e.g. AAPL or BRK-B"),
                       "Weight": st.column_config.NumberColumn(
                           format="%.4f", step=0.0001,
                           help="A fraction (0.25) or a percent (25). Leave every weight blank for equal weight.")},
    )
    notional = st.number_input("Portfolio value ($)", min_value=0.0, value=float(default_value), step=100.0,
                               format="%.2f", key=f"value-{editor_key}",
                               help="Used for dollar VaR/ES. Leave at 0 to show percentages only.")
    st.header("Model")
    method_label = st.radio("Covariance", ["Ledoit-Wolf", "EWMA"], horizontal=True,
                            help="Drives the headline tiles, component risk and correlation heatmap.")
    lam = st.slider("EWMA lambda", 0.80, 0.99, 0.94, 0.01)
    conf = st.radio("Confidence", [0.95, 0.99], index=1, horizontal=True, format_func=lambda c: f"{c:.0%}")
    normalize = st.checkbox("Rescale weights to 100% (no cash)", value=False)
    include_mean = st.checkbox("Use sample mean (default: zero mean)", value=False)
    st.header("Comparison")
    st.caption("Reference portfolios for the volatility comparison (any Yahoo ticker).")
    references = {name: st.text_input(name, value=sym).strip().upper()
                  for name, sym in rm.DEFAULT_REFERENCES.items()}
    references = {k: v for k, v in references.items() if v}
    st.header("Analysis")
    rf_default, rf_source = load_risk_free(start_ts, end_ts)
    rf = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=15.0, value=round(rf_default * 100, 2),
                         step=0.25, help=f"Default: {rf_source}.") / 100.0
    if abs(rf - rf_default) > 1e-9:
        rf_source = "user input"
    erp = st.number_input("Equity risk premium (%)", min_value=0.0, max_value=15.0, value=an.ERP_DEFAULT * 100,
                          step=0.5, help="Drives CAPM expected returns in the long-term projections.") / 100.0
    horizon = st.slider("Projection horizon (years)", 1, 20, 5)
    cap = st.slider("Single-name cap for the rebalancing scenario", 0.05, 0.50, 0.20, 0.05,
                    format="%.2f", help="Excess weight is spread across the other holdings pro rata.")
    mu_label = st.radio("Frontier expected returns", ["CAPM", "Historical (window)"], horizontal=True,
                        help="CAPM: risk-free + beta x equity premium (default). Historical: mean "
                             "returns over the window, which are noisy and mostly reflect what already went up.")
    mu_source = "capm" if mu_label == "CAPM" else "historical"

holdings = holdings.copy()
holdings["Ticker"] = holdings["Ticker"].fillna("").astype(str).str.strip().str.upper()
holdings = holdings[holdings["Ticker"] != ""]
if len(holdings) and holdings["Weight"].isna().all():
    holdings["Weight"] = 1.0 / len(holdings)                          # tickers only: equal weight
elif holdings["Weight"].isna().any():
    st.error("Give every ticker a weight, or leave all the weights blank for equal weighting.")
    st.stop()
holdings = holdings[holdings["Weight"] != 0]
if holdings["Weight"].sum() > 1.5:                                    # typed as percentages
    holdings["Weight"] = holdings["Weight"] / 100.0
if len(holdings) < 2:
    st.title("Portfolio risk")
    st.info("**Start by adding a portfolio** in the sidebar:\n\n"
            "- **Upload** a Schwab positions CSV (transactions CSV optional), or\n"
            "- **Type** at least two tickers under *Holdings*, with weights as fractions (0.25) or percents (25). "
            "Leave the weights blank for equal weighting.\n\n"
            "Prices come from Yahoo Finance. Uploaded files are processed in memory for your session and "
            "not saved.",
            icon=":material/upload_file:")
    st.stop()
if holdings["Ticker"].duplicated().any():
    st.error(f"Duplicate tickers: {sorted(holdings.loc[holdings['Ticker'].duplicated(), 'Ticker'])}")
    st.stop()

tickers = holdings["Ticker"].tolist()
weights = holdings["Weight"].astype(float).tolist()
method = "ledoit_wolf" if method_label == "Ledoit-Wolf" else "ewma"
alt_label = "EWMA" if method == "ledoit_wolf" else "LedoitWolf"
pct = int(round(conf * 100))


try:
    prices = load_prices(tuple(sorted(tickers)), start_ts, end_ts)
    cfg = rm.RiskModelConfig(tickers=tickers, weights=weights, end=end_ts, start=start_ts,
                             ewma_lambda=lam, primary_method=method, normalize_weights=normalize,
                             include_mean=include_mean, notional=notional or None)
    report = rm.PortfolioRiskModel(cfg, prices=prices).run()
except Exception as exc:  # surfaced to the operator rather than a stack trace
    st.error(f"Model run failed: {exc}")
    st.stop()

meta = report.meta
summary = report.summary
comp = report.components.drop(index="TOTAL")
col_m = rm.METHOD_LABEL[method]
port = report.port_returns

# Reference comparison and analysis are computed once here; every section below reads these objects.
vc: rm.VolComparison | None = None
analysis: an.Analysis | None = None
analysis_error = None
if references:
    try:
        ref_px = load_prices(tuple(sorted(set(references.values()))), start_ts, end_ts)
        vc = rm.volatility_comparison(port, report.ewma_vol, ref_px, references, lam)
        top_name = comp["CompVaR_99_pct"].idxmax()
        lots = pi.fifo_lots(transactions, top_name)[0] if transactions is not None else None
        analysis = an.build_analysis(report, cfg, prices, vc, rf, rf_source, erp=erp, horizon=horizon, cap=cap,
                                     notional=notional or None, positions=positions, lots=lots,
                                     mu_source=mu_source)
    except Exception as exc:
        analysis_error = str(exc)
entity_colors: dict[str, str] = (dict(zip(vc.returns.columns, (P["s1"], P["s2"], P["s3"], P["s4"])))
                                 if vc is not None else {})

# ------------------------------------------------------------------------------------------
# Header + headline tiles
# ------------------------------------------------------------------------------------------
st.title("Portfolio risk")
cash_txt = ("weights rescaled to 100%" if meta["normalized"]
            else f"{meta['cash_weight']:.2%} held as cash")
st.caption(f"{meta['window_start']} to {meta['window_end']} | {meta['n_obs']} daily returns "
           f"({meta['window_years']:.1f} years) | {cash_txt} | 1-day horizon, losses shown as positive % of "
           f"portfolio value")
if meta.get("late_start"):
    late_t, late_d = meta["late_start"]
    st.info(f"The window starts on {meta['window_start']}, later than the {start_ts.date()} you chose, because "
            f"{late_t} has no prices before {late_d}. Every holding needs the same dates.", icon=":material/info:")
if meta.get("short_window"):
    st.warning(f"Only {meta['n_obs']} daily returns in this window: VaR, ES and correlations are imprecise, and the "
               f"99% historical tail rests on {meta['hist_tail_days'][0.99]} day(s). Choose an earlier start.")
if not meta["cf_valid"]:
    st.warning("Cornish-Fisher is outside its valid range for this skew/kurtosis, so treat the CF figures with caution.")

if positions is not None or transactions is not None:
    issues: list[str] = list(positions.warnings) if positions is not None else []
    if transactions is not None and len(transactions.unhandled):
        issues.append(f"{len(transactions.unhandled)} transaction rows have an unknown share effect and were "
                      f"ignored: {sorted(transactions.unhandled['Action'].unique())}")
    rec = pi.reconcile(positions, transactions) if positions is not None and transactions is not None else None
    n_bad = 0 if rec is None else int((rec["Status"] != "ok").sum())
    label = "Import details"
    if rec is not None:
        label += f": {len(rec) - n_bad}/{len(rec)} symbols reconcile to the transaction history"
    with st.expander(label, expanded=bool(issues) or n_bad > 0):
        for msg in issues:
            st.warning(msg)
        if positions is not None:
            st.caption(f"Account value ${positions.total_value:,.2f} | cash ${positions.cash:,.2f} | "
                       f"{len(positions.holdings)} modelled holdings")
            st.dataframe(positions.holdings.drop(columns=["SchwabSymbol"]).style.format(
                {"Quantity": "{:,.4f}", "Price": "${:,.2f}", "MarketValue": "${:,.2f}", "Weight": "{:.2%}"}),
                width="stretch")
        if rec is not None:
            if n_bad:
                st.warning(f"{n_bad} symbols do not match. Trades after the transaction export's last date "
                           f"({transactions.last_date.date()}) are the usual cause.")
            st.dataframe(rec.style.format({"Positions": "{:,.4f}", "Transactions": "{:,.4f}",
                                           "Difference": "{:,.4f}"}, na_rep=""), width="stretch")
            basis = pi.cost_basis_check(positions, transactions)
            n_fifo = int(basis["FIFO matches"].sum())
            st.markdown(f"**Cost basis check:** FIFO lots reproduce the broker's cost basis for {n_fifo} of "
                        f"{len(basis)} positions. Where they don't, the account relieves lots another way "
                        f"(specific lot or tax-lot optimizer), so FIFO lot dates and gains are not shown for them.")
            st.dataframe(basis.style.format({"FIFO cost": "${:,.2f}", "Broker cost": "${:,.2f}",
                                             "Difference": "${:,.2f}"}, na_rep=""), width="stretch")

money = (lambda frac: {"delta": f"${frac * notional:,.0f}", "delta_color": "off", "delta_arrow": "off"}
         if notional else {})
top = comp[f"CompVaR_{pct}_pct"].idxmax()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Annualized vol", f"{summary.loc['Ann. Volatility', col_m]:.1%}", border=True,
          help=f"{method_label} covariance. Realized (historical): {summary.loc['Ann. Volatility', 'Historical']:.1%}.")
var_cf = summary.loc[f"VaR {pct}% (Cornish-Fisher)", col_m]
es_cf = summary.loc[f"ES {pct}% (Cornish-Fisher)", col_m]
var_h = summary.loc[f"VaR {pct}% (Historical)", "Historical"]
k2.metric(f"VaR {pct}% (Cornish-Fisher)", f"{var_cf:.2%}", border=True, **money(var_cf))
k3.metric(f"ES {pct}% (Cornish-Fisher)", f"{es_cf:.2%}", border=True, **money(es_cf))
k4.metric(f"VaR {pct}% (Historical)", f"{var_h:.2%}", border=True,
          help=f"Based on {meta['hist_tail_days'][conf]} tail days.", **money(var_h))
k5.metric("Largest risk contributor", f"{top} {comp.loc[top, f'CompVaR_{pct}_pct']:.0%}",
          f"on {comp.loc[top, 'Weight']:.0%} of capital", delta_color="off", delta_arrow="off", border=True)

if analysis is not None:
    st.info(f"**Bottom line:** {analysis.headline} Full recommendation at the end of the page.", icon=":material/flag:")
elif analysis_error:
    st.warning(f"Analysis sections unavailable: {analysis_error}")

# ------------------------------------------------------------------------------------------
# Row 1: VaR/ES by method | risk vs capital
# ------------------------------------------------------------------------------------------
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("VaR and ES by method")
    cats = ["VaR 95%", "ES 95%", "VaR 99%", "ES 99%"]
    series = {
        "Cornish-Fisher, Ledoit-Wolf": ("LedoitWolf", "Cornish-Fisher", P["s1"]),
        "Cornish-Fisher, EWMA": ("EWMA", "Cornish-Fisher", P["s2"]),
        "Historical": ("Historical", "Historical", P["s3"]),
    }
    tbl = pd.DataFrame(index=cats)
    fig = go.Figure()
    for name, (col, src, color) in series.items():
        vals = [summary.loc[f"{c.split()[0]} {c.split()[1]} ({src})", col] for c in cats]
        tbl[name] = vals
        fig.add_bar(name=name, x=cats, y=vals, marker=dict(color=color, cornerradius=4),
                    hovertemplate="%{x}<br>" + name + ": %{y:.2%}<extra></extra>")
    chart(styled(fig, pct_y=True), tbl)

with c2:
    st.subheader("Share of risk vs share of capital")
    share_col = f"CompVaR_{pct}_pct"
    d = comp.sort_values(share_col)
    fig = go.Figure()
    fig.add_bar(name="Capital weight", y=d.index, x=d["Weight"], orientation="h",
                marker=dict(color=P["s1"], cornerradius=4),
                hovertemplate="%{y} capital: %{x:.2%}<extra></extra>")
    fig.add_bar(name=f"Share of VaR ({method_label})", y=d.index, x=d[share_col], orientation="h",
                marker=dict(color=P["s2"], cornerradius=4),
                hovertemplate="%{y} share of VaR: %{x:.2%}<extra></extra>")
    fig = styled(fig, height=420, pct_x=True)
    fig.update_layout(bargap=0.25, bargroupgap=0.08)
    top_row = d.iloc[-1]
    fig.add_annotation(x=top_row[share_col], y=d.index[-1], text=f"{top_row[share_col]:.0%}", xanchor="left",
                       showarrow=False, xshift=4, yshift=6, font=dict(color=P["ink"]))
    chart(fig, d[["Weight", share_col, f"CompVaR_99_pct_{alt_label}", "HistCompES_95_pct"]].iloc[::-1]
          .rename(columns={share_col: f"Share of VaR ({method_label})",
                           f"CompVaR_99_pct_{alt_label}": f"Share of VaR ({alt_label})",
                           "HistCompES_95_pct": "Share of hist. ES 95%"}))

# ------------------------------------------------------------------------------------------
# Row 2: return distribution | volatility through time
# ------------------------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Daily portfolio returns")
    var_v = summary.loc[f"VaR {pct}% (Cornish-Fisher)", col_m]
    es_v = summary.loc[f"ES {pct}% (Cornish-Fisher)", col_m]
    fig = go.Figure()
    fig.add_histogram(x=port, nbinsx=50, name="Days", marker=dict(color=P["s1"], line=dict(color=P["surface"], width=1)),
                      hovertemplate="Return %{x}<br>%{y} days<extra></extra>", xbins=None)
    fig = styled(fig, pct_x=True)
    fig.update_layout(showlegend=False, bargap=0.04)
    # ES sits left of VaR, so its label goes left of its line and VaR's goes right: they never collide.
    for val, label, dash, side in [(-var_v, f"VaR {pct}%<br>{var_v:.2%}", "solid", "left"),
                                   (-es_v, f"ES {pct}%<br>{es_v:.2%}", "dot", "right")]:
        fig.add_vline(x=val, line=dict(color=P["ink"], width=1.5, dash=dash))
        fig.add_annotation(x=val, y=1, yref="paper", text=label, showarrow=False, xanchor=side, align=side,
                           xshift=4 if side == "left" else -4, yanchor="top", font=dict(color=P["ink"], size=12))
    fig.update_yaxes(title=dict(text="Days", font=dict(color=P["muted"])))
    worst = port.nsmallest(10).rename("Return").to_frame()
    worst.index = worst.index.date
    chart(fig, worst)

with c2:
    st.subheader("Volatility through time (annualized)")
    ann = np.sqrt(rm.TRADING_DAYS)
    vol = pd.DataFrame({f"EWMA (lambda {lam:.2f})": report.ewma_vol * ann,
                        "Realized, 21-day rolling": port.rolling(21).std() * ann}).iloc[20:]
    fig = go.Figure()
    for name, color in zip(vol.columns, (P["s1"], P["s2"])):
        fig.add_scatter(x=vol.index, y=vol[name], name=name, mode="lines", line=dict(color=color, width=2),
                        hovertemplate="%{y:.1%}<extra>" + name + "</extra>")
    fig = styled(fig, pct_y=True)
    fig.update_layout(hovermode="x unified")
    last = vol.iloc[-1, 0]
    fig.add_annotation(x=vol.index[-1], y=last, text=f"{last:.0%}", showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=P["ink"]))
    vt = vol.copy()
    vt.index = vt.index.date
    chart(fig, vt.iloc[::-1])

# ------------------------------------------------------------------------------------------
# Row 2b: volatility vs benchmark / aggressive / conservative references
# ------------------------------------------------------------------------------------------
if vc is not None:
    st.subheader("Volatility vs reference portfolios")
    bench_col = vc.horizons.columns[1]
    ratio_12m = vc.vs_benchmark.dropna()
    if len(ratio_12m):
        h_last = ratio_12m.index[-1]
        st.caption(f"Portfolio volatility is {ratio_12m.iloc[-1]:.1f}x the {bench_col} over {h_last}. "
                   f"References: " + ", ".join(f"{k} {v}" for k, v in references.items())
                   + " (AOA/AOK are iShares 80/20 and 30/70 allocation funds).")
    colors = entity_colors
    c1, c2 = st.columns(2)
    with c1:
        paths = vc.paths.iloc[20:]
        fig = go.Figure()
        for name in paths.columns:
            fig.add_scatter(x=paths.index, y=paths[name], name=name, mode="lines",
                            line=dict(color=colors[name], width=2.5 if name == "Portfolio" else 2),
                            hovertemplate="%{y:.1%}<extra>" + name + "</extra>")
        fig = styled(fig, height=360, pct_y=True)
        fig.update_layout(hovermode="x unified", margin=dict(r=120))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[paths.index[0], paths.index[-1]])
        end_labels(fig, paths.iloc[-1], "{:.0%}", paths)
        st.markdown("**EWMA volatility through time** (annualized)")
        vt = paths.copy()
        vt.index = vt.index.date
        chart(fig, vt.iloc[::-1])
    with c2:
        fig = go.Figure()
        for name in vc.horizons.columns:
            fig.add_bar(name=name, x=vc.horizons.index, y=vc.horizons[name],
                        marker=dict(color=colors[name], cornerradius=4),
                        hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>")
        fig = styled(fig, height=360, pct_y=True)
        st.markdown("**Realized volatility by trailing window** (annualized)")
        tbl = vc.horizons.copy()
        tbl["Portfolio / benchmark"] = vc.vs_benchmark
        fmt_h = {c: "{:.1%}" for c in vc.horizons.columns} | {"Portfolio / benchmark": "{:.2f}x"}
        chart(fig, tbl, fmt_h)

# ------------------------------------------------------------------------------------------
# Row 3: cumulative return | drawdown, portfolio vs references (one axis each)
# ------------------------------------------------------------------------------------------
series_returns = vc.returns if vc is not None else port.to_frame("Portfolio")
line_colors = entity_colors or {"Portfolio": P["s1"]}
wd = {c: an.wealth_drawdown(series_returns[c]) for c in series_returns.columns}
cum = pd.DataFrame({c: w - 1 for c, (w, _) in wd.items()})
dd = pd.DataFrame({c: d for c, (_, d) in wd.items()})
st.caption("Historical lines are a backtest of TODAY'S weights held constant over the window, not the "
           "account's actual return. Today's weights reflect which names already rose, which flatters the backtest.")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Cumulative return")
    fig = go.Figure()
    for c in cum.columns:
        fig.add_scatter(x=cum.index, y=cum[c], mode="lines", name=c,
                        line=dict(color=line_colors[c], width=2.5 if c == "Portfolio" else 2),
                        hovertemplate="%{y:.1%}<extra>" + c + "</extra>")
    fig = styled(fig, height=340, pct_y=True)
    fig.update_layout(hovermode="x unified", margin=dict(r=120))
    fig.update_xaxes(range=[cum.index[0], cum.index[-1]])
    end_labels(fig, cum.iloc[-1], "{:+.0%}", cum)
    t = cum.copy()
    t.index = t.index.date
    chart(fig, t.iloc[::-1])
with c2:
    st.subheader("Drawdown from peak")
    fig = go.Figure()
    for c in dd.columns:
        fig.add_scatter(x=dd.index, y=dd[c], mode="lines", name=c,
                        line=dict(color=line_colors[c], width=2.5 if c == "Portfolio" else 2),
                        hovertemplate="%{y:.1%}<extra>" + c + "</extra>")
    fig = styled(fig, height=340, pct_y=True)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(range=[dd.index[0], dd.index[-1]])
    pdd = dd["Portfolio"]
    fig.add_annotation(x=pdd.idxmin(), y=pdd.min(), text=f"Portfolio max {pdd.min():.1%}", showarrow=False,
                       yanchor="top", yshift=-4, font=dict(color=P["ink"]))
    chart(fig, pd.DataFrame({"Max drawdown": dd.min(), "Trough date": [d.date() for d in dd.idxmin()]}),
          {"Max drawdown": "{:.2%}"})

# ------------------------------------------------------------------------------------------
# Performance, long-term upside/downside, rebalancing scenarios (all from `analysis`)
# ------------------------------------------------------------------------------------------
if analysis is not None:
    perf = analysis.performance
    names = list(perf.columns)
    st.subheader("Risk-adjusted performance vs reference portfolios")
    st.caption(f"Backtest of today's weights; risk-free rate {rf:.2%} ({rf_source}). Whiskers show +/- 1 standard "
               f"error of the Sharpe ratio: with {meta['window_years']:.1f} years of data, overlapping whiskers mean "
               f"the difference is noise.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sharpe ratio**")
        fig = go.Figure()
        fig.add_bar(x=names, y=perf.loc["Sharpe ratio"], marker=dict(color=[entity_colors[n] for n in names],
                    cornerradius=4), error_y=dict(type="data", array=perf.loc["Sharpe std. error"],
                    color=P["ink2"], thickness=1.5, width=6),
                    customdata=perf.loc["Sharpe std. error"],
                    hovertemplate="%{x}<br>Sharpe %{y:.2f} +/- %{customdata:.2f}<extra></extra>")
        fig = styled(fig, height=320)
        fig.update_layout(showlegend=False)
        chart(fig, perf.loc[["Sharpe ratio", "Sharpe std. error", "Sortino ratio", "Calmar ratio"]].T, "{:.2f}")
    with c2:
        st.markdown("**Max drawdown**")
        fig = go.Figure()
        fig.add_bar(x=names, y=perf.loc["Max drawdown"], marker=dict(color=[entity_colors[n] for n in names],
                    cornerradius=4), text=[f"{v:.1%}" for v in perf.loc["Max drawdown"]], textposition="outside",
                    textfont=dict(color=P["ink"]), hovertemplate="%{x}<br>%{y:.1%}<extra></extra>")
        fig = styled(fig, height=320, pct_y=True)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[float(perf.loc["Max drawdown"].min()) * 1.25, 0])
        chart(fig, perf.loc[["Max drawdown", "Worst day", "Annualized volatility"]].T)
    with st.expander("All performance metrics"):
        pct_rows = ["Total return", "Annualized return", "Annualized volatility", "Max drawdown", "Worst day"]
        shown = perf.copy().astype(object)
        for r_ in perf.index:
            shown.loc[r_] = [f"{v:.1%}" if r_ in pct_rows else f"{v:.2f}" for v in perf.loc[r_]]
        st.dataframe(shown, width="stretch")

    # ---- long-term upside / downside -----------------------------------------------------
    st.subheader("Long-term upside and downside")
    proj = analysis.projections
    inputs = proj.xs(horizon, level="Years")[["Beta", "Expected return (CAPM)", "Volatility (log)"]]
    st.caption(f"Lognormal projection with CAPM expected returns: risk-free {rf:.2%} + beta x {erp:.1%} equity premium. "
               f"These ranges come from the assumptions, not from past performance; the equity-only premium "
               f"understates bond-heavy funds (AOA, AOK).")
    base = float(notional) if notional else 10_000.0
    base_lbl = f"${base:,.0f}"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Range of outcomes for {base_lbl} after {horizon} years**")
        at_h = proj.xs(horizon, level="Years")
        fig = go.Figure()
        for n in at_h.index[::-1]:
            row, col = at_h.loc[n], entity_colors[n]
            fig.add_scatter(x=[row["P5"] * base, row["P95"] * base], y=[n, n], mode="lines",
                            line=dict(color=col, width=3), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=[row["P25"] * base, row["P75"] * base], y=[n, n], mode="lines",
                            line=dict(color=col, width=12), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=[row["P50"] * base], y=[n], mode="markers", showlegend=False,
                            marker=dict(color=P["surface"], size=11, line=dict(color=P["ink"], width=2)),
                            customdata=[[row["P5"] * base, row["P25"] * base, row["P75"] * base, row["P95"] * base,
                                         row["Prob. of loss"]]],
                            hovertemplate=(n + "<br>median $%{x:,.0f}<br>25-75%: $%{customdata[1]:,.0f} - "
                                           "$%{customdata[2]:,.0f}<br>5-95%: $%{customdata[0]:,.0f} - "
                                           "$%{customdata[3]:,.0f}<br>P(loss) %{customdata[4]:.0%}<extra></extra>"))
        fig.add_vline(x=base, line=dict(color=P["muted"], width=1))
        fig = styled(fig, height=320)
        fig.update_xaxes(type="log", tickprefix="$", tickformat=",.0f", title=dict(
            text="thin line 5th-95th percentile, thick 25th-75th, dot median (log scale)",
            font=dict(color=P["muted"], size=11)))
        tbl = at_h[["P5", "P25", "P50", "P75", "P95"]] * base
        tbl["Prob. of loss"] = at_h["Prob. of loss"]
        tbl["Median annual growth"] = at_h["Median annual growth"]
        chart(fig, tbl.join(inputs), {c: "${:,.0f}" for c in ["P5", "P25", "P50", "P75", "P95"]}
              | {"Prob. of loss": "{:.0%}", "Median annual growth": "{:.1%}", "Beta": "{:.2f}",
                 "Expected return (CAPM)": "{:.1%}", "Volatility (log)": "{:.1%}"})
    with c2:
        st.markdown("**Chance of ending below the starting value, by horizon**")
        years = list(range(1, max(20, horizon) + 1))
        ploss = pd.DataFrame({n: [an.lognormal_outcomes(float(inputs.loc[n, "Expected return (CAPM)"]),
                                                        float(inputs.loc[n, "Volatility (log)"]), y)["Prob. of loss"]
                                  for y in years] for n in inputs.index}, index=years)
        fig = go.Figure()
        for n in ploss.columns:
            fig.add_scatter(x=years, y=ploss[n], mode="lines", name=n,
                            line=dict(color=entity_colors[n], width=2.5 if n == "Portfolio" else 2),
                            hovertemplate="%{y:.0%}<extra>" + n + "</extra>")
        fig = styled(fig, height=320, pct_y=True)
        fig.update_layout(hovermode="x unified", margin=dict(r=120))
        fig.update_xaxes(title=dict(text="years", font=dict(color=P["muted"])), range=[1, years[-1]])
        fig.update_yaxes(rangemode="tozero")
        end_labels(fig, ploss.iloc[-1], "{:.0%}", ploss)
        chart(fig, ploss)

    # ---- rebalancing scenarios -----------------------------------------------------------
    st.subheader("Rebalancing scenarios")
    sc = analysis.scenarios
    st.caption("Each scenario is re-run through the full model (its own covariance, Cornish-Fisher tails and "
               "component risk). Cash is unchanged; the cap spreads excess weight pro rata across the other holdings.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1-day 99% VaR by scenario**")
        fig = go.Figure()
        fig.add_bar(x=sc.index, y=sc["VaR 99%"].astype(float), marker=dict(color=P["s1"], cornerradius=4),
                    text=[f"{v:.2%}" + (f" (${v * notional:,.0f})" if notional else "") for v in sc["VaR 99%"]],
                    textposition="outside", textfont=dict(color=P["ink"]),
                    hovertemplate="%{x}<br>VaR 99% %{y:.2%}<extra></extra>")
        fig = styled(fig, height=320, pct_y=True)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, float(sc["VaR 99%"].max()) * 1.2])
        chart(fig, sc[["VaR 99%", "ES 99%", "Ann. volatility"]].astype(float))
    with c2:
        st.markdown("**Projected median annual growth by scenario**")
        fig = go.Figure()
        fig.add_bar(x=sc.index, y=sc["Median annual growth"].astype(float), marker=dict(color=P["s1"], cornerradius=4),
                    text=[f"{v:.1%}" for v in sc["Median annual growth"]], textposition="outside",
                    textfont=dict(color=P["ink"]), hovertemplate="%{x}<br>%{y:.1%}<extra></extra>")
        fig = styled(fig, height=320, pct_y=True)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[min(0.0, float(sc["Median annual growth"].min()) * 1.2),
                                float(sc["Median annual growth"].max()) * 1.25])
        chart(fig, sc[["Expected return (CAPM)", "Volatility (log)" if "Volatility (log)" in sc else "Ann. volatility",
                       "Median annual growth"]].astype(float))
    fmt_sc = {c: "{:.1%}" for c in sc.columns if c not in ("Top risk name", "Effective N", "Beta", "VaR 99% ($)")}
    fmt_sc |= {"Effective N": "{:.1f}", "Beta": "{:.2f}", "VaR 99% ($)": "${:,.0f}", "VaR 99%": "{:.2%}",
               "ES 99%": "{:.2%}"}
    st.dataframe(sc.style.format(fmt_sc, na_rep=""), width="stretch")

# ------------------------------------------------------------------------------------------
# Markowitz efficient frontier (from analysis.frontier)
# ------------------------------------------------------------------------------------------
if analysis is not None and analysis.frontier is not None:
    fr = analysis.frontier
    st.subheader("Efficient frontier (Markowitz)")
    src_txt = (f"CAPM expected returns (risk-free {rf:.2%} + beta x {erp:.1%} premium)" if fr.mu_source == "capm"
               else "historical mean returns over the window (noisy: they mostly reflect what already went up)")
    st.caption(f"Long-only portfolios of your current holdings, cash held at {1 - fr.invested:.1%}. Risk from the "
               f"Ledoit-Wolf covariance; {src_txt}. Points above the curve are impossible with these holdings; "
               f"points below it are inefficient.")
    pts, g = fr.points, fr.gaps
    fig = go.Figure()
    styles = {"Long-only": dict(color=P["s1"], width=2.5),
              **{k: dict(color=P["s1"], width=2, dash="dash") for k in fr.curves if k != "Long-only"}}
    for name, curve in fr.curves.items():
        fig.add_scatter(x=curve["Volatility"], y=curve["Expected return"], mode="lines", name=f"Frontier: {name}",
                        line=styles[name], customdata=curve["Sharpe"],
                        hovertemplate="vol %{x:.1%}<br>return %{y:.1%}<br>Sharpe %{customdata:.2f}<extra>" + name + "</extra>")
    # Capital market line through the max-Sharpe portfolio.
    ms = pts.loc["Max Sharpe"]
    x_max = float(max(fr.assets["Volatility"].max(), pts["Volatility"].max())) * 1.05
    fig.add_scatter(x=[0, x_max], y=[rf, rf + ms["Sharpe"] * x_max], mode="lines", name="Capital market line",
                    line=dict(color=P["muted"], width=1.5, dash="dot"), hoverinfo="skip")
    # Individual holdings.
    fig.add_scatter(x=fr.assets["Volatility"], y=fr.assets["Expected return"], mode="markers+text",
                    name="Holdings", text=fr.assets.index, textposition="top center",
                    textfont=dict(color=P["muted"], size=10), marker=dict(color=P["muted"], size=7),
                    customdata=fr.assets["Beta"],
                    hovertemplate="%{text}<br>vol %{x:.1%}<br>return %{y:.1%}<br>beta %{customdata:.2f}<extra></extra>")
    # Reference funds (same colors as everywhere else on the page).
    if vc is not None:
        for ref in vc.returns.columns[1:]:
            vol_r = float(analysis.performance.loc["Annualized volatility", ref])
            mu_r = float(analysis.projections.loc[(ref, horizon), "Expected return (CAPM)"])
            fig.add_scatter(x=[vol_r], y=[mu_r], mode="markers", name=ref,
                            marker=dict(color=entity_colors.get(ref, P["muted"]), size=11, symbol="circle-open",
                                        line=dict(width=2.5)),
                            hovertemplate=ref + "<br>vol %{x:.1%}<br>return %{y:.1%}<extra></extra>")
    # Named portfolios.
    marks = {"Current": "circle", "Min variance": "triangle-up", "Max Sharpe": "star",
             "Equal weight": "square", **{k: "diamond" for k in pts.index if k.startswith("Cap ")}}
    for name, sym in marks.items():
        if name not in pts.index:
            continue
        r_ = pts.loc[name]
        big = name == "Current"
        fig.add_scatter(x=[r_["Volatility"]], y=[r_["Expected return"]], mode="markers+text", name=name,
                        text=[name], textposition="middle right" if name != "Min variance" else "middle left",
                        textfont=dict(color=P["ink"], size=12 if big else 11),
                        marker=dict(color=P["ink"] if big else P["surface"], size=14 if big else 11, symbol=sym,
                                    line=dict(color=P["ink"], width=2)),
                        hovertemplate=name + "<br>vol %{x:.1%}<br>return %{y:.1%}<br>Sharpe " + f"{r_['Sharpe']:.2f}"
                                      + "<extra></extra>")
    fig = styled(fig, height=520, pct_x=True, pct_y=True)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, yanchor="top"))
    fig.update_xaxes(title=dict(text="Annualized volatility", font=dict(color=P["muted"])), rangemode="tozero")
    fig.update_yaxes(title=dict(text="Expected return", font=dict(color=P["muted"])))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    c1, c2, c3 = st.columns(3)
    c1.metric("Frontier return at your volatility", f"{g['frontier_return_at_current_vol']:.1%}",
              f"{g['return_gap_same_vol']:+.1%} vs now", delta_color="off", delta_arrow="off", border=True)
    c2.metric("Volatility for your expected return", f"{g['frontier_vol_at_current_return']:.1%}",
              f"{-g['vol_saving_same_return']:+.1%} vs now", delta_color="off", delta_arrow="off", border=True)
    c3.metric("Max-Sharpe portfolio", f"Sharpe {ms['Sharpe']:.2f}", f"vs {pts.loc['Current', 'Sharpe']:.2f} now",
              delta_color="off", delta_arrow="off", border=True,
              help="Expected Sharpe from the frontier's expected returns, not the historical Sharpe shown above.")
    with st.expander("Table view: named portfolios and their weights"):
        st.dataframe(pts.style.format({"Volatility": "{:.1%}", "Expected return": "{:.1%}", "Sharpe": "{:.2f}"}),
                     width="stretch")
        st.dataframe(fr.weights.T.style.format("{:.1%}"), width="stretch")
        st.dataframe(fr.assets.style.format({"Volatility": "{:.1%}", "Expected return": "{:.1%}", "Beta": "{:.2f}"}),
                     width="stretch")

# ------------------------------------------------------------------------------------------
# Row 4: correlation heatmap | risk by sector
# ------------------------------------------------------------------------------------------
c1, c2 = st.columns([3, 2])
order = rm.sector_order(tickers, weights)
with c1:
    st.subheader(f"Correlation ({method_label})")
    corr = report.corr[method].loc[order, order]
    z = corr.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=order, y=order, zmin=-1, zmax=1, xgap=2, ygap=2,
        colorscale=[[0, P["div_neg"]], [0.5, P["div_mid"]], [1, P["div_pos"]]],
        colorbar=dict(tickformat=".1f", outlinewidth=0, tickfont=dict(color=P["muted"]), thickness=12),
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    dark = P is PALETTES["dark"]
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            v = z[i, j]
            white = dark or v < -0.6
            fig.add_annotation(x=b, y=a, text=f"{v:.2f}", showarrow=False,
                               font=dict(size=11, color="#ffffff" if white else P["ink"]))
    groups = [rm.SECTORS.get(t, "Other") for t in order]
    for k in range(1, len(order)):
        if groups[k] != groups[k - 1]:
            for kw in (dict(x0=k - 0.5, x1=k - 0.5, y0=-0.5, y1=len(order) - 0.5),
                       dict(y0=k - 0.5, y1=k - 0.5, x0=-0.5, x1=len(order) - 0.5)):
                fig.add_shape(type="line", line=dict(color=P["ink2"], width=2), **kw)
    fig = styled(fig, height=560)
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False)
    fig.update_xaxes(showgrid=False, showline=False, side="top")
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    st.caption("Groups (inferred): " + " | ".join(
        f"{g}: {', '.join(t for t in order if rm.SECTORS.get(t, 'Other') == g)}" for g in dict.fromkeys(groups)))
    with st.expander("Table view"):
        st.dataframe(corr.style.format("{:.2f}"), width="stretch")

with c2:
    st.subheader("Risk by sector")
    sec = comp.groupby("Sector")[["Weight", f"CompVaR_{pct}_pct"]].sum().sort_values(f"CompVaR_{pct}_pct")
    fig = go.Figure()
    fig.add_bar(name="Capital weight", y=sec.index, x=sec["Weight"], orientation="h",
                marker=dict(color=P["s1"], cornerradius=4), hovertemplate="%{y} capital: %{x:.1%}<extra></extra>")
    fig.add_bar(name="Share of VaR", y=sec.index, x=sec[f"CompVaR_{pct}_pct"], orientation="h",
                marker=dict(color=P["s2"], cornerradius=4), hovertemplate="%{y} share of VaR: %{x:.1%}<extra></extra>")
    fig = styled(fig, height=360, pct_x=True)
    chart(fig, sec.iloc[::-1].rename(columns={f"CompVaR_{pct}_pct": "Share of VaR"}))

# ------------------------------------------------------------------------------------------
# Detail tables + downloads
# ------------------------------------------------------------------------------------------
st.subheader("Risk metrics (detail)")
st.dataframe(summary.style.format("{:.2%}", na_rep=""), width="stretch")
st.caption(f"Ledoit-Wolf shrinkage {meta['lw_shrinkage']:.3f} | portfolio skew {meta['port_skew']:+.2f}, "
           f"excess kurtosis {meta['port_excess_kurtosis']:+.2f} | historical tail days: "
           + ", ".join(f"{int(round(c * 100))}%: {n}" for c, n in meta["hist_tail_days"].items()))

st.subheader(f"Component risk ({method_label}, Cornish-Fisher)")
fmt = {c: "{:.2%}" for c in report.components.columns if c not in ("Sector",)}
fmt.update({c: "{:.4f}" for c in report.components.columns if c.startswith("MVaR")})
fmt["RiskToWeight_99"] = "{:.2f}x"
st.dataframe(report.components.style.format(fmt, na_rep=""), width="stretch")
st.caption("MVaR: change in VaR per unit of weight. CompVaR: weight x MVaR; components sum to total VaR. "
           "Shares are identical at 95% and 99% because skew and kurtosis are held fixed (Euler approximation for "
           "Cornish-Fisher). RiskToWeight above 1 means a position carries more risk than capital.")

d1, d2 = st.columns(2)
d1.download_button("Download risk summary (CSV)", summary.to_csv().encode(), "risk_summary.csv", "text/csv")
d2.download_button("Download component risk (CSV)", report.components.to_csv().encode(), "component_risk.csv",
                   "text/csv")

# ------------------------------------------------------------------------------------------
# Narrative: risk summary -> investment analysis -> findings -> recommendation (text from analysis.py)
# ------------------------------------------------------------------------------------------
if analysis is not None:
    def md(t: str) -> str:
        return t.replace("$", r"\$")  # Streamlit markdown treats $...$ as LaTeX

    st.divider()
    st.header("Risk summary")
    st.markdown("\n".join(f"- {md(t)}" for t in analysis.summary))

    st.header("Investment analysis")
    st.markdown("\n".join(f"- {md(t)}" for t in analysis.analysis))

    st.header("Findings")
    alert = {"High": (st.error, ":material/error:"), "Medium": (st.warning, ":material/warning:"),
             "Low": (st.info, ":material/info:"), "Info": (st.info, ":material/info:")}
    for f in analysis.findings:
        fn, icon = alert[f.severity]
        fn(f"**{f.severity}: {md(f.title)}.** {md(f.detail)}  \nRule: {md(f.rule)}", icon=icon)

    st.header("Recommendation")
    body, disclaimer = analysis.recommendation[:-1], analysis.recommendation[-1]
    # Indented lines (the tax note) become their own paragraph inside the preceding list item.
    st.markdown("\n".join(("\n" + md(t)) if t.startswith("   ") else md(t) for t in body))
    st.caption(disclaimer)
    st.download_button("Download full report (.md)",
                       ("# Portfolio risk report\n\n" + an.render_text(analysis)).encode(),
                       "risk_report.md", "text/markdown")

    # ---- AI analyst brief ----------------------------------------------------------------
    # Visually hidden but present in the page text, so a browser assistant (Claude, Gemini) reading the
    # page finds it. A copy for the user lives in the expander below.
    brief = an.ai_brief(analysis, report, vc, notional or None, transactions=transactions, positions=positions)
    st.html('<section id="ai-analyst-brief" aria-label="AI analyst brief" style="position:absolute;width:1px;'
            'height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0">'
            f'<h2>AI analyst brief</h2><pre>{html.escape(brief)}</pre></section>')
    with st.expander("AI analyst brief: ask Claude or Gemini for an expert review"):
        st.markdown("**1. In your browser assistant (Claude in Chrome, Gemini in Chrome), say:**")
        st.code(an.AI_STARTER_PROMPT, language=None, wrap_lines=True)
        st.markdown("**2. Or paste the full brief into any chat** (instructions plus all the numbers"
                    + ("; includes the transaction ledger with external transfers reduced to dated "
                       "Cash in / Cash out amounts" if transactions is not None else "")
                    + "; no account number, names or banks):")
        st.download_button("Download brief (.md)", brief.encode(), "ai_analyst_brief.md", "text/markdown")
        st.code(brief, language="markdown", wrap_lines=True, height=320)
