"""Importers for Charles Schwab CSV exports.

  * Positions export     ("Positions for account ... as of 12:07 PM ET, 2026/10/01" banner, then a
                           Symbol/Description/Price/.../Asset Type table, a cash row and a total row).
  * Transactions export  (Date/Action/Symbol/Description/Quantity/Price/Fees & Comm/Amount).

Positions give current weights and account value. Transactions rebuild share quantities as of any
date, which is used to reconcile against a positions file or to price holdings on a past date.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

Source = Union[str, Path, bytes, IO[str], IO[bytes]]

# Asset types the return model can price from yfinance. Everything else (options, bonds, ...) is
# reported as excluded and left in the account total, i.e. treated as zero-risk like cash.
MODELLED_ASSET_TYPES = {"Equity", "ETFs & Closed End Funds", "ETF", "Exchange Traded Fund"}

BUY_ACTIONS = {"Buy", "Reinvest Shares", "Buy to Open", "Buy to Close"}
SELL_ACTIONS = {"Sell", "Sell Short", "Sell to Open", "Sell to Close"}
SIGNED_ACTIONS = {"Journal", "Stock Split", "Reverse Split", "Security Transfer", "Internal Transfer"}
QTY_TOL = 1e-6


def _read_text(src: Source) -> str:
    if isinstance(src, bytes):
        return src.decode("utf-8-sig")
    if hasattr(src, "read"):
        data = src.read()  # type: ignore[union-attr]
        return data.decode("utf-8-sig") if isinstance(data, bytes) else data
    path = Path(src)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8-sig")


def _num(s: str | None) -> float:
    """'$1,311.49' -> 1311.49, '-$82.84' -> -82.84, '0.28%' -> 0.28, '--' -> nan."""
    if s is None:
        return np.nan
    s = s.strip()
    if s in ("", "--", "-", "N/A", "n/a"):
        return np.nan
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[,$%()\s]", "", s).lstrip("-+")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def to_yahoo_symbol(symbol: str) -> str:
    """Schwab writes share classes with a slash (BRK/B); Yahoo uses a dash (BRK-B)."""
    return symbol.strip().upper().replace("/", "-").replace(".", "-")


def _short_header(h: str) -> str:
    """'Mkt Val (Market Value)' -> 'Mkt Val'."""
    return h.split(" (")[0].strip()


def detect_format(src: Source) -> str:
    text = _read_text(src)
    head = text[:600]
    if head.lstrip('"').startswith("Positions for account") or '"Mkt Val' in head:
        return "positions"
    if head.startswith('"Date","Action"') or ("Action" in head and "Amount" in head):
        return "transactions"
    raise ValueError("Not a recognised Schwab positions or transactions export")


# ------------------------------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PositionsImport:
    holdings: pd.DataFrame          # index Symbol: Description, Quantity, Price, MarketValue, Weight, AssetType,
                                    # CostBasis, UnrealizedGain (NaN when the export omits them)
    cash: float
    total_value: float              # account value incl. cash and excluded holdings
    as_of: pd.Timestamp | None
    account: str | None
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: tuple[str, ...] = ()

    @property
    def tickers(self) -> list[str]:
        return self.holdings.index.tolist()

    @property
    def weights(self) -> list[float]:
        return self.holdings["Weight"].tolist()


def parse_schwab_positions(src: Source) -> PositionsImport:
    text = _read_text(src)
    rows = list(csv.reader(io.StringIO(text)))
    banner = rows[0][0] if rows and rows[0] else ""
    account = (m.group(1).strip() if (m := re.search(r"account (.+?) as of", banner)) else None)
    as_of = (pd.Timestamp(m.group(1).replace("/", "-"))
             if (m := re.search(r"(\d{4}/\d{2}/\d{2})", banner)) else None)

    try:
        h = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Symbol")
    except StopIteration:
        raise ValueError("Positions file has no 'Symbol' header row") from None
    cols = [_short_header(c) for c in rows[h]]
    idx = {c: i for i, c in enumerate(cols) if c}
    for need in ("Symbol", "Qty", "Mkt Val"):
        if need not in idx:
            raise ValueError(f"Positions file is missing the '{need}' column")

    def cell(r: list[str], name: str) -> str:
        i = idx.get(name)
        return r[i] if i is not None and i < len(r) else ""

    recs, excluded, cash, reported_total = [], [], 0.0, np.nan
    for r in rows[h + 1:]:
        if not r or not r[0].strip():
            continue
        sym = r[0].strip()
        if sym == "Positions Total" or sym.startswith("Account Total"):
            reported_total = _num(cell(r, "Mkt Val"))
            continue
        asset = cell(r, "Asset Type").strip()
        mv = _num(cell(r, "Mkt Val"))
        if sym.startswith("Cash") or asset == "Cash and Money Market":
            cash += 0.0 if np.isnan(mv) else mv
            continue
        rec = {"Symbol": to_yahoo_symbol(sym), "SchwabSymbol": sym, "Description": cell(r, "Description"),
               "Quantity": _num(cell(r, "Qty")), "Price": _num(cell(r, "Price")), "MarketValue": mv,
               "PctOfAcct": _num(cell(r, "% of Acct")) / 100.0, "AssetType": asset,
               "CostBasis": _num(cell(r, "Cost Basis")), "UnrealizedGain": _num(cell(r, "Gain $"))}
        (recs if asset in MODELLED_ASSET_TYPES or not asset else excluded).append(rec)

    if not recs:
        raise ValueError("Positions file contains no equity or ETF holdings")
    hold = pd.DataFrame(recs).set_index("Symbol")
    excl = pd.DataFrame(excluded).set_index("Symbol") if excluded else pd.DataFrame()
    total = float(hold["MarketValue"].sum() + (excl["MarketValue"].sum() if len(excl) else 0.0) + cash)

    warnings: list[str] = []
    if not np.isnan(reported_total):
        # Each market value is rounded to the cent, so allow half a cent per row of drift.
        tol = 0.005 * (len(hold) + len(excl) + 1)
        if abs(total - reported_total) > tol:
            raise ValueError(f"Parsed market values sum to {total:,.2f} but the file's total row says "
                             f"{reported_total:,.2f}; rows were lost or misread")
    hold["Weight"] = hold["MarketValue"] / total
    off = hold[(hold["PctOfAcct"] - hold["Weight"]).abs() > 0.0001]
    if len(off):
        warnings.append(f"Weight differs from Schwab's '% of Acct' by >0.01% for {off.index.tolist()}")
    if len(excl):
        warnings.append(f"Not modelled (treated as zero risk, like cash): {excl.index.tolist()} "
                        f"({excl['MarketValue'].sum() / total:.2%} of account)")
    if hold.index.duplicated().any():
        raise ValueError(f"Duplicate symbols in positions file: {hold.index[hold.index.duplicated()].tolist()}")

    cols_out = ["SchwabSymbol", "Description", "AssetType", "Quantity", "Price", "MarketValue", "Weight",
                "CostBasis", "UnrealizedGain"]
    return PositionsImport(hold[cols_out].sort_values("Weight", ascending=False), cash, total, as_of, account,
                           excl, tuple(warnings))


# ------------------------------------------------------------------------------------------
# Transactions
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransactionsImport:
    ledger: pd.DataFrame        # Date, AsOfDate, Action, Symbol, Description, ShareDelta, Quantity, Price, Fees, Amount
    unhandled: pd.DataFrame     # rows carrying a quantity whose action has no known share effect

    def holdings_as_of(self, date: pd.Timestamp | str | None = None) -> pd.Series:
        """Share quantity per symbol after all trades dated on or before `date` (default: all)."""
        led = self.ledger if date is None else self.ledger[self.ledger["Date"] <= pd.Timestamp(date)]
        q = led.groupby("Symbol")["ShareDelta"].sum()
        q = q[q.abs() > QTY_TOL].round(6).sort_index()
        q.name = "Quantity"
        return q

    @property
    def first_date(self) -> pd.Timestamp:
        return self.ledger["Date"].min()

    @property
    def last_date(self) -> pd.Timestamp:
        return self.ledger["Date"].max()


def parse_schwab_transactions(src: Source) -> TransactionsImport:
    df = pd.read_csv(io.StringIO(_read_text(src)), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for need in ("Date", "Action", "Symbol", "Quantity"):
        if need not in df.columns:
            raise ValueError(f"Transactions file is missing the '{need}' column")

    # "08/27/2026 as of 08/26/2026": first date is the trade/settle date Schwab books it on.
    parts = df["Date"].str.extract(r"^\s*(\d{2}/\d{2}/\d{4})(?:\s+as of\s+(\d{2}/\d{2}/\d{4}))?")
    df["Date"] = pd.to_datetime(parts[0], format="%m/%d/%Y", errors="coerce")
    df["AsOfDate"] = pd.to_datetime(parts[1], format="%m/%d/%Y", errors="coerce").fillna(df["Date"])
    df = df[df["Date"].notna()].copy()  # drops "Transactions Total" style footer rows

    df["Symbol"] = df["Symbol"].map(lambda s: to_yahoo_symbol(s) if s.strip() else "")
    df["Action"] = df["Action"].str.strip()
    for c, src_col in (("Quantity", "Quantity"), ("Price", "Price"), ("Fees", "Fees & Comm"), ("Amount", "Amount")):
        df[c] = df[src_col].map(_num) if src_col in df.columns else np.nan

    qty = df["Quantity"].fillna(0.0)
    sign = np.select([df["Action"].isin(BUY_ACTIONS), df["Action"].isin(SELL_ACTIONS),
                      df["Action"].isin(SIGNED_ACTIONS)], [1.0, -1.0, 0.0], default=np.nan)
    # Buy/Sell quantities are unsigned in the export; Journal/split/transfer rows carry their own sign.
    delta = np.where(sign == 0.0, qty, qty.abs() * sign)
    known = ~np.isnan(sign)
    has_qty = qty.abs() > 0
    df["ShareDelta"] = np.where(known & (df["Symbol"] != ""), delta, 0.0)

    unhandled = df[has_qty & ~known]
    cols = ["Date", "AsOfDate", "Action", "Symbol", "Description", "ShareDelta", "Quantity", "Price", "Fees", "Amount"]
    ledger = df.reindex(columns=cols).sort_values("Date", kind="stable").reset_index(drop=True)
    return TransactionsImport(ledger, unhandled.reindex(columns=cols).reset_index(drop=True))


def fifo_lots(transactions: TransactionsImport, symbol: str,
              as_of: pd.Timestamp | str | None = None) -> tuple[pd.DataFrame, float]:
    """Open tax lots for `symbol` under first-in-first-out. Journal pairs net to zero and are skipped.

    Returns (lots with Acquired/Quantity/Price, unmatched sell quantity). A non-zero unmatched
    quantity means the export starts after some shares were bought, so the oldest lots are unknown.
    """
    led = transactions.ledger
    led = led[(led["Symbol"] == symbol) & led["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)]
    if as_of is not None:
        led = led[led["Date"] <= pd.Timestamp(as_of)]
    lots: list[list] = []
    unmatched = 0.0
    for row in led.itertuples(index=False):
        q = abs(float(row.Quantity))
        if row.Action in BUY_ACTIONS:
            lots.append([row.Date, q, row.Price])
            continue
        while q > QTY_TOL and lots:
            take = min(q, lots[0][1])
            lots[0][1] -= take
            q -= take
            if lots[0][1] <= QTY_TOL:
                lots.pop(0)
        unmatched += q if q > QTY_TOL else 0.0
    return pd.DataFrame(lots, columns=["Acquired", "Quantity", "Price"]), unmatched


# ------------------------------------------------------------------------------------------
# Cross-file helpers
# ------------------------------------------------------------------------------------------
def reconcile(positions: PositionsImport, transactions: TransactionsImport,
              tol: float = 1e-4) -> pd.DataFrame:
    """Compare position quantities with quantities rebuilt from the transaction history."""
    as_of = positions.as_of if positions.as_of is not None else transactions.last_date
    rebuilt = transactions.holdings_as_of(as_of)
    pos = positions.holdings["Quantity"]
    if len(positions.excluded):
        pos = pd.concat([pos, positions.excluded["Quantity"]])
    out = pd.DataFrame({"Positions": pos, "Transactions": rebuilt})
    out["Difference"] = out["Positions"].fillna(0.0) - out["Transactions"].fillna(0.0)
    out["Status"] = np.select(
        [out["Positions"].isna(), out["Transactions"].isna(), out["Difference"].abs() <= tol],
        ["only in transactions", "only in positions", "ok"], default="mismatch")
    out.index.name = "Symbol"
    return out.sort_index()


def anonymized_ledger(transactions: TransactionsImport) -> pd.DataFrame:
    """Full ledger safe to share with an external AI.

    The Description column is dropped on every row: for cash movements it names the counterparty
    (bank, payee, payer). Rows without a security become "Cash in" / "Cash out" by the sign of the
    amount, so only the date and dollar amount of each external transfer remain. Trades keep the
    signed share quantity and the price per share.
    """
    led = transactions.ledger
    cash = led["Symbol"] == ""
    qty = led["ShareDelta"].where(led["ShareDelta"] != 0.0)
    # Actions with a quantity but no known share effect (e.g. a merger) keep their raw quantity.
    qty = qty.fillna(led["Quantity"].where(~cash))
    out = pd.DataFrame({
        "Date": led["Date"].dt.strftime("%Y-%m-%d"),
        "Type": np.where(cash, np.where(led["Amount"].fillna(0.0) >= 0, "Cash in", "Cash out"), led["Action"]),
        "Symbol": led["Symbol"],
        "Qty": qty,
        "Price": led["Price"],
        "Amount": led["Amount"],
    })
    return out.sort_values("Date", kind="stable").reset_index(drop=True)


def external_cash_summary(ledger: pd.DataFrame) -> dict[str, float]:
    """Totals of the anonymized "Cash in" / "Cash out" rows."""
    amt = ledger["Amount"].fillna(0.0)
    cash_in = float(amt[ledger["Type"] == "Cash in"].sum())
    cash_out = float(amt[ledger["Type"] == "Cash out"].sum())
    return {"cash_in": cash_in, "cash_out": cash_out, "net": cash_in + cash_out,
            "transfers": int(ledger["Type"].isin(["Cash in", "Cash out"]).sum())}


def basis_matches(fifo_cost: float, broker_cost: float, tol_abs: float = 0.50, tol_rel: float = 0.0025) -> bool:
    """True when FIFO-rebuilt lot cost agrees with the broker's cost basis (the one tolerance rule)."""
    diff = fifo_cost - broker_cost
    return bool(np.isfinite(diff) and abs(diff) <= max(tol_abs, tol_rel * abs(broker_cost)))


def cost_basis_check(positions: PositionsImport, transactions: TransactionsImport,
                     tol_abs: float = 0.50, tol_rel: float = 0.0025) -> pd.DataFrame:
    """Compare the cost of FIFO-rebuilt open lots with the broker's reported cost basis.

    A mismatch means the account relieves lots by another method (specific lot, tax-lot optimizer,
    average cost), so FIFO lot dates and gains must not be presented as the account's.
    Tolerance covers rounding on tiny reinvestment lots and same-day ordering.
    """
    rows = {}
    for sym in positions.holdings.index:
        lots, unmatched = fifo_lots(transactions, sym, positions.as_of)
        fifo = float((lots["Quantity"] * lots["Price"]).sum()) if len(lots) else np.nan
        broker = float(positions.holdings.loc[sym, "CostBasis"])
        ok = unmatched == 0.0 and basis_matches(fifo, broker, tol_abs, tol_rel)
        rows[sym] = {"FIFO cost": fifo, "Broker cost": broker, "Difference": fifo - broker, "FIFO matches": ok}
    out = pd.DataFrame(rows).T
    out.index.name = "Symbol"
    return out


def weights_from_quantities(quantities: pd.Series, prices: pd.Series,
                            cash: float = 0.0) -> tuple[list[str], list[float], float]:
    """Price share counts and convert to weights of (market value + cash). Returns tickers, weights, total."""
    px = prices.reindex(quantities.index)
    missing = px[px.isna()].index.tolist()
    if missing:
        raise ValueError(f"No price for {missing}")
    mv = quantities * px
    total = float(mv.sum() + cash)
    w = (mv / total).sort_values(ascending=False)
    return w.index.tolist(), w.tolist(), total
