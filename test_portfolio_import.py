"""Tests for portfolio_import.py on synthetic files in the Schwab export layout.

Run with `py -m pytest test_portfolio_import.py` or `py test_portfolio_import.py` (non-zero exit on failure).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import portfolio_import as pi

POSITIONS = '''"Positions for account Individual ...123 as of 4:00 PM ET, 2026/09/30"

"Symbol","Description","Price","Qty (Quantity)","Mkt Val (Market Value)","Cost Basis","% of Acct (% of Account)","Asset Type",
"AAA","ALPHA INC","1,000.00","2","$2,000.00","$1,800.00","50%","Equity",
"BRK/B","BERKSHIRE HATHAWAY CL B","500.00","2","$1,000.00","$900.00","25%","Equity",
"XYZ 12/19/2026 100.00 C","CALL XYZ","5.00","1","$500.00","$400.00","12.5%","Option",
"Cash & Cash Investments","--","--","--","$500.00","--","12.5%","Cash and Money Market",
"Positions Total","","--","--","$4,000.00","$3,100.00","--","--",
'''

TRANSACTIONS = '''"Date","Action","Symbol","Description","Quantity","Price","Fees & Comm","Amount"
"09/20/2026","Sell","AAA","ALPHA INC","1","$1,010.00","$0.01","$1,009.99"
"09/10/2026 as of 09/09/2026","Reinvest Shares","BRK/B","BERKSHIRE","0.5","$480.00","","-$240.00"
"09/05/2026","Journal","AAA","ALPHA INC","-3","","",""
"09/05/2026","Journal","AAA","ALPHA INC","3","","",""
"09/01/2026","Stock Merger","OLD","OLDCO","10","","",""
"08/01/2026","Buy","AAA","ALPHA INC","3","$900.00","","-$2,700.00"
"08/01/2026","Buy","BRK/B","BERKSHIRE","1.5","$450.00","","-$675.00"
"07/01/2026","Qualified Dividend","AAA","ALPHA INC","","","","$1.00"
"Transactions Total","","","","","","","-$2,604.01"
'''


def test_num_parsing() -> None:
    assert pi._num("$1,311.49") == 1311.49
    assert pi._num("-$82.84") == -82.84
    assert pi._num("(12.50)") == -12.5
    assert pi._num("0.28%") == 0.28
    assert np.isnan(pi._num("--")) and np.isnan(pi._num(""))


def test_positions_weights_cash_and_exclusions() -> None:
    p = pi.parse_schwab_positions(POSITIONS.encode())
    assert p.account == "Individual ...123"
    assert p.as_of == pd.Timestamp("2026-09-30")
    assert p.tickers == ["AAA", "BRK-B"]             # slash class share mapped to Yahoo form; option excluded
    assert np.isclose(p.total_value, 4000.0)
    assert np.isclose(p.cash, 500.0)
    assert np.allclose(p.weights, [0.5, 0.25])        # weights are of TOTAL account incl. cash and option
    assert len(p.excluded) == 1 and p.excluded["AssetType"].iloc[0] == "Option"
    assert any("Not modelled" in w for w in p.warnings)


def test_positions_total_mismatch_raises() -> None:
    bad = POSITIONS.replace('"$4,000.00"', '"$4,100.00"')
    try:
        pi.parse_schwab_positions(bad.encode())
    except ValueError as e:
        assert "total row" in str(e)
    else:
        raise AssertionError("dropped/misread rows were not detected")


def test_transactions_signs_dates_and_unhandled() -> None:
    t = pi.parse_schwab_transactions(TRANSACTIONS.encode())
    assert len(t.ledger) == 8                          # footer row dropped
    reinv = t.ledger[t.ledger["Action"] == "Reinvest Shares"].iloc[0]
    assert reinv["Date"] == pd.Timestamp("2026-09-10") and reinv["AsOfDate"] == pd.Timestamp("2026-09-09")
    assert list(t.unhandled["Action"]) == ["Stock Merger"]
    q = t.holdings_as_of()
    assert np.isclose(q["AAA"], 2.0)                   # +3 buy, journal pair nets 0, -1 sell
    assert np.isclose(q["BRK-B"], 2.0)                 # 1.5 + 0.5 reinvest
    assert "OLD" not in q.index                        # unknown action contributes nothing
    q_aug = t.holdings_as_of("2026-08-31")
    assert np.isclose(q_aug["AAA"], 3.0) and np.isclose(q_aug["BRK-B"], 1.5)


def test_reconcile_flags_mismatch() -> None:
    p = pi.parse_schwab_positions(POSITIONS.encode())
    t = pi.parse_schwab_transactions(TRANSACTIONS.encode())
    rec = pi.reconcile(p, t)
    assert rec.loc["AAA", "Status"] == "ok" and rec.loc["BRK-B", "Status"] == "ok"
    t2 = pi.parse_schwab_transactions(TRANSACTIONS.replace('"Sell","AAA","ALPHA INC","1"', '"Sell","AAA","ALPHA INC","0.5"').encode())
    rec2 = pi.reconcile(p, t2)
    assert rec2.loc["AAA", "Status"] == "mismatch" and np.isclose(rec2.loc["AAA", "Difference"], -0.5)
    assert (rec2.loc[rec2.index.str.startswith("XYZ"), "Status"] == "only in positions").all()


def test_cost_basis_check_flags_non_fifo_relief() -> None:
    p = pi.parse_schwab_positions(POSITIONS.encode())
    t = pi.parse_schwab_transactions(TRANSACTIONS.encode())
    chk = pi.cost_basis_check(p, t)
    # AAA: FIFO leaves 2 @ $900 = $1,800 = broker. BRK-B: FIFO $915 vs broker $900 -> different relief.
    assert np.isclose(chk.loc["AAA", "FIFO cost"], 1800.0) and chk.loc["AAA", "FIFO matches"]
    assert np.isclose(chk.loc["BRK-B", "FIFO cost"], 915.0) and not chk.loc["BRK-B", "FIFO matches"]
    assert pi.basis_matches(115.07, 115.02) and not pi.basis_matches(1687.57, 1060.77)


PRIVATE = ["JANE Q PUBLIC", "FIRST EXAMPLE BANK", "ACME PAYROLL", "Tfr"]
CASH_TX = TRANSACTIONS.replace('"Transactions Total"', '''"09/22/2026","MoneyLink Transfer","","Tfr FIRST EXAMPLE BANK, JANE Q PUBLIC","","","","-$150.00"
"09/21/2026 as of 09/20/2026","MoneyLink Deposit","","ACME PAYROLL","","","","$400.00"
"Transactions Total"''')


def test_anonymized_ledger_hides_counterparties() -> None:
    t = pi.parse_schwab_transactions(CASH_TX.encode())
    led = pi.anonymized_ledger(t)
    text = led.to_csv(index=False)
    assert not [w for w in PRIVATE + ["MoneyLink", "BERKSHIRE", "ALPHA INC"] if w in text]   # no descriptions at all
    assert list(led.columns) == ["Date", "Type", "Symbol", "Qty", "Price", "Amount"]
    cash = led[led["Type"].isin(["Cash in", "Cash out"])]
    assert sorted(cash["Amount"]) == [-150.0, 400.0] and (cash["Symbol"] == "").all()
    assert pi.external_cash_summary(led) == {"cash_in": 400.0, "cash_out": -150.0, "net": 250.0, "transfers": 2}
    sell = led[led["Type"] == "Sell"].iloc[0]
    assert sell["Qty"] == -1.0 and sell["Price"] == 1010.0          # signed qty, price per share kept
    assert len(led) == len(t.ledger)                                  # full ledger, nothing dropped


def test_weights_from_quantities() -> None:
    t, w, total = pi.weights_from_quantities(pd.Series({"A": 2.0, "B": 1.0}), pd.Series({"A": 10.0, "B": 30.0}))
    assert t == ["B", "A"] and np.allclose(w, [0.6, 0.4]) and total == 50.0
    try:
        pi.weights_from_quantities(pd.Series({"A": 1.0, "C": 1.0}), pd.Series({"A": 1.0}))
    except ValueError as e:
        assert "C" in str(e)
    else:
        raise AssertionError("missing price not detected")


def test_detect_format() -> None:
    assert pi.detect_format(POSITIONS.encode()) == "positions"
    assert pi.detect_format(TRANSACTIONS.encode()) == "transactions"


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
