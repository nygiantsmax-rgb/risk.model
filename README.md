# Portfolio risk model

Equity portfolio risk dashboard: Ledoit-Wolf and EWMA covariance, Cornish-Fisher and historical
VaR / expected shortfall, component (Euler) risk per position, benchmark comparison, long-term
projections, rebalancing scenarios and a Markowitz efficient frontier.

The app starts blank. Upload a Schwab positions CSV (a transactions CSV is optional), or type
tickers and weights in the sidebar. Prices come from Yahoo Finance. Uploaded files are processed
in memory for the session and are not saved.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Command-line report:

```bash
python risk_model.py --tickers AAPL,MSFT,NVDA --weights 40,35,25
python risk_model.py --positions path/to/Positions.csv --transactions path/to/Transactions.csv
```

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with GitHub.
2. **Create app** -> pick this repository, branch `main`, main file `dashboard.py`.
3. Under **Advanced settings**, choose Python 3.13, then **Deploy**.

The app gets an `https://<name>.streamlit.app` address that can be embedded in a website.

## Tests

```bash
python test_risk_model.py && python test_portfolio_import.py && python test_analysis.py
```

Analysis output is hypothetical and not investment advice.
