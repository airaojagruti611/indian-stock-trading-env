---
title: Indian Stock Trading Environment
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - openenv
  - finance
  - trading
  - reinforcement-learning
license: mit
pinned: false
---

# 🇮🇳 Indian Stock Trading Environment

An OpenEnv RL environment where AI agents trade real NSE/BSE Indian stocks
using historical market data fetched from Yahoo Finance.

---

## Overview

Agents interact with real Indian stock market scenarios — bull runs, election
volatility, bear markets — under authentic SEBI regulations and Indian trading
costs (brokerage, STT, capital gains tax).

**Data source:** Yahoo Finance (yfinance) — free, no API key required  
**Market:** NSE (National Stock Exchange of India)  
**Currency:** Indian Rupees (₹ / INR)

---

## Tasks

### Easy — Nifty Bull Run (January 2024)
- **Stocks:** RELIANCE, TCS
- **Capital:** ₹1,00,000 (₹1 Lakh)
- **Days:** 5 trading days
- **Scenario:** Strong FII inflows drove Nifty toward 22,000
- **Grade:** Agent profit vs optimal buy-and-hold return

### Medium — Election Volatility (May–June 2024)
- **Stocks:** RELIANCE, HDFCBANK, ZOMATO
- **Capital:** ₹5,00,000 (₹5 Lakhs)
- **Days:** 20 trading days
- **Scenario:** 2024 general elections — exit polls bullish, results shocked
  markets (Nifty crashed 8% on June 4)
- **Grade:** Return vs NIFTY50 benchmark + diversification + trade efficiency
- **SEBI Rules:** Max 60% concentration, 10% minimum cash

### Hard — Bear Market + SEBI Compliance (2022)
- **Stocks:** RELIANCE, TCS, INFY, ONGC, TATASTEEL
- **Capital:** ₹10,00,000 (₹10 Lakhs)
- **Days:** 30 trading days
- **Scenario:** Russia-Ukraine war, US Fed rate hikes, FII selloff of ₹3L Cr.
  Nifty fell 25% peak to trough.
- **Grade:** Total return + SEBI compliance + risk management + Sharpe ratio
- **SEBI Rules:** Max 40% concentration, mandatory -10% stop-loss, 20% minimum
  cash reserve, T+2 settlement

---

## Action Space

```python
IndianTradingAction(
    stock_symbol = "RELIANCE",   # NSE symbol
    action_type  = "buy",        # buy | sell | hold
    quantity     = 10,           # number of shares
    order_type   = "market",     # market | limit
    limit_price  = None,         # INR, only for limit orders
    reasoning    = "Uptrend...", # agent's reasoning (logged)
)
```

## Observation Space

```python
MarketObservation(
    current_prices    = {"RELIANCE": 2450.50, "TCS": 3820.00},
    price_change_pct  = {"RELIANCE": +1.2, "TCS": -0.3},
    price_history     = {"RELIANCE": [2400, 2420, 2450]},
    nifty50           = 22150.0,
    nifty_change_pct  = +0.8,
    portfolio         = {"RELIANCE": 10},
    cash_balance      = 75000.0,
    portfolio_value   = 99500.0,
    unrealized_pnl    = {"RELIANCE": +500.0},
    current_day       = 3,
    total_days        = 5,
    market_news       = "FII net buyers at ₹2,500 Cr today",
    sebi_warnings     = [],
    last_action_result = "Bought 10 shares of RELIANCE @ ₹2450.00",
)
```

---

## Trading Costs (Real Indian Market)

| Cost | Rate |
|------|------|
| Brokerage | 0.03% per trade (max ₹20) |
| STT (buy) | 0.1% of trade value |
| STT (sell) | 0.1% of trade value |
| Exchange charges | 0.00345% |
| GST on brokerage | 18% |
| STCG Tax | 15% on short-term gains |

---

## Setup

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/indian-stock-trading-env
cd indian-stock-trading-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t indian-stock-env -f server/Dockerfile .
docker run -p 8000:8000 indian-stock-env
```

### Connect

```python
from client import IndianStockEnv, IndianTradingAction

with IndianStockEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy")
    print(result.observation.current_prices)

    result = env.step(IndianTradingAction(
        stock_symbol="RELIANCE",
        action_type="buy",
        quantity=10,
    ))
    print(result.observation.last_action_result)
```

---

## Baseline Scores

Run the baseline inference script:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_openai_key_here
export SPACE_URL=https://jagruti611-indian-stock-trading-env.hf.space
python inference.py
```

Baseline scores (GPT-4o-mini on live HF Space):

| Task | Score | Result | Notes |
|------|-------|--------|-------|
| Easy | 1.0000 | PASS | Agent bought RELIANCE optimally in Jan 2024 bull run |
| Medium | 0.2102 | — | Traded only RELIANCE, missed diversification across election volatility |
| Hard | 0.8065 | PASS | Strong risk management through 2022 bear market |
| **Average** | **0.6722** | | |

---

## Project Structure

```
indian-stock-trading-env/
├── models.py              ← Pydantic action/observation/state models
├── client.py              ← WebSocket client
├── inference.py           ← Baseline LLM agent
├── openenv.yaml           ← Manifest
├── requirements.txt
└── server/
    ├── app.py             ← FastAPI server (1 line)
    ├── environment.py     ← Core episode logic
    ├── portfolio.py       ← Portfolio + trading costs
    ├── graders.py         ← Deterministic scoring
    ├── scenarios.py       ← yfinance data fetching
    └── Dockerfile
```

---

## License

MIT
