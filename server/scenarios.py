"""
scenarios.py — Real Indian market scenarios using yfinance historical data.

Three scenarios using actual NSE events:
  easy   → Jan 2024 Bull Run        (5 days,  2 stocks, ₹1L)
  medium → Election Volatility 2024 (20 days, 3 stocks, ₹5L)
  hard   → 2022 Bear Market Crash   (30 days, 5 stocks, ₹10L)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE Symbol Mapping (plain → Yahoo Finance format)
# ---------------------------------------------------------------------------

NSE_TO_YF = {
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "WIPRO":      "WIPRO.NS",
    "ZOMATO":     "ZOMATO.NS",
    "ONGC":       "ONGC.NS",
    "TATASTEEL":  "TATASTEEL.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ADANIENT":   "ADANIENT.NS",
    "LTIM":       "LTIM.NS",
    "NYKAA":      "NYKAA.NS",
    "PAYTM":      "PAYTM.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "AXISBANK":   "AXISBANK.NS",
    "MARUTI":     "MARUTI.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
}

# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "easy": {
        "name":              "Nifty Bull Run — January 2024",
        "description":       (
            "Indian markets rallied strongly in Jan 2024 on FII inflows "
            "and positive global cues. Agent must buy low and sell high "
            "within 5 trading days."
        ),
        "start_date":        "2024-01-02",
        "end_date":          "2024-01-31",
        "stocks":            ["RELIANCE", "TCS"],
        "starting_capital":  100_000,   # ₹1 Lakh
        "max_steps":         5,
        "difficulty":        "easy",
        "benchmark":         "^NSEI",
        "news_events": {
            "2024-01-02": "Markets open strong — FII net buyers at ₹2,500 Cr",
            "2024-01-08": "IT sector rallies on positive US jobs data",
            "2024-01-15": "RELIANCE gains on Jio subscriber growth numbers",
            "2024-01-22": "Nifty hits all-time high above 22,000",
            "2024-01-29": "Markets consolidate ahead of Budget 2024",
        },
        "sebi_rules": {
            "max_single_stock_pct": 1.0,   # No limit for easy task
            "stop_loss_pct":        None,
            "min_cash_reserve_pct": 0.0,
            "t2_settlement":        False,
        },
    },

    "medium": {
        "name":              "Election Volatility — May–June 2024",
        "description":       (
            "India's 2024 general elections caused extreme market volatility. "
            "Exit polls on June 1 predicted a massive BJP win. Actual results "
            "on June 4 shocked markets — Nifty crashed 8% intraday. "
            "Agent must navigate this volatility across 3 stocks."
        ),
        "start_date":        "2024-05-01",
        "end_date":          "2024-06-28",
        "stocks":            ["RELIANCE", "HDFCBANK", "ZOMATO"],
        "starting_capital":  500_000,   # ₹5 Lakhs
        "max_steps":         20,
        "difficulty":        "medium",
        "benchmark":         "^NSEI",
        "news_events": {
            "2024-05-01": "Phase 1 of Lok Sabha elections — markets cautious",
            "2024-05-13": "FIIs selling — rupee weakens to 83.50",
            "2024-05-20": "Exit poll predictions leak — markets speculate",
            "2024-06-01": "Exit polls: BJP to win 350+ seats — Nifty rallies",
            "2024-06-04": "SHOCK: Election results — BJP wins only 240 seats. "
                          "Nifty crashes 8% — circuit breaker triggered!",
            "2024-06-10": "Markets recover — coalition government takes shape",
            "2024-06-18": "RBI holds repo rate at 6.5% — markets stable",
            "2024-06-28": "Nifty recovers to pre-election levels",
        },
        "sebi_rules": {
            "max_single_stock_pct": 0.60,  # Max 60% in one stock
            "stop_loss_pct":        None,
            "min_cash_reserve_pct": 0.10,  # Keep 10% cash
            "t2_settlement":        False,
        },
        "crash_dates": ["2024-06-04"],
    },

    "hard": {
        "name":              "Bear Market + SEBI Compliance — 2022",
        "description":       (
            "2022 was brutal for Indian markets: Russia-Ukraine war, "
            "US Fed rate hikes, FII selloff of ₹3+ lakh crore. "
            "Nifty fell from 18,000 to 15,183 (peak to trough -25%). "
            "Agent must trade 5 stocks under strict SEBI rules, "
            "manage risk, and preserve capital in a bear market."
        ),
        "start_date":        "2022-01-03",
        "end_date":          "2022-08-31",
        "stocks":            ["RELIANCE", "TCS", "INFY", "ONGC", "TATASTEEL"],
        "starting_capital":  1_000_000,  # ₹10 Lakhs
        "max_steps":         30,
        "difficulty":        "hard",
        "benchmark":         "^NSEI",
        "news_events": {
            "2022-01-03": "New Year rally — FII buying in IT and banking",
            "2022-01-17": "FII selling accelerates — US Fed hawkish signals",
            "2022-02-24": "RUSSIA INVADES UKRAINE — Nifty crashes 5% in one day",
            "2022-03-07": "Crude oil hits $130/barrel — India import bill surges",
            "2022-04-06": "RBI emergency rate hike — first in years",
            "2022-05-04": "US Fed hikes 50bps — FII selloff intensifies",
            "2022-05-12": "Nifty breaks below 16,000 — panic selling",
            "2022-06-16": "US Fed hikes 75bps — biggest since 1994",
            "2022-06-17": "Nifty hits 52-week low at 15,183",
            "2022-07-01": "Markets stabilize — ONGC gains on high oil prices",
            "2022-08-15": "Independence Day — Sensex at 60,000 milestone",
            "2022-08-31": "Recovery continues — FII net buyers return",
        },
        "sebi_rules": {
            "max_single_stock_pct": 0.40,  # Max 40% in any single stock
            "stop_loss_pct":        0.10,  # Mandatory stop-loss at -10%
            "min_cash_reserve_pct": 0.20,  # Always keep 20% cash
            "t2_settlement":        True,  # T+2 settlement delay
            "circuit_limit_pct":    0.20,  # 20% upper/lower circuit
        },
        "crash_dates": ["2022-02-24", "2022-05-12", "2022-06-17"],
    },
}

# ---------------------------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------------------------

class MarketDataFetcher:
    """
    Fetches real NSE historical data from Yahoo Finance.
    Caches data to avoid repeated API calls.
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_scenario_data(self, scenario_id: str) -> Dict:
        """
        Fetch all data for a scenario from Yahoo Finance (yfinance).
        Uses real NSE historical price data only.
        """
        scenario = SCENARIOS[scenario_id]
        start    = scenario["start_date"]
        end      = scenario["end_date"]
        stocks   = scenario["stocks"]

        # Fetch stock prices from yfinance
        prices       = self._fetch_prices(stocks, start, end)
        index_data   = self._fetch_index(scenario["benchmark"], start, end)
        trading_days = self._get_trading_days(prices)

        if not trading_days:
            raise RuntimeError(
                f"Failed to fetch market data for scenario '{scenario_id}' "
                f"from Yahoo Finance. Please check your internet connection "
                f"and try again."
            )

        logger.info(f"Using real yfinance data for scenario '{scenario_id}'")

        max_steps = scenario["max_steps"]
        if len(trading_days) > max_steps:
            step         = len(trading_days) // max_steps
            sampled_days = trading_days[::step][:max_steps]
        else:
            sampled_days = trading_days[:max_steps]

        price_matrix = {}
        for stock in stocks:
            if stock in prices:
                price_matrix[stock] = [
                    round(float(prices[stock].loc[day]), 2)
                    if day in prices[stock].index else None
                    for day in sampled_days
                ]

        nifty_series = [
            round(float(index_data.loc[day]), 2)
            if (not index_data.empty and day in index_data.index) else None
            for day in sampled_days
        ]

        return {
            "scenario_id":   scenario_id,
            "scenario_name": scenario["name"],
            "trading_days":  [str(d.date()) for d in sampled_days],
            "price_matrix":  price_matrix,
            "nifty_series":  nifty_series,
            "stocks":        stocks,
            "start_date":    start,
            "end_date":      end,
            "news_events":   scenario.get("news_events", {}),
            "sebi_rules":    scenario.get("sebi_rules", {}),
            "crash_dates":   scenario.get("crash_dates", []),
            "data_source":   "yfinance_live",
        }

    def _fetch_prices(self, stocks: List[str],
                      start: str, end: str) -> Dict[str, pd.Series]:
        prices = {}
        for stock in stocks:
            yf_sym = NSE_TO_YF.get(stock, stock + ".NS")
            cache_key = f"{yf_sym}_{start}_{end}"

            if cache_key in self._cache:
                df = self._cache[cache_key]
            else:
                try:
                    df = yf.download(
                        yf_sym,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=True,
                    )
                    self._cache[cache_key] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {yf_sym}: {e}")
                    continue

            if not df.empty:
                prices[stock] = df["Close"].squeeze()
        return prices

    def _fetch_index(self, symbol: str,
                     start: str, end: str) -> pd.Series:
        cache_key = f"{symbol}_{start}_{end}"
        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                )
                self._cache[cache_key] = df
            except Exception as e:
                logger.error(f"Failed to fetch index {symbol}: {e}")
                return pd.Series(dtype=float)
        return df["Close"].squeeze()

    def _get_trading_days(self,
                           prices: Dict[str, pd.Series]) -> List:
        """Get sorted list of dates where all stocks have data."""
        if not prices:
            return []
        sets = [set(s.dropna().index) for s in prices.values()]
        common = sorted(set.intersection(*sets))
        return common

    def get_news_for_day(self, scenario_id: str,
                          date_str: str) -> str:
        """Get news headline for a specific date."""
        scenario  = SCENARIOS[scenario_id]
        news_map  = scenario.get("news_events", {})
        for news_date, headline in sorted(news_map.items(), reverse=True):
            if news_date <= date_str:
                return headline
        return "Markets trading normally today."


# Singleton fetcher
_fetcher = MarketDataFetcher()

def get_scenario(scenario_id: str) -> Dict:
    """Public API — get full scenario data."""
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. "
            f"Choose from: {list(SCENARIOS.keys())}"
        )
    return _fetcher.fetch_scenario_data(scenario_id)

def get_scenario_config(scenario_id: str) -> Dict:
    """Get scenario config without fetching data."""
    return SCENARIOS[scenario_id]

def list_scenarios() -> List[Dict]:
    """List all available scenarios."""
    return [
        {
            "id":          sid,
            "name":        s["name"],
            "difficulty":  s["difficulty"],
            "stocks":      s["stocks"],
            "days":        s["max_steps"],
            "capital":     s["starting_capital"],
        }
        for sid, s in SCENARIOS.items()
    ]