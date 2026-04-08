"""
models.py — Typed Pydantic models for Indian Stock Trading Environment
All actions, observations, and state for NSE/BSE trading agent.
"""

from typing import Dict, List, Optional, Any
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class IndianTradingAction(Action):
    """
    Action taken by the trading agent on NSE/BSE.
    """
    stock_symbol: str = Field(
        ...,
        description="NSE stock symbol e.g. RELIANCE, TCS, INFY"
    )
    action_type: str = Field(
        ...,
        description="One of: buy | sell | hold"
    )
    quantity: int = Field(
        default=0,
        description="Number of shares to buy or sell. 0 for hold."
    )
    order_type: str = Field(
        default="market",
        description="Order type: market | limit"
    )
    limit_price: Optional[float] = Field(
        default=None,
        description="Limit price in INR. Only used when order_type=limit."
    )
    reasoning: str = Field(
        default="",
        description="Agent's reasoning for this action (used for logging)."
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class MarketObservation(Observation):
    """
    What the agent sees at each step.
    done and reward are inherited from Observation base class.
    """
    # Market data
    current_prices: Dict[str, float] = Field(
        default_factory=dict,
        description="Current closing prices in INR for each stock."
    )
    price_history: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Last N days of closing prices per stock."
    )
    price_change_pct: Dict[str, float] = Field(
        default_factory=dict,
        description="Day-over-day percentage change per stock."
    )

    # Index data
    nifty50: float = Field(default=0.0, description="NIFTY 50 index value.")
    sensex: float  = Field(default=0.0, description="BSE SENSEX value.")
    nifty_change_pct: float = Field(default=0.0)

    # Portfolio state
    portfolio: Dict[str, int] = Field(
        default_factory=dict,
        description="Current holdings: {symbol: quantity}"
    )
    cash_balance: float = Field(
        default=0.0,
        description="Available cash in INR."
    )
    portfolio_value: float = Field(
        default=0.0,
        description="Total portfolio value (cash + holdings) in INR."
    )
    unrealized_pnl: Dict[str, float] = Field(
        default_factory=dict,
        description="Unrealized P&L per stock in INR."
    )

    # Episode progress
    current_day: int  = Field(default=1)
    total_days: int   = Field(default=5)
    current_date: str = Field(default="", description="Current trading date (YYYY-MM-DD).")
    task_id: str      = Field(default="easy")
    task_name: str    = Field(default="")

    # Market context
    market_news: str  = Field(default="Markets trading normally.")
    circuit_breakers: Dict[str, str] = Field(
        default_factory=dict,
        description="Stocks hitting upper/lower circuit today."
    )
    market_status: str = Field(
        default="open",
        description="open | closed | pre-open"
    )

    # Feedback
    last_action_result: str = Field(
        default="",
        description="Result of the agent's last action."
    )
    sebi_warnings: List[str] = Field(
        default_factory=list,
        description="Any SEBI rule violations detected."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PortfolioState(State):
    """
    Episode metadata — returned by state() method.
    episode_id and step_count are inherited from State base class.
    """
    task_name: str            = Field(default="")
    task_id: str              = Field(default="easy")
    mode: str                 = Field(default="historical")

    # Capital tracking
    starting_capital: float   = Field(default=100000.0)
    current_value: float      = Field(default=100000.0)
    peak_value: float         = Field(default=100000.0)
    total_return_pct: float   = Field(default=0.0)

    # Trade stats
    total_trades: int         = Field(default=0)
    winning_trades: int       = Field(default=0)
    losing_trades: int        = Field(default=0)
    total_brokerage_paid: float = Field(default=0.0)
    total_stt_paid: float     = Field(default=0.0)

    # Risk tracking
    max_drawdown_pct: float   = Field(default=0.0)
    sebi_violations: int      = Field(default=0)
    circuit_hits: int         = Field(default=0)

    # Scenario info
    scenario_name: str        = Field(default="")
    start_date: str           = Field(default="")
    end_date: str             = Field(default="")
    stocks_traded: List[str]  = Field(default_factory=list)

    # Current score
    current_score: float      = Field(default=0.0)
