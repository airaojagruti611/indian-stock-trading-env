"""
environment.py — Core Indian Stock Trading Environment.

Implements OpenEnv's Environment interface:
  reset()  → start a new trading episode
  step()   → execute one trading action
  state    → return episode metadata

Uses real NSE historical data via yfinance.
"""

import uuid
import logging
from typing import Dict, List, Optional

from openenv.core.env_server import Environment

# Import from sibling modules
from .scenarios   import get_scenario, get_scenario_config, SCENARIOS
from .portfolio   import Portfolio
from .graders     import (
    grade_easy, grade_medium, grade_hard, calc_step_reward
)

# Import models from parent package
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import IndianTradingAction, MarketObservation, PortfolioState

logger = logging.getLogger(__name__)


class IndianStockEnvironment(Environment):
    """
    Indian Stock Trading RL Environment.

    Scenarios:
      easy   → 5-day Bull Run, 2 stocks, ₹1 Lakh
      medium → 20-day Election Volatility, 3 stocks, ₹5 Lakhs
      hard   → 30-day Bear Market + SEBI Rules, 5 stocks, ₹10 Lakhs
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._reset_internals()

    def _reset_internals(self):
        """Clear all episode state."""
        self._scenario_id:    str             = "easy"
        self._scenario_data:  Dict            = {}
        self._portfolio:      Optional[Portfolio] = None
        self._current_step:   int             = 0
        self._done:           bool            = False
        self._episode_id:     str             = ""
        self._daily_values:   List[float]     = []
        self._crash_hit:      List[bool]      = []
        self._state:          PortfolioState  = PortfolioState()
        self._prev_value:     float           = 0.0

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(
        self,
        task_id:    str = "easy",
        episode_id: str = None,
        seed:       int = None,
        **kwargs,
    ) -> MarketObservation:
        """
        Start a new trading episode.

        Args:
            task_id: "easy" | "medium" | "hard"
        """
        self._reset_internals()

        task_id = task_id if task_id in SCENARIOS else "easy"
        self._scenario_id = task_id
        self._episode_id  = episode_id or str(uuid.uuid4())

        # Fetch real market data
        logger.info(f"Fetching scenario data for task_id={task_id}...")
        self._scenario_data = get_scenario(task_id)

        config = get_scenario_config(task_id)
        starting_capital = config["starting_capital"]
        sebi_rules = config.get("sebi_rules", {})

        # Initialize portfolio with Indian trading costs
        self._portfolio = Portfolio(
            starting_capital=starting_capital,
            enable_t2=sebi_rules.get("t2_settlement", False),
            enable_costs=True,
        )

        self._current_step = 0
        self._done         = False
        self._daily_values = [starting_capital]
        self._crash_hit    = []
        self._prev_value   = starting_capital

        # Initialize episode state
        self._state = PortfolioState(
            episode_id=self._episode_id,
            step_count=0,
            task_name=self._scenario_data["scenario_name"],
            task_id=task_id,
            mode="historical",
            starting_capital=starting_capital,
            current_value=starting_capital,
            peak_value=starting_capital,
            total_return_pct=0.0,
            scenario_name=self._scenario_data["scenario_name"],
            start_date=self._scenario_data["start_date"],
            end_date=self._scenario_data["end_date"],
            stocks_traded=self._scenario_data["stocks"],
        )

        return self._build_observation(
            reward=None,
            done=False,
            action_result=f"Episode started. Trade {config['stocks']} over "
                          f"{config['max_steps']} days with "
                          f"₹{starting_capital:,.0f} starting capital.",
        )

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(
        self,
        action: IndianTradingAction,
        timeout_s: float = None,
        **kwargs,
    ) -> MarketObservation:
        """
        Execute one trading action.

        The agent can buy, sell, or hold any stock.
        Advances the episode by one trading day.
        """
        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                action_result="Episode already finished.",
            )

        if self._portfolio is None:
            return self._build_observation(
                reward=0.0,
                done=True,
                action_result="Environment not initialized. Call reset() first.",
            )

        self._current_step += 1
        self._state.step_count = self._current_step

        # Get current day's prices
        current_prices = self._get_prices_for_day(self._current_step - 1)
        current_date   = self._get_date_for_step(self._current_step - 1)

        # Process T+2 settlements
        config = get_scenario_config(self._scenario_id)
        sebi_rules = config.get("sebi_rules", {})
        if sebi_rules.get("t2_settlement"):
            self._portfolio.process_t2_settlements(self._current_step, current_date)

        # Execute the action
        action_result = self._execute_action(
            action, current_prices, current_date
        )

        # Check SEBI rules
        sebi_warnings = self._portfolio.check_sebi_rules(sebi_rules, current_prices)

        # Calculate portfolio value
        portfolio_value = self._portfolio.total_value(current_prices)
        self._daily_values.append(portfolio_value)

        # Track crash behavior (was agent hedged on crash days?)
        crash_dates = self._scenario_data.get("crash_dates", [])
        if current_date in crash_dates:
            cash_pct = self._portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0
            self._crash_hit.append(cash_pct >= 0.30)  # Had 30%+ cash = hedged

        # Step reward
        daily_pnl   = portfolio_value - self._prev_value
        step_reward = calc_step_reward(
            daily_pnl=daily_pnl,
            starting_capital=config["starting_capital"],
            sebi_violations_now=len(sebi_warnings),
            n_holdings=len(self._portfolio.get_holdings()),
            total_trades=len(self._portfolio.trade_history),
            action_type=action.action_type,
        )
        self._prev_value = portfolio_value

        # Update state
        return_pct = self._portfolio.return_pct(current_prices)
        self._state.current_value    = portfolio_value
        self._state.peak_value       = max(self._state.peak_value, portfolio_value)
        self._state.total_return_pct = return_pct
        self._state.total_trades     = len(self._portfolio.trade_history)
        self._state.winning_trades   = self._portfolio.winning_trades_count()
        self._state.losing_trades    = self._portfolio.losing_trades_count()
        self._state.sebi_violations  = self._portfolio.sebi_violations
        self._state.max_drawdown_pct = self._portfolio.max_drawdown()

        # Check if episode is done
        max_steps = config["max_steps"]
        done = (self._current_step >= max_steps)

        if done:
            step_reward = self._calc_final_reward()
            self._done  = True
            action_result += f" | Episode complete. Final return: {return_pct:.2f}%."

        self._state.current_score = step_reward

        return self._build_observation(
            reward=step_reward,
            done=done,
            action_result=action_result,
            sebi_warnings=sebi_warnings,
        )

    # -----------------------------------------------------------------------
    # state property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> PortfolioState:
        return self._state

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _execute_action(
        self,
        action:         IndianTradingAction,
        current_prices: Dict[str, float],
        date_str:       str,
    ) -> str:
        """Execute buy/sell/hold and return result message."""
        symbol      = action.stock_symbol.upper().strip()
        action_type = action.action_type.lower().strip()
        quantity    = max(0, action.quantity)

        # Validate symbol
        available_stocks = self._scenario_data.get("stocks", [])
        if symbol not in available_stocks and action_type != "hold":
            return (
                f"Invalid symbol '{symbol}'. "
                f"Available stocks: {available_stocks}."
            )

        if action_type == "hold" or quantity == 0:
            return f"Holding position. Portfolio value: ₹{self._portfolio.total_value(current_prices):,.2f}."

        price = current_prices.get(symbol)
        if price is None:
            return f"No price data for {symbol} today. Action skipped."

        if action_type == "buy":
            success, msg = self._portfolio.buy(
                symbol=symbol,
                quantity=quantity,
                price=price,
                day=self._current_step,
                date_str=date_str,
            )
            return msg

        elif action_type == "sell":
            success, msg = self._portfolio.sell(
                symbol=symbol,
                quantity=quantity,
                price=price,
                day=self._current_step,
                date_str=date_str,
            )
            return msg

        return f"Unknown action type: '{action_type}'. Use buy/sell/hold."

    def _get_prices_for_day(self, step_idx: int) -> Dict[str, float]:
        """Get all stock prices for a given step index."""
        price_matrix = self._scenario_data.get("price_matrix", {})
        prices = {}
        for symbol, series in price_matrix.items():
            if step_idx < len(series) and series[step_idx] is not None:
                prices[symbol] = series[step_idx]
        return prices

    def _get_date_for_step(self, step_idx: int) -> str:
        """Get the date string for a given step index."""
        days = self._scenario_data.get("trading_days", [])
        if step_idx < len(days):
            return str(days[step_idx])
        return ""

    def _get_price_history(self, symbol: str, up_to_step: int) -> List[float]:
        """Get price history for a stock up to current step."""
        price_matrix = self._scenario_data.get("price_matrix", {})
        series = price_matrix.get(symbol, [])
        return [p for p in series[:up_to_step + 1] if p is not None]

    def _get_nifty_for_step(self, step_idx: int) -> float:
        """Get NIFTY50 value for current step."""
        nifty_series = self._scenario_data.get("nifty_series", [])
        if step_idx < len(nifty_series) and nifty_series[step_idx] is not None:
            return nifty_series[step_idx]
        return 0.0

    def _calc_final_reward(self) -> float:
        """Calculate final episode reward based on task."""
        config      = get_scenario_config(self._scenario_id)
        prices      = self._get_prices_for_day(self._current_step - 1)
        snapshot    = self._portfolio.snapshot(prices)
        price_matrix = self._scenario_data.get("price_matrix", {})
        nifty_series = self._scenario_data.get("nifty_series", [])

        if self._scenario_id == "easy":
            return grade_easy(
                portfolio_snapshot=snapshot,
                price_matrix=price_matrix,
                starting_capital=config["starting_capital"],
            )

        elif self._scenario_id == "medium":
            return grade_medium(
                portfolio_snapshot=snapshot,
                price_matrix=price_matrix,
                nifty_series=nifty_series,
                starting_capital=config["starting_capital"],
                sebi_violations=snapshot.get("sebi_violations", 0),
            )

        elif self._scenario_id == "hard":
            return grade_hard(
                portfolio_snapshot=snapshot,
                price_matrix=price_matrix,
                nifty_series=nifty_series,
                starting_capital=config["starting_capital"],
                sebi_violations=snapshot.get("sebi_violations", 0),
                daily_values=self._daily_values,
                crash_dates_hit=self._crash_hit,
            )

        return 0.0

    def _build_observation(
        self,
        reward:         Optional[float],
        done:           bool,
        action_result:  str = "",
        sebi_warnings:  List[str] = None,
    ) -> MarketObservation:
        """Construct MarketObservation from current state."""
        step_idx     = max(0, self._current_step - 1)
        prices       = self._get_prices_for_day(step_idx)
        date_str     = self._get_date_for_step(step_idx)
        nifty_val    = self._get_nifty_for_step(step_idx)

        # Price changes vs previous day
        prev_prices  = self._get_prices_for_day(max(0, step_idx - 1))
        change_pct   = {}
        for sym, p in prices.items():
            prev = prev_prices.get(sym, p)
            change_pct[sym] = round((p - prev) / prev * 100, 2) if prev else 0.0

        # Price history (last 5 days max)
        history = {}
        stocks  = self._scenario_data.get("stocks", [])
        for sym in stocks:
            history[sym] = self._get_price_history(sym, step_idx)[-5:]

        # Portfolio info
        portfolio_value = (
            self._portfolio.total_value(prices) if self._portfolio else 0.0
        )
        holdings = self._portfolio.get_holdings() if self._portfolio else {}
        cash     = self._portfolio.cash if self._portfolio else 0.0
        u_pnl    = self._portfolio.unrealized_pnl(prices) if self._portfolio else {}

        # News
        news_events = self._scenario_data.get("news_events", {})
        news = "Markets trading normally today."
        for news_date, headline in sorted(news_events.items(), reverse=True):
            if news_date <= date_str:
                news = headline
                break

        # NIFTY change
        prev_nifty   = self._get_nifty_for_step(max(0, step_idx - 1))
        nifty_change = round((nifty_val - prev_nifty) / prev_nifty * 100, 2) if prev_nifty else 0.0

        config = get_scenario_config(self._scenario_id) if self._scenario_id else {}

        return MarketObservation(
            done=done,
            reward=reward,
            current_prices=prices,
            price_history=history,
            price_change_pct=change_pct,
            nifty50=nifty_val,
            sensex=0.0,
            nifty_change_pct=nifty_change,
            portfolio=holdings,
            cash_balance=round(cash, 2),
            portfolio_value=round(portfolio_value, 2),
            unrealized_pnl=u_pnl,
            current_day=self._current_step,
            total_days=config.get("max_steps", 5),
            task_id=self._scenario_id,
            task_name=self._scenario_data.get("scenario_name", ""),
            market_news=news,
            circuit_breakers={},
            market_status="open",
            last_action_result=action_result,
            sebi_warnings=sebi_warnings or [],
        )
