"""
client.py — WebSocket client for Indian Stock Trading Environment.

Usage:
    from indian_stock_trading_env import IndianStockEnv, IndianTradingAction

    with IndianStockEnv(base_url="https://YOUR-SPACE.hf.space").sync() as env:
        result = env.reset(task_id="easy")
        print(result.observation.current_prices)

        result = env.step(IndianTradingAction(
            stock_symbol="RELIANCE",
            action_type="buy",
            quantity=10,
            reasoning="Strong uptrend on high volume",
        ))
        print(result.observation.last_action_result)
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import IndianTradingAction, MarketObservation, PortfolioState


class IndianStockEnv(EnvClient[IndianTradingAction, MarketObservation, PortfolioState]):
    """
    Client for the Indian Stock Trading Environment.
    Connects via WebSocket to the hosted server.
    """

    def _step_payload(self, action: IndianTradingAction) -> dict:
        """Convert action to wire format."""
        return {
            "stock_symbol": action.stock_symbol,
            "action_type":  action.action_type,
            "quantity":     action.quantity,
            "order_type":   action.order_type,
            "limit_price":  action.limit_price,
            "reasoning":    action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse server response into typed StepResult."""
        obs_data = payload.get("observation", {})

        observation = MarketObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),

            current_prices=obs_data.get("current_prices", {}),
            price_history=obs_data.get("price_history", {}),
            price_change_pct=obs_data.get("price_change_pct", {}),

            nifty50=obs_data.get("nifty50", 0.0),
            sensex=obs_data.get("sensex", 0.0),
            nifty_change_pct=obs_data.get("nifty_change_pct", 0.0),

            portfolio=obs_data.get("portfolio", {}),
            cash_balance=obs_data.get("cash_balance", 0.0),
            portfolio_value=obs_data.get("portfolio_value", 0.0),
            unrealized_pnl=obs_data.get("unrealized_pnl", {}),

            current_day=obs_data.get("current_day", 1),
            total_days=obs_data.get("total_days", 5),
            current_date=obs_data.get("current_date", ""),
            task_id=obs_data.get("task_id", "easy"),
            task_name=obs_data.get("task_name", ""),

            market_news=obs_data.get("market_news", ""),
            circuit_breakers=obs_data.get("circuit_breakers", {}),
            market_status=obs_data.get("market_status", "open"),

            last_action_result=obs_data.get("last_action_result", ""),
            sebi_warnings=obs_data.get("sebi_warnings", []),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> PortfolioState:
        """Parse server state response into typed PortfolioState."""
        return PortfolioState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            task_id=payload.get("task_id", "easy"),
            mode=payload.get("mode", "historical"),
            starting_capital=payload.get("starting_capital", 0.0),
            current_value=payload.get("current_value", 0.0),
            peak_value=payload.get("peak_value", 0.0),
            total_return_pct=payload.get("total_return_pct", 0.0),
            total_trades=payload.get("total_trades", 0),
            winning_trades=payload.get("winning_trades", 0),
            losing_trades=payload.get("losing_trades", 0),
            total_brokerage_paid=payload.get("total_brokerage_paid", 0.0),
            total_stt_paid=payload.get("total_stt_paid", 0.0),
            max_drawdown_pct=payload.get("max_drawdown_pct", 0.0),
            sebi_violations=payload.get("sebi_violations", 0),
            circuit_hits=payload.get("circuit_hits", 0),
            scenario_name=payload.get("scenario_name", ""),
            start_date=payload.get("start_date", ""),
            end_date=payload.get("end_date", ""),
            stocks_traded=payload.get("stocks_traded", []),
            current_score=payload.get("current_score", 0.0),
        )
