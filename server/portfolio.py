"""
portfolio.py — Portfolio tracker for Indian stock trading.

Includes real Indian trading costs:
  - Brokerage (Zerodha-style flat fee)
  - Securities Transaction Tax (STT)
  - Exchange transaction charges
  - Short-term capital gains tax (STCG)
  - T+2 settlement tracking
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import date
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indian Trading Cost Constants
# ---------------------------------------------------------------------------

BROKERAGE_PCT       = 0.0003   # 0.03% per trade (Zerodha style)
BROKERAGE_MAX_INR   = 20.0     # Max ₹20 per order
STT_BUY_PCT         = 0.001    # 0.1% on buy (equity delivery)
STT_SELL_PCT        = 0.001    # 0.1% on sell
EXCHANGE_CHARGES    = 0.0000345 # NSE exchange charges
SEBI_CHARGES        = 0.000001  # SEBI turnover charges
GST_ON_BROKERAGE    = 0.18     # 18% GST on brokerage
STCG_TAX_RATE       = 0.15     # 15% Short-term capital gains tax
LTCG_TAX_RATE       = 0.10     # 10% Long-term capital gains tax
LTCG_THRESHOLD_INR  = 100_000  # ₹1 Lakh LTCG exemption


@dataclass
class Trade:
    """Record of a single trade."""
    day:        int
    date:       str
    symbol:     str
    action:     str          # buy | sell
    quantity:   int
    price:      float
    value:      float        # quantity * price
    brokerage:  float
    stt:        float
    other_charges: float
    net_amount: float        # actual cash impact
    pnl:        Optional[float] = None   # realized P&L (sell only)


@dataclass
class Position:
    """Current holding in a stock."""
    symbol:     str
    quantity:   int
    avg_price:  float        # average buy price
    buy_date:   str
    t2_day:     Optional[int] = None  # day shares actually arrive


class Portfolio:
    """
    Tracks cash, holdings, trades, and P&L.
    Enforces Indian trading costs and T+2 settlement.
    """

    def __init__(
        self,
        starting_capital: float,
        enable_t2: bool = False,
        enable_costs: bool = True,
    ):
        self.starting_capital  = starting_capital
        self.cash              = starting_capital
        self.enable_t2         = enable_t2
        self.enable_costs      = enable_costs

        self.positions: Dict[str, Position] = {}
        self.pending_t2: Dict[str, List[Tuple[int, int, float]]] = {}
        # pending_t2[symbol] = [(settle_day, qty, avg_price), ...]

        self.trade_history:    List[Trade]  = []
        self.daily_values:     List[float]  = []
        self._active_violation_keys: set = set()

        self.total_brokerage:  float = 0.0
        self.total_stt:        float = 0.0
        self.total_realized_pnl: float = 0.0
        self.sebi_violations:  int   = 0

    # -----------------------------------------------------------------------
    # Buy
    # -----------------------------------------------------------------------

    def buy(
        self,
        symbol:     str,
        quantity:   int,
        price:      float,
        day:        int,
        date_str:   str,
    ) -> Tuple[bool, str]:
        """
        Execute a buy order.
        Returns (success, message).
        """
        if quantity <= 0:
            return False, "Quantity must be positive."

        gross_value  = quantity * price
        brokerage    = self._calc_brokerage(gross_value)
        stt          = gross_value * STT_BUY_PCT
        other        = gross_value * (EXCHANGE_CHARGES + SEBI_CHARGES)
        gst          = brokerage * GST_ON_BROKERAGE
        total_cost   = gross_value + brokerage + stt + other + gst

        if total_cost > self.cash:
            return False, (
                f"Insufficient cash. Need ₹{total_cost:,.2f}, "
                f"have ₹{self.cash:,.2f}."
            )

        # Deduct cash
        self.cash -= total_cost

        # Update position (T+2 or immediate)
        if self.enable_t2:
            # Shares arrive after T+2
            if symbol not in self.pending_t2:
                self.pending_t2[symbol] = []
            self.pending_t2[symbol].append((day + 2, quantity, price))
        else:
            self._add_position(symbol, quantity, price, date_str, day)

        # Record trade
        trade = Trade(
            day=day, date=date_str, symbol=symbol,
            action="buy", quantity=quantity, price=price,
            value=gross_value, brokerage=brokerage, stt=stt,
            other_charges=other + gst,
            net_amount=-total_cost,
        )
        self.trade_history.append(trade)
        self.total_brokerage += brokerage
        self.total_stt       += stt

        return True, (
            f"Bought {quantity} shares of {symbol} @ ₹{price:,.2f}. "
            f"Cost: ₹{total_cost:,.2f} (incl. ₹{brokerage:.2f} brokerage, "
            f"₹{stt:.2f} STT). Cash remaining: ₹{self.cash:,.2f}."
        )

    # -----------------------------------------------------------------------
    # Sell
    # -----------------------------------------------------------------------

    def sell(
        self,
        symbol:   str,
        quantity: int,
        price:    float,
        day:      int,
        date_str: str,
    ) -> Tuple[bool, str]:
        """
        Execute a sell order.
        Returns (success, message).
        """
        if quantity <= 0:
            return False, "Quantity must be positive."

        position = self.positions.get(symbol)
        if not position or position.quantity < quantity:
            held = position.quantity if position else 0
            return False, (
                f"Cannot sell {quantity} shares of {symbol}. "
                f"You hold {held} shares."
            )

        gross_value  = quantity * price
        brokerage    = self._calc_brokerage(gross_value)
        stt          = gross_value * STT_SELL_PCT
        other        = gross_value * (EXCHANGE_CHARGES + SEBI_CHARGES)
        gst          = brokerage * GST_ON_BROKERAGE

        # Realized P&L
        buy_value    = quantity * position.avg_price
        gross_pnl    = gross_value - buy_value
        tax          = max(0, gross_pnl) * STCG_TAX_RATE
        net_proceeds = gross_value - brokerage - stt - other - gst - tax

        # Update cash and position
        self.cash += net_proceeds
        self.total_realized_pnl += gross_pnl
        position.quantity -= quantity
        if position.quantity == 0:
            del self.positions[symbol]

        trade = Trade(
            day=day, date=date_str, symbol=symbol,
            action="sell", quantity=quantity, price=price,
            value=gross_value, brokerage=brokerage, stt=stt,
            other_charges=other + gst + tax,
            net_amount=net_proceeds,
            pnl=gross_pnl,
        )
        self.trade_history.append(trade)
        self.total_brokerage    += brokerage
        self.total_stt          += stt

        pnl_str = f"+₹{gross_pnl:,.2f}" if gross_pnl >= 0 else f"-₹{abs(gross_pnl):,.2f}"
        return True, (
            f"Sold {quantity} shares of {symbol} @ ₹{price:,.2f}. "
            f"P&L: {pnl_str}. Net proceeds: ₹{net_proceeds:,.2f}."
        )

    # -----------------------------------------------------------------------
    # Portfolio value & metrics
    # -----------------------------------------------------------------------

    def _pending_quantity(self, symbol: str) -> int:
        return sum(qty for _, qty, _ in self.pending_t2.get(symbol, []))

    def _pending_market_value(self, current_prices: Dict[str, float]) -> float:
        value = 0.0
        for sym, pending_lots in self.pending_t2.items():
            px = current_prices.get(sym)
            for _, qty, avg_price in pending_lots:
                ref_price = px if px is not None else avg_price
                value += qty * ref_price
        return value

    def holdings_market_value(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Market value per symbol including both settled and pending T+2 lots.
        This gives the true economic exposure used by scoring/compliance.
        """
        values: Dict[str, float] = {}

        for sym, pos in self.positions.items():
            values[sym] = values.get(sym, 0.0) + (
                pos.quantity * current_prices.get(sym, pos.avg_price)
            )

        for sym, pending_lots in self.pending_t2.items():
            px = current_prices.get(sym)
            for _, qty, avg_price in pending_lots:
                ref_price = px if px is not None else avg_price
                values[sym] = values.get(sym, 0.0) + (qty * ref_price)

        return {sym: round(val, 2) for sym, val in values.items() if val > 0}

    def record_daily_value(self, portfolio_value: float):
        """Append one mark-to-market value for drawdown/Sharpe metrics."""
        self.daily_values.append(round(portfolio_value, 2))

    def total_value(self, current_prices: Dict[str, float]) -> float:
        """Total value = cash + settled holdings + pending T+2 holdings."""
        holdings_value = sum(
            pos.quantity * current_prices.get(sym, pos.avg_price)
            for sym, pos in self.positions.items()
        )
        pending_value = self._pending_market_value(current_prices)
        return round(self.cash + holdings_value + pending_value, 2)

    def unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Unrealized P&L per position."""
        pnl = {}
        for sym, pos in self.positions.items():
            current = current_prices.get(sym, pos.avg_price)
            pnl[sym] = round((current - pos.avg_price) * pos.quantity, 2)
        return pnl

    def return_pct(self, current_prices: Dict[str, float]) -> float:
        """Total return % vs starting capital."""
        current = self.total_value(current_prices)
        return round((current - self.starting_capital) / self.starting_capital * 100, 4)

    def concentration(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Percentage of portfolio in each stock."""
        total = self.total_value(current_prices)
        if total == 0:
            return {}
        conc = {}
        exposure_values = self.holdings_market_value(current_prices)
        for sym, mkt_val in exposure_values.items():
            conc[sym] = round(mkt_val / total, 4)
        conc["CASH"] = round(self.cash / total, 4)
        return conc

    def max_drawdown(self) -> float:
        """Max drawdown % from peak in daily_values history."""
        if len(self.daily_values) < 2:
            return 0.0
        peak = self.daily_values[0]
        max_dd = 0.0
        for v in self.daily_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)

    def get_holdings(self) -> Dict[str, int]:
        """Return {symbol: quantity} for all holdings."""
        return {sym: pos.quantity for sym, pos in self.positions.items()}

    def winning_trades_count(self) -> int:
        sells = [t for t in self.trade_history if t.action == "sell"]
        return sum(1 for t in sells if t.pnl is not None and t.pnl > 0)

    def losing_trades_count(self) -> int:
        sells = [t for t in self.trade_history if t.action == "sell"]
        return sum(1 for t in sells if t.pnl is not None and t.pnl <= 0)

    # -----------------------------------------------------------------------
    # T+2 settlement
    # -----------------------------------------------------------------------

    def process_t2_settlements(self, current_day: int, date_str: str):
        """Settle pending T+2 buys that are now due."""
        for symbol in list(self.pending_t2.keys()):
            remaining = []
            for (settle_day, qty, avg_price) in self.pending_t2[symbol]:
                if current_day >= settle_day:
                    self._add_position(symbol, qty, avg_price, date_str, current_day)
                else:
                    remaining.append((settle_day, qty, avg_price))
            if remaining:
                self.pending_t2[symbol] = remaining
            else:
                del self.pending_t2[symbol]

    # -----------------------------------------------------------------------
    # SEBI rule checks
    # -----------------------------------------------------------------------

    def check_sebi_rules(
        self,
        rules: Dict,
        current_prices: Dict[str, float],
    ) -> List[str]:
        """
        Check SEBI compliance rules.
        Returns list of violation messages.
        """
        violations = []
        current_keys = set()

        max_pct = rules.get("max_single_stock_pct")
        if max_pct:
            conc = self.concentration(current_prices)
            for sym, pct in conc.items():
                if sym != "CASH" and pct > max_pct:
                    key = ("max_single_stock_pct", sym)
                    current_keys.add(key)
                    violations.append(
                        f"SEBI violation: {sym} concentration {pct:.1%} "
                        f"exceeds limit of {max_pct:.1%}."
                    )

        min_cash = rules.get("min_cash_reserve_pct")
        if min_cash:
            total  = self.total_value(current_prices)
            c_pct  = self.cash / total if total > 0 else 1.0
            if c_pct < min_cash:
                key = ("min_cash_reserve_pct",)
                current_keys.add(key)
                violations.append(
                    f"SEBI violation: Cash {c_pct:.1%} below "
                    f"minimum reserve of {min_cash:.1%}."
                )

        stop_loss_pct = rules.get("stop_loss_pct")
        if stop_loss_pct:
            for sym, pos in self.positions.items():
                current = current_prices.get(sym, pos.avg_price)
                loss_pct = (pos.avg_price - current) / pos.avg_price
                if loss_pct >= stop_loss_pct:
                    key = ("stop_loss_pct", sym)
                    current_keys.add(key)
                    violations.append(
                        f"Stop-loss breach: {sym} down {loss_pct:.1%} "
                        f"from buy price. Must sell."
                    )

        # Count only newly introduced breaches; persistent breaches are not re-counted.
        new_violation_events = current_keys - self._active_violation_keys
        self.sebi_violations += len(new_violation_events)
        self._active_violation_keys = current_keys
        return violations

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _add_position(
        self,
        symbol:    str,
        quantity:  int,
        price:     float,
        date_str:  str,
        day:       int,
    ):
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty   = pos.quantity + quantity
            avg_price   = (pos.quantity * pos.avg_price + quantity * price) / total_qty
            pos.quantity  = total_qty
            pos.avg_price = round(avg_price, 2)
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=round(price, 2),
                buy_date=date_str,
                t2_day=day + 2 if self.enable_t2 else day,
            )

    def _calc_brokerage(self, trade_value: float) -> float:
        if not self.enable_costs:
            return 0.0
        return min(trade_value * BROKERAGE_PCT, BROKERAGE_MAX_INR)

    def snapshot(self, current_prices: Dict[str, float]) -> Dict:
        """Full portfolio snapshot for logging/debugging."""
        winning = self.winning_trades_count()
        losing = self.losing_trades_count()
        return {
            "cash":            round(self.cash, 2),
            "holdings":        self.get_holdings(),
            "holdings_market_value": self.holdings_market_value(current_prices),
            "total_value":     self.total_value(current_prices),
            "return_pct":      self.return_pct(current_prices),
            "unrealized_pnl":  self.unrealized_pnl(current_prices),
            "realized_pnl":    round(self.total_realized_pnl, 2),
            "total_trades":    len(self.trade_history),
            "winning_trades":  winning,
            "losing_trades":   losing,
            "closed_trades":   winning + losing,
            "total_brokerage": round(self.total_brokerage, 2),
            "total_stt":       round(self.total_stt, 2),
            "max_drawdown":    self.max_drawdown(),
            "sebi_violations": self.sebi_violations,
        }
