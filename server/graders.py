"""
graders.py — Deterministic scoring functions for all 3 tasks.

All graders return a float in (0.0, 1.0) exclusive.
Scores are deterministic — same inputs always produce same score.

Easy   → profit vs optimal profit
Medium → return vs benchmark + diversification + trade efficiency
Hard   → return + SEBI compliance + risk management + crash behavior
"""

from typing import Dict, List, Optional
import math

_EPS = 1e-4


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1)."""
    return round(min(1.0 - _EPS, max(_EPS, score)), 4)


# ---------------------------------------------------------------------------
# Easy Task Grader — Bull Run Profit
# ---------------------------------------------------------------------------

def grade_easy(
    portfolio_snapshot: Dict,
    price_matrix:       Dict[str, List[float]],
    starting_capital:   float,
) -> float:
    """
    Grade the easy task: how close to optimal profit?

    Optimal strategy = buy everything on day 1, sell on last day.
    Score = agent_return / optimal_return (capped at 1.0)

    Partial credit:
    - Made any profit at all       → minimum 0.2
    - Broke even                   → 0.1
    - Lost money                   → 0.0 to 0.1
    """
    agent_return_pct = portfolio_snapshot.get("return_pct", 0.0)
    optimal_return   = _calc_optimal_return(price_matrix, starting_capital)

    if optimal_return <= 0:
        # Degenerate scenario — just reward not losing money
        if agent_return_pct >= 0:
            return _clamp(0.5)
        return _clamp(0.5 + agent_return_pct / 100)

    if agent_return_pct <= 0:
        # Lost money — small partial credit for trying
        return _clamp(0.05 + agent_return_pct / optimal_return * 0.1)

    score = agent_return_pct / optimal_return
    return _clamp(score)


# ---------------------------------------------------------------------------
# Medium Task Grader — Election Volatility
# ---------------------------------------------------------------------------

def grade_medium(
    portfolio_snapshot:  Dict,
    price_matrix:        Dict[str, List[float]],
    nifty_series:        List[float],
    starting_capital:    float,
    sebi_violations:     int,
) -> float:
    """
    Grade the medium task: 3 weighted components.

    1. Return vs NIFTY benchmark   (50%)
    2. Diversification score       (30%)
    3. Trade efficiency            (20%)
    Penalty: -0.05 per SEBI violation
    """
    # 1. Return vs benchmark
    agent_return   = portfolio_snapshot.get("return_pct", 0.0) / 100
    nifty_return   = _calc_index_return(nifty_series)
    return_score   = _relative_return_score(agent_return, nifty_return)

    # 2. Diversification
    holdings       = portfolio_snapshot.get("holdings", {})
    total_value    = portfolio_snapshot.get("total_value", starting_capital)
    holding_values = portfolio_snapshot.get("holdings_market_value", {})
    div_score      = _diversification_score(
        holdings=holdings,
        total_value=total_value,
        n_stocks=3,
        holdings_market_value=holding_values,
    )

    # 3. Trade efficiency
    total_trades   = portfolio_snapshot.get("total_trades", 0)
    winning        = portfolio_snapshot.get("winning_trades", 0)
    closed_trades  = portfolio_snapshot.get("closed_trades")
    if closed_trades is None:
        closed_trades = winning + portfolio_snapshot.get("losing_trades", 0)
    efficiency     = _trade_efficiency_score(
        total_trades=total_trades,
        winning_trades=winning,
        closed_trades=closed_trades,
    )

    raw_score = (
        0.50 * return_score  +
        0.30 * div_score     +
        0.20 * efficiency
    )

    # SEBI violation penalty
    penalty = min(0.30, sebi_violations * 0.05)
    final   = raw_score - penalty

    return _clamp(final)


# ---------------------------------------------------------------------------
# Hard Task Grader — Bear Market + SEBI Compliance
# ---------------------------------------------------------------------------

def grade_hard(
    portfolio_snapshot:  Dict,
    price_matrix:        Dict[str, List[float]],
    nifty_series:        List[float],
    starting_capital:    float,
    sebi_violations:     int,
    daily_values:        List[float],
    crash_dates_hit:     List[bool],
) -> float:
    """
    Grade the hard task: 5 weighted components.

    1. Total return vs benchmark   (35%)
    2. SEBI compliance             (25%)
    3. Risk management (drawdown)  (20%)
    4. Sharpe ratio                (10%)
    5. Crash behavior              (10%)
    """
    agent_return  = portfolio_snapshot.get("return_pct", 0.0) / 100
    nifty_return  = _calc_index_return(nifty_series)

    # 1. Return score
    return_score  = _relative_return_score(agent_return, nifty_return)

    # 2. SEBI compliance score
    compliance    = _compliance_score(sebi_violations)

    # 3. Risk management (max drawdown)
    max_dd        = portfolio_snapshot.get("max_drawdown", 0.0)
    risk_score    = _drawdown_score(max_dd)

    # 4. Sharpe ratio approximation
    sharpe        = _sharpe_score(daily_values, starting_capital)

    # 5. Crash behavior (did agent reduce risk before crashes?)
    crash_score   = _crash_behavior_score(crash_dates_hit)

    raw_score = (
        0.35 * return_score  +
        0.25 * compliance    +
        0.20 * risk_score    +
        0.10 * sharpe        +
        0.10 * crash_score
    )

    return _clamp(raw_score)


# ---------------------------------------------------------------------------
# Step-level reward (called every step during episode)
# ---------------------------------------------------------------------------

def calc_step_reward(
    daily_pnl:           float,
    starting_capital:    float,
    sebi_violations_now: int,
    n_holdings:          int,
    total_trades:        int,
    action_type:         str,
) -> float:
    """
    Dense reward signal focused on real value creation.
    Returns small reward in [-0.1, +0.1].

    Design:
    - No built-in positive baseline.
    - Primary driver is daily portfolio improvement.
    - Compliance and overtrading only apply penalties.
    - Diversification gives only a small bonus, and only when P&L is positive.
    """
    pnl_pct = daily_pnl / starting_capital if starting_capital else 0.0
    pnl_score = max(-1.0, min(1.0, pnl_pct * 120.0))

    # Reward is mostly portfolio change.
    raw = 0.85 * pnl_score

    # Small diversification upside only when value actually increased.
    if pnl_pct > 0 and n_holdings >= 2:
        raw += min(0.15, (n_holdings - 1) * 0.05)

    # Penalty for rule breaches in current step.
    raw -= min(1.0, sebi_violations_now * 0.35)

    # Action-aware overtrading penalty.
    atype = (action_type or "hold").lower().strip()
    if atype in {"buy", "sell"}:
        raw -= min(0.20, max(0, total_trades - 12) * 0.01)

    # Scale to small range so step rewards don't dominate final score.
    return round(max(-0.1, min(0.1, raw * 0.1)), 6)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_optimal_return(
    price_matrix:     Dict[str, List[float]],
    starting_capital: float,
) -> float:
    """
    Calculate the best possible return:
    Buy everything on day 0, sell on last day.
    """
    if not price_matrix:
        return 0.0

    total_return = 0.0
    n_stocks = len(price_matrix)
    capital_per_stock = starting_capital / n_stocks

    for symbol, prices in price_matrix.items():
        clean = [p for p in prices if p is not None and p > 0]
        if len(clean) < 2:
            continue
        shares     = int(capital_per_stock // clean[0])
        buy_cost   = shares * clean[0]
        sell_value = shares * clean[-1]
        total_return += (sell_value - buy_cost)

    return total_return / starting_capital * 100 if starting_capital else 0


def _calc_index_return(nifty_series: List[float]) -> float:
    """Calculate NIFTY return % over the series."""
    clean = [p for p in nifty_series if p is not None and p > 0]
    if len(clean) < 2:
        return 0.0
    return (clean[-1] - clean[0]) / clean[0]


def _relative_return_score(
    agent_return:  float,
    index_return:  float,
) -> float:
    """
    Score based on agent return vs index.
    Beat index by 5%+ → 1.0
    Match index       → 0.6
    Zero return       → 0.3
    Lost money        → 0.0–0.2
    """
    if agent_return >= index_return + 0.05:
        return 1.0
    elif agent_return >= index_return:
        # Beat index — linearly from 0.6 to 1.0
        excess = (agent_return - index_return) / 0.05
        return round(0.6 + 0.4 * excess, 4)
    elif agent_return >= 0:
        # Positive but below index
        ratio = agent_return / (index_return + 1e-6)
        return round(0.3 + 0.3 * min(1.0, ratio), 4)
    else:
        # Lost money
        return round(max(0.0, 0.3 + agent_return * 3), 4)


def _diversification_score(
    holdings:    Dict[str, int],
    total_value: float,
    n_stocks:    int,
    holdings_market_value: Optional[Dict[str, float]] = None,
) -> float:
    """
    Score based on how well diversified the portfolio is.
    Holding all n_stocks equally → 1.0
    All in one stock             → 0.3
    No holdings                  → 0.0
    """
    if not holdings:
        return 0.0
    n_held = len(holdings)
    coverage = min(1.0, n_held / max(1, n_stocks))

    # Herfindahl-Hirschman concentration using actual position values when available.
    values = {}
    if holdings_market_value:
        values = {sym: max(0.0, float(v)) for sym, v in holdings_market_value.items()}
    elif holdings:
        values = {sym: float(max(0, qty)) for sym, qty in holdings.items()}

    value_sum = sum(values.values())
    if total_value <= 0 or value_sum <= 0:
        div_quality = 0.0
    else:
        weights = [v / value_sum for v in values.values() if v > 0]
        hhi = sum(w ** 2 for w in weights)
        # Normalize HHI between ideal diversification and full concentration.
        ideal_hhi = 1.0 / max(1, min(n_stocks, len(weights)))
        if 1.0 - ideal_hhi <= 1e-9:
            div_quality = 1.0
        else:
            concentration = (hhi - ideal_hhi) / (1.0 - ideal_hhi)
            div_quality = max(0.0, min(1.0, 1.0 - concentration))

    return round(0.6 * coverage + 0.4 * div_quality, 4)


def _trade_efficiency_score(
    total_trades: int,
    winning_trades: int,
    closed_trades: Optional[int] = None,
) -> float:
    """
    Score based on trade win rate and avoiding overtrading.
    """
    if total_trades == 0:
        return 0.2  # Did nothing — small penalty

    closed = closed_trades if closed_trades is not None else total_trades
    if closed <= 0:
        win_rate = 0.2
    else:
        win_rate = min(1.0, max(0.0, winning_trades / closed))
    overtrading_ok = min(1.0, max(0.0, 1.0 - total_trades / 20))

    return round(0.75 * win_rate + 0.25 * overtrading_ok, 4)


def _compliance_score(sebi_violations: int) -> float:
    """Score for SEBI compliance. Zero violations → 1.0."""
    if sebi_violations == 0:
        return 1.0
    return round(max(0.0, 1.0 - sebi_violations * 0.15), 4)


def _drawdown_score(max_drawdown_pct: float) -> float:
    """
    Score based on max drawdown.
    <5% drawdown  → 1.0
    <15% drawdown → 0.7
    <25% drawdown → 0.4
    >25% drawdown → 0.0–0.3
    """
    if max_drawdown_pct < 5:
        return 1.0
    elif max_drawdown_pct < 15:
        return round(1.0 - (max_drawdown_pct - 5) / 10 * 0.3, 4)
    elif max_drawdown_pct < 25:
        return round(0.7 - (max_drawdown_pct - 15) / 10 * 0.3, 4)
    else:
        return round(max(0.0, 0.4 - (max_drawdown_pct - 25) / 50), 4)


def _sharpe_score(daily_values: List[float], starting_capital: float) -> float:
    """
    Approximate Sharpe ratio score from daily portfolio values.
    Higher Sharpe → higher score.
    """
    if len(daily_values) < 3:
        return 0.5

    returns = []
    for i in range(1, len(daily_values)):
        if daily_values[i - 1] > 0:
            r = (daily_values[i] - daily_values[i - 1]) / daily_values[i - 1]
            returns.append(r)

    if not returns:
        return 0.5

    mean_r  = sum(returns) / len(returns)
    std_r   = math.sqrt(
        sum((r - mean_r) ** 2 for r in returns) / len(returns)
    ) if len(returns) > 1 else 0.01

    sharpe  = mean_r / std_r if std_r > 0 else 0.0

    # Map Sharpe to [0, 1]
    # Sharpe > 2 → 1.0, Sharpe = 0 → 0.5, Sharpe < -1 → 0.0
    if sharpe >= 2.0:
        return 1.0
    elif sharpe >= 0:
        return round(0.5 + sharpe / 2.0 * 0.5, 4)
    else:
        return round(max(0.0, 0.5 + sharpe * 0.5), 4)


def _crash_behavior_score(crash_dates_hit: List[bool]) -> float:
    """
    Score based on behavior during market crashes.
    crash_dates_hit[i] = True if agent was properly hedged on crash day i.
    """
    if not crash_dates_hit:
        return 0.5  # No crashes in scenario
    hedged = sum(1 for h in crash_dates_hit if h)
    return round(hedged / len(crash_dates_hit), 4)
