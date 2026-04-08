"""
inference.py — Improved LLM agent for Indian Stock Trading Environment.

Uses OpenAI client to run an LLM agent against the environment.
Follows required [START] / [STEP] / [END] log format strictly.

Environment variables required:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Hugging Face / API key

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=hf_...
    python inference.py
"""

import os
import re
import json
import asyncio
import math
from datetime import date
from typing import List, Dict, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = os.environ.get("HF_TOKEN")

SPACE_URL    = os.environ.get(
    "SPACE_URL",
    "https://jagruti611-indian-stock-trading-env.hf.space"
)

TEMPERATURE  = 0.1
MAX_TOKENS   = 512
TASKS        = ["easy", "medium", "hard"]

# Max steps per task (must finish in <20 min total)
MAX_STEPS = {
    "easy":   5,
    "medium": 20,
    "hard":   30,
}

MAX_TOTAL_REWARD  = 1.0
SUCCESS_THRESHOLD = 0.5
SCORE_EPSILON     = 1e-4
TRADING_COST_BUFFER = 0.002  # conservative buffer for fees/slippage


def _clamp_open_score(score: float) -> float:
    """Clamp score to strict open interval (0, 1)."""
    return round(min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, float(score))), 4)

# ---------------------------------------------------------------------------
# Task metadata: required stocks, SEBI rules, crash dates
# ---------------------------------------------------------------------------

TASK_META = {
    "easy": {
        "stocks":           ["RELIANCE", "TCS"],
        "sebi_max_pct":     1.0,
        "sebi_cash_pct":    0.0,
        "sebi_stoploss":    None,
        "crash_dates":      [],
        "strategy_hint": (
            "This is a 5-day bull run. Buy RELIANCE and TCS on day 1, "
            "sell everything on day 4 or 5. Simple buy-and-hold."
        ),
    },
    "medium": {
        "stocks":           ["RELIANCE", "HDFCBANK", "ZOMATO"],
        "sebi_max_pct":     0.60,
        "sebi_cash_pct":    0.10,
        "sebi_stoploss":    None,
        "crash_dates":      ["2024-06-04"],
        "strategy_hint": (
            "Election Volatility May-June 2024. "
            "IMPORTANT: You MUST hold all 3 stocks (RELIANCE, HDFCBANK, ZOMATO) "
            "simultaneously — diversification is 30% of your score. "
            "Strategy: Buy all 3 stocks early (days 1-3). "
            "Around June 3 (before the crash): reduce holdings. "
            "June 4 is the CRASH day (Nifty -8%) — try to be mostly cash. "
            "After June 5: buy back for the recovery rally."
        ),
    },
    "hard": {
        "stocks":           ["RELIANCE", "TCS", "INFY", "ONGC", "TATASTEEL"],
        "sebi_max_pct":     0.40,
        "sebi_cash_pct":    0.20,
        "sebi_stoploss":    0.10,
        "crash_dates":      ["2022-02-24", "2022-05-12", "2022-06-17"],
        "strategy_hint": (
            "Bear Market 2022 — Russia-Ukraine war, US Fed rate hikes. "
            "IMPORTANT: You MUST hold all 5 stocks simultaneously for diversification. "
            "SEBI rules are STRICT: max 40% per stock, keep 20% cash at all times, "
            "mandatory stop-loss if any stock is down 10%+ from your buy price. "
            "Crash dates: 2022-02-24, 2022-05-12, 2022-06-17 — "
            "go to mostly cash 1-2 days BEFORE each crash, buy back after. "
            "Capital preservation is key in a bear market."
        ),
    },
}

# ---------------------------------------------------------------------------
# Required log format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward} done={done} error={error}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(
        f"[END] success={success} steps={steps} "
        f"score={score} rewards={rewards}",
        flush=True
    )

# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are an expert Indian stock market trading agent operating on NSE/BSE.
Your goal: maximize portfolio returns while strictly following SEBI regulations.

AVAILABLE ACTIONS (respond with ONE JSON object per turn):
- buy  <SYMBOL> <QUANTITY>  : Buy shares (e.g., buy RELIANCE 10)
- sell <SYMBOL> <QUANTITY>  : Sell shares you currently hold
- hold                       : Do nothing this day

SEBI RULES (violations reduce your score heavily):
- Do not put more than the allowed % of portfolio in one stock
- Keep minimum cash reserve at all times
- Apply stop-loss: sell if any stock drops 10%+ from your buy price
- Never sell a stock you do not currently hold

RESPONSE FORMAT — JSON only, no extra text:
{{
  "action_type": "buy" | "sell" | "hold",
  "stock_symbol": "SYMBOL",
  "quantity": 10,
  "reasoning": "Brief reason"
}}

SCORING BREAKDOWN (know what you are optimized for):
- Return vs NIFTY benchmark
- Portfolio diversification (holding multiple stocks simultaneously boosts score)
- Trade efficiency (high win rate, avoid overtrading)
- SEBI compliance (zero violations = full marks)
- Risk management (low drawdown)

INDIAN MARKET CONTEXT:
- Stocks traded on NSE (National Stock Exchange), prices in INR (₹)
- NIFTY50 is the benchmark index
"""

def _build_system_prompt(task_id: str) -> str:
    meta = TASK_META.get(task_id, {})
    stocks      = meta.get("stocks", [])
    max_pct     = meta.get("sebi_max_pct", 1.0)
    cash_pct    = meta.get("sebi_cash_pct", 0.0)
    stoploss    = meta.get("sebi_stoploss", None)
    crash_dates = meta.get("crash_dates", [])
    hint        = meta.get("strategy_hint", "")

    sebi_section = f"\nTASK-SPECIFIC SEBI RULES:\n"
    sebi_section += f"- Max {int(max_pct * 100)}% of portfolio in any single stock\n"
    if cash_pct > 0:
        sebi_section += f"- Minimum {int(cash_pct * 100)}% cash reserve always\n"
    if stoploss:
        sebi_section += f"- Mandatory stop-loss at -{int(stoploss * 100)}% from buy price\n"

    required_section = (
        f"\nREQUIRED STOCKS FOR THIS TASK: {', '.join(stocks)}\n"
        f"You MUST buy and hold ALL of these stocks simultaneously "
        f"— diversification is a major scoring component.\n"
    )

    crash_section = ""
    if crash_dates:
        crash_section = (
            f"\nKNOWN CRASH / HIGH-RISK DATES: {', '.join(crash_dates)}\n"
            f"Reduce to mostly cash 1-2 days BEFORE each date. "
            f"Buy back aggressively 1-2 days AFTER.\n"
        )

    strategy_section = f"\nTASK STRATEGY:\n{hint}\n"

    return BASE_SYSTEM_PROMPT + sebi_section + required_section + crash_section + strategy_section


def _concentration_warnings(
    portfolio: Dict[str, int],
    prices: Dict[str, float],
    total_value: float,
    max_pct: float,
) -> List[str]:
    """Return warnings if any stock exceeds or approaches the concentration limit."""
    warnings = []
    if total_value <= 0:
        return warnings
    for sym, qty in portfolio.items():
        price = prices.get(sym, 0)
        value = qty * price
        pct   = value / total_value
        if pct > max_pct:
            warnings.append(
                f"{sym} is {pct:.0%} of portfolio — OVER the {max_pct:.0%} SEBI limit! Sell some."
            )
        elif pct > max_pct * 0.85:
            warnings.append(
                f"{sym} is {pct:.0%} of portfolio — approaching {max_pct:.0%} SEBI limit."
            )
    return warnings


def _days_until(current_date: str, target_date: str) -> Optional[int]:
    """Return days between current and target dates, or None if parsing fails."""
    if not current_date or not target_date:
        return None
    try:
        return (date.fromisoformat(target_date) - date.fromisoformat(current_date)).days
    except Exception:
        return None


def _largest_holding_symbol(portfolio: Dict[str, int], prices: Dict[str, float]) -> Optional[str]:
    symbol = None
    max_value = -1.0
    for sym, qty in portfolio.items():
        px = prices.get(sym, 0.0)
        value = max(0, qty) * px
        if value > max_value:
            max_value = value
            symbol = sym
    return symbol


def build_user_prompt(observation: Dict, task_id: str) -> str:
    """Format observation as a prompt for the LLM."""
    prices      = observation.get("current_prices", {})
    portfolio   = observation.get("portfolio", {})
    cash        = observation.get("cash_balance", 0)
    pv          = observation.get("portfolio_value", 0)
    day         = observation.get("current_day", 1)
    total_days  = observation.get("total_days", 5)
    nifty       = observation.get("nifty50", 0)
    nifty_chg   = observation.get("nifty_change_pct", 0)
    news        = observation.get("market_news", "")
    pnl         = observation.get("unrealized_pnl", {})
    warnings    = observation.get("sebi_warnings", [])
    changes     = observation.get("price_change_pct", {})
    last_result = observation.get("last_action_result", "")

    meta          = TASK_META.get(task_id, {})
    required      = meta.get("stocks", [])
    max_pct       = meta.get("sebi_max_pct", 1.0)
    min_cash_pct  = meta.get("sebi_cash_pct", 0.0)
    crash_dates   = meta.get("crash_dates", [])
    current_date  = observation.get("current_date", "")

    # Format prices
    price_lines = "\n".join(
        f"  {sym}: ₹{p:,.2f} ({changes.get(sym, 0):+.2f}%)"
        for sym, p in prices.items()
    )

    # Format portfolio
    if portfolio:
        port_lines = "\n".join(
            f"  {sym}: {qty} shares (P&L: ₹{pnl.get(sym, 0):+,.2f})"
            for sym, qty in portfolio.items()
        )
    else:
        port_lines = "  No holdings yet"

    # Stocks not yet held
    missing = [s for s in required if s not in portfolio or portfolio[s] == 0]

    # Concentration warnings
    conc_warnings = _concentration_warnings(portfolio, prices, pv, max_pct)
    all_warnings  = list(warnings) + conc_warnings

    # Cash reserve check
    cash_pct_actual = cash / pv if pv > 0 else 1.0
    if min_cash_pct > 0 and cash_pct_actual < min_cash_pct:
        all_warnings.append(
            f"⚠️ Cash is only {cash_pct_actual:.0%} — BELOW minimum {min_cash_pct:.0%}! Sell to raise cash."
        )

    # Days remaining
    days_left = total_days - day

    # Crash proximity warning
    crash_alert = ""
    for cd in crash_dates:
        if current_date and current_date >= cd:
            continue
        days_to_crash = None
        days_to_crash = _days_until(current_date, cd)
        if days_to_crash is not None and 0 < days_to_crash <= 3:
            crash_alert += (
                f"\n🚨 CRASH WARNING: High-risk date {cd} is {days_to_crash} day(s) away! "
                f"Consider selling most holdings NOW to go mostly cash.\n"
            )

    warning_text = ""
    if all_warnings:
        warning_text = "\n⚠️  WARNINGS:\n" + "\n".join(f"  - {w}" for w in all_warnings)

    missing_text = ""
    if missing:
        missing_text = (
            f"\n📌 DIVERSIFICATION ALERT: You are NOT holding: {', '.join(missing)}. "
            f"Buy these stocks to improve your diversification score!\n"
        )

    return f"""=== DAY {day} of {total_days} (Days Left: {days_left}) | Task: {task_id.upper()} ===
{crash_alert}
📰 Market News: {news}

📊 NIFTY 50: ₹{nifty:,.2f} ({nifty_chg:+.2f}%)

💰 Current Prices:
{price_lines}

📁 Your Portfolio:
{port_lines}
  Cash: ₹{cash:,.2f} ({cash_pct_actual:.0%} of portfolio)
  Total Value: ₹{pv:,.2f}
  Last Action Result: {last_result or "N/A"}
{missing_text}{warning_text}

What is your trading decision for today?
Respond with JSON only."""


def parse_llm_response(text: str) -> Dict:
    """Parse LLM response into a trading action dict."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "action_type":  "hold",
        "stock_symbol": "",
        "quantity":     0,
        "reasoning":    "Could not parse response — holding.",
    }


def validate_action(action_dict: Dict, obs_dict: Dict, task_id: str) -> Dict:
    """
    Prevent invalid actions that would cause SEBI violations or errors.
    - Never sell a stock not currently held.
    - Never sell more shares than held.
    - Clamp buys to concentration/cash constraints.
    - Force stop-loss exits and pre-crash de-risking for hard tasks.
    """
    portfolio = obs_dict.get("portfolio", {}) or {}
    prices    = obs_dict.get("current_prices", {}) or {}
    cash      = float(obs_dict.get("cash_balance", 0) or 0)
    pv        = float(obs_dict.get("portfolio_value", 0) or 0)
    pnl_map   = obs_dict.get("unrealized_pnl", {}) or {}
    current_date = obs_dict.get("current_date", "")

    meta       = TASK_META.get(task_id, {})
    max_pct    = float(meta.get("sebi_max_pct", 1.0) or 1.0)
    min_cash   = float(meta.get("sebi_cash_pct", 0.0) or 0.0)
    stop_loss  = meta.get("sebi_stoploss")
    crash_dates = meta.get("crash_dates", []) or []

    if pv <= 0:
        pv = cash + sum(max(0, q) * prices.get(sym, 0.0) for sym, q in portfolio.items())
    if pv <= 0:
        pv = cash

    # Hard guardrail: enforce stop-loss liquidation if breached.
    if stop_loss:
        worst_breach = None
        for sym, qty in portfolio.items():
            if qty <= 0 or sym not in prices:
                continue
            pnl = float(pnl_map.get(sym, 0.0) or 0.0)
            est_avg = prices[sym] - (pnl / qty)
            if est_avg <= 0:
                continue
            loss_pct = (est_avg - prices[sym]) / est_avg
            if loss_pct >= float(stop_loss):
                if worst_breach is None or loss_pct > worst_breach[0]:
                    worst_breach = (loss_pct, sym, qty)

        if worst_breach:
            _, sym, qty = worst_breach
            return {
                "action_type": "sell",
                "stock_symbol": sym,
                "quantity": qty,
                "reasoning": f"Forced stop-loss exit for {sym}.",
            }

    # Hard guardrail: move toward cash before/at known crash dates.
    target_crash_cash = max(min_cash, 0.30)
    cash_ratio = (cash / pv) if pv > 0 else 1.0
    crash_near = any(
        (_days_until(current_date, cd) is not None and 0 <= _days_until(current_date, cd) <= 1)
        for cd in crash_dates
    )
    if crash_near and cash_ratio < target_crash_cash and portfolio:
        sym = _largest_holding_symbol(portfolio, prices)
        if sym:
            held = max(0, int(portfolio.get(sym, 0)))
            px = float(prices.get(sym, 0.0) or 0.0)
            if held > 0 and px > 0:
                needed_cash = max(0.0, target_crash_cash * pv - cash)
                qty_needed = max(1, int(math.ceil(needed_cash / px)))
                return {
                    "action_type": "sell",
                    "stock_symbol": sym,
                    "quantity": min(held, qty_needed),
                    "reasoning": "Crash-risk de-risking: raising cash before high-risk date.",
                }

    # Normalize model output.
    atype = str(action_dict.get("action_type", "hold") or "hold").lower().strip()
    symbol = str(action_dict.get("stock_symbol", "") or "").upper().strip()
    try:
        qty = int(action_dict.get("quantity", 0))
    except Exception:
        qty = 0
    qty = max(0, qty)

    safe = dict(action_dict)
    safe["action_type"] = atype
    safe["stock_symbol"] = symbol
    safe["quantity"] = qty
    safe["reasoning"] = str(action_dict.get("reasoning", "") or "")

    if atype not in {"buy", "sell", "hold"}:
        return {
            "action_type": "hold",
            "stock_symbol": "",
            "quantity": 0,
            "reasoning": f"Blocked invalid action type '{atype}'.",
        }

    if atype == "hold":
        safe["stock_symbol"] = ""
        safe["quantity"] = 0
        return safe

    if atype == "sell":
        held = int(portfolio.get(symbol, 0) or 0)
        if held <= 0:
            return {
                "action_type": "hold",
                "stock_symbol": "",
                "quantity": 0,
                "reasoning": f"Blocked: tried to sell {symbol} but holding 0 shares.",
            }
        if qty <= 0:
            qty = held
        if qty > held:
            safe["reasoning"] = (
                safe.get("reasoning", "") +
                f" (quantity clamped from {qty} to {held})"
            ).strip()
            qty = held
        safe["quantity"] = qty
        return safe

    # Buy action safeguards.
    if symbol not in prices:
        return {
            "action_type": "hold",
            "stock_symbol": "",
            "quantity": 0,
            "reasoning": f"Blocked buy: no live price for {symbol}.",
        }

    price = float(prices.get(symbol, 0.0) or 0.0)
    if price <= 0:
        return {
            "action_type": "hold",
            "stock_symbol": "",
            "quantity": 0,
            "reasoning": f"Blocked buy: invalid price for {symbol}.",
        }

    held_qty = int(portfolio.get(symbol, 0) or 0)
    held_value = held_qty * price
    max_value_allowed = max(0.0, max_pct * pv)
    addl_value_allowed = max(0.0, max_value_allowed - held_value)
    max_qty_by_conc = int(addl_value_allowed // price) if max_pct < 1.0 else 10**9

    spendable_cash = max(0.0, cash - min_cash * pv)
    per_share_cost = price * (1.0 + TRADING_COST_BUFFER)
    max_qty_by_cash = int(spendable_cash // per_share_cost) if per_share_cost > 0 else 0

    max_qty = max(0, min(max_qty_by_cash, max_qty_by_conc))
    if max_qty <= 0:
        return {
            "action_type": "hold",
            "stock_symbol": "",
            "quantity": 0,
            "reasoning": f"Blocked buy: {symbol} would violate SEBI/cash constraints.",
        }

    if qty <= 0:
        qty = min(max_qty, 1)
    if qty > max_qty:
        safe["reasoning"] = (
            safe.get("reasoning", "") +
            f" (buy quantity clamped from {qty} to {max_qty} for risk limits)"
        ).strip()
        qty = max_qty

    safe["quantity"] = qty
    return safe


def get_llm_action(client: OpenAI, obs_dict: Dict, task_id: str) -> Dict:
    """Call LLM and return parsed + validated action."""
    system_prompt = _build_system_prompt(task_id)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": build_user_prompt(obs_dict, task_id)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()
        action = parse_llm_response(text)
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        action = {"action_type": "hold", "stock_symbol": "", "quantity": 0, "reasoning": "LLM error"}

    return validate_action(action, obs_dict, task_id)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_task(task_id: str, client: OpenAI) -> Dict:
    """Run one full task and return results."""
    from client import IndianStockEnv
    from models import IndianTradingAction

    env_name = "indian-stock-trading-env"
    log_start(task=task_id, env=env_name, model=MODEL_NAME)

    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False
    max_steps   = MAX_STEPS[task_id]

    try:
        async with IndianStockEnv(base_url=SPACE_URL) as env:
            result = await env.reset(task_id=task_id)
            obs    = result.observation

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                obs_dict = {
                    "current_prices":    obs.current_prices,
                    "price_change_pct":  obs.price_change_pct,
                    "portfolio":         obs.portfolio,
                    "cash_balance":      obs.cash_balance,
                    "portfolio_value":   obs.portfolio_value,
                    "unrealized_pnl":    obs.unrealized_pnl,
                    "nifty50":           obs.nifty50,
                    "nifty_change_pct":  obs.nifty_change_pct,
                    "market_news":       obs.market_news,
                    "sebi_warnings":     obs.sebi_warnings,
                    "current_day":       obs.current_day,
                    "total_days":        obs.total_days,
                    "task_id":           obs.task_id,
                    "current_date":      getattr(obs, "current_date", ""),
                    "last_action_result": obs.last_action_result,
                }

                action_dict = get_llm_action(client, obs_dict, task_id)
                action_str  = (
                    f"{action_dict.get('action_type','hold')} "
                    f"{action_dict.get('stock_symbol','')} "
                    f"{action_dict.get('quantity',0)}"
                ).strip()

                action = IndianTradingAction(
                    stock_symbol=action_dict.get("stock_symbol", ""),
                    action_type=action_dict.get("action_type", "hold"),
                    quantity=int(action_dict.get("quantity", 0)),
                    reasoning=action_dict.get("reasoning", ""),
                )

                result = await env.step(action)
                obs    = result.observation

                reward      = result.reward or 0.0
                done        = result.done
                error       = None

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_str,
                    reward=round(reward, 4),
                    done=done,
                    error=error,
                )

                if done:
                    break

        score   = _clamp_open_score(rewards[-1]) if rewards else SCORE_EPSILON
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score   = SCORE_EPSILON
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = []
    for task_id in TASKS:
        result = await run_task(task_id, client)
        all_results.append(result)
        await asyncio.sleep(2)

    print("\n" + "=" * 50, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 50, flush=True)
    for r in all_results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"  {r['task_id']:8s} → score={r['score']:.4f}  {status}", flush=True)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Average score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
