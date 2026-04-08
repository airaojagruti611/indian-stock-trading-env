"""
inference.py — Baseline LLM agent for Indian Stock Trading Environment.

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
MAX_TOKENS   = 256
TASKS        = ["easy", "medium", "hard"]

# Max steps per task (must finish in <20 min total)
MAX_STEPS = {
    "easy":   5,
    "medium": 20,
    "hard":   30,
}

MAX_TOTAL_REWARD = 1.0
SUCCESS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Required log format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True
    )

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

SYSTEM_PROMPT = """You are an expert Indian stock market trading agent operating on NSE/BSE.

Your goal: maximize portfolio returns while following SEBI regulations.

AVAILABLE ACTIONS:
- buy  <SYMBOL> <QUANTITY>  : Buy shares (e.g., buy RELIANCE 10)
- sell <SYMBOL> <QUANTITY>  : Sell shares (e.g., sell TCS 5)
- hold                       : Do nothing this day

SEBI RULES TO FOLLOW:
- Do not put more than 60% of portfolio in one stock
- Keep minimum 10% cash reserve at all times
- Stop-loss: sell if any stock is down 10%+ from your buy price

RESPONSE FORMAT:
Respond with a JSON object only. No explanation outside the JSON.
{
  "action_type": "buy" | "sell" | "hold",
  "stock_symbol": "RELIANCE",
  "quantity": 10,
  "reasoning": "Brief reason"
}

INDIAN MARKET CONTEXT:
- Stocks are traded on NSE (National Stock Exchange)
- Prices are in Indian Rupees (INR / ₹)
- NIFTY50 is the benchmark index
- Consider news headlines when making decisions
"""

def build_user_prompt(observation: Dict) -> str:
    """Format observation as a prompt for the LLM."""
    prices = observation.get("current_prices", {})
    portfolio = observation.get("portfolio", {})
    cash = observation.get("cash_balance", 0)
    pv = observation.get("portfolio_value", 0)
    day = observation.get("current_day", 1)
    total_days = observation.get("total_days", 5)
    nifty = observation.get("nifty50", 0)
    nifty_chg = observation.get("nifty_change_pct", 0)
    news = observation.get("market_news", "")
    pnl = observation.get("unrealized_pnl", {})
    warnings = observation.get("sebi_warnings", [])
    changes = observation.get("price_change_pct", {})
    task = observation.get("task_id", "easy")

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

    # Format warnings
    warning_text = ""
    if warnings:
        warning_text = "\n⚠️  SEBI WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings)

    return f"""=== DAY {day} of {total_days} | Task: {task.upper()} ===

📰 Market News: {news}

📊 NIFTY 50: ₹{nifty:,.2f} ({nifty_chg:+.2f}%)

💰 Current Prices:
{price_lines}

📁 Your Portfolio:
{port_lines}
  Cash: ₹{cash:,.2f}
  Total Value: ₹{pv:,.2f}
{warning_text}

What is your trading decision for today?
Respond with JSON only."""


def parse_llm_response(text: str) -> Dict:
    """Parse LLM response into a trading action dict."""
    # Try direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from text
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: hold
    return {
        "action_type":  "hold",
        "stock_symbol": "",
        "quantity":     0,
        "reasoning":    "Could not parse response — holding.",
    }


def get_llm_action(client: OpenAI, obs_dict: Dict) -> Dict:
    """Call LLM and return parsed action."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs_dict)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()
        return parse_llm_response(text)
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return {"action_type": "hold", "stock_symbol": "", "quantity": 0, "reasoning": "LLM error"}


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
            obs = result.observation

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                # Convert observation to dict for LLM
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
                }

                # Get LLM decision
                action_dict = get_llm_action(client, obs_dict)
                action_str  = (
                    f"{action_dict.get('action_type','hold')} "
                    f"{action_dict.get('stock_symbol','')} "
                    f"{action_dict.get('quantity',0)}"
                ).strip()

                # Execute action
                action = IndianTradingAction(
                    stock_symbol=action_dict.get("stock_symbol", ""),
                    action_type=action_dict.get("action_type", "hold"),
                    quantity=int(action_dict.get("quantity", 0)),
                    reasoning=action_dict.get("reasoning", ""),
                )

                result = await env.step(action)
                obs    = result.observation

                reward    = result.reward or 0.0
                done      = result.done
                error     = None

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

        # Final score = last reward (grader score from final step)
        score   = rewards[-1] if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score   = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = []
    for task_id in TASKS:
        result = await run_task(task_id, client)
        all_results.append(result)
        await asyncio.sleep(2)  # Brief pause between tasks

    # Summary
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
