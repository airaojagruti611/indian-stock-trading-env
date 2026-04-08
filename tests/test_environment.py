"""
tests/test_environment.py — Basic sanity tests for the environment.

Run: python tests/test_environment.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.portfolio import Portfolio
from server.graders import grade_easy, grade_medium, grade_hard, calc_step_reward
from server.scenarios import MarketDataFetcher
from inference import validate_action


def test_portfolio_buy_sell():
    print("Testing portfolio buy/sell...")
    p = Portfolio(starting_capital=100_000, enable_t2=False)

    # Buy 10 shares of RELIANCE at ₹2400
    success, msg = p.buy("RELIANCE", 10, 2400.0, day=1, date_str="2024-01-02")
    assert success, f"Buy failed: {msg}"
    assert "RELIANCE" in p.positions
    assert p.positions["RELIANCE"].quantity == 10
    print(f"  ✅ Buy: {msg[:60]}...")

    # Check portfolio value
    prices = {"RELIANCE": 2500.0}
    val = p.total_value(prices)
    assert val > 100_000, f"Value should be > starting capital: {val}"
    print(f"  ✅ Portfolio value after price rise: ₹{val:,.2f}")

    # Sell 5 shares
    success, msg = p.sell("RELIANCE", 5, 2500.0, day=2, date_str="2024-01-03")
    assert success, f"Sell failed: {msg}"
    assert p.positions["RELIANCE"].quantity == 5
    print(f"  ✅ Sell: {msg[:60]}...")

    # Test insufficient cash
    success, msg = p.buy("TCS", 1000, 3500.0, day=3, date_str="2024-01-04")
    assert not success, "Should fail — not enough cash"
    print(f"  ✅ Insufficient cash correctly rejected")

    print("Portfolio tests PASSED ✅\n")


def test_grader_easy():
    print("Testing easy grader...")
    snapshot = {
        "return_pct": 5.0,   # 5% return
        "total_trades": 2,
        "winning_trades": 1,
        "max_drawdown": 2.0,
        "sebi_violations": 0,
    }
    price_matrix = {
        "RELIANCE": [2400.0, 2420.0, 2450.0, 2480.0, 2520.0],
        "TCS":      [3500.0, 3520.0, 3560.0, 3590.0, 3640.0],
    }
    score = grade_easy(snapshot, price_matrix, starting_capital=100_000)
    assert 0.0 < score < 1.0, f"Score must be strictly between 0 and 1: {score}"
    print(f"  ✅ Easy score: {score:.4f} (agent made 5% return)")

    # Test zero return
    snapshot["return_pct"] = 0.0
    score_zero = grade_easy(snapshot, price_matrix, starting_capital=100_000)
    assert score_zero < score, "Zero return should score less than 5% return"
    print(f"  ✅ Zero return score: {score_zero:.4f} (correctly lower)")

    print("Easy grader tests PASSED ✅\n")


def test_graders_strict_open_interval():
    print("Testing strict-open score bounds for all graders...")

    price_matrix_easy = {
        "RELIANCE": [2400.0, 2420.0, 2450.0, 2480.0, 2520.0],
        "TCS":      [3500.0, 3520.0, 3560.0, 3590.0, 3640.0],
    }
    easy_snapshot = {
        "return_pct": 8.0,
        "total_trades": 2,
        "winning_trades": 1,
        "max_drawdown": 2.0,
        "sebi_violations": 0,
    }
    easy_score = grade_easy(easy_snapshot, price_matrix_easy, starting_capital=100_000)
    assert 0.0 < easy_score < 1.0, f"Easy score not strict-open: {easy_score}"

    price_matrix_medium = {
        "RELIANCE": [2400.0, 2420.0, 2450.0, 2480.0],
        "HDFCBANK": [1500.0, 1510.0, 1520.0, 1530.0],
        "ZOMATO":   [160.0, 162.0, 165.0, 168.0],
    }
    medium_snapshot = {
        "return_pct": 4.0,
        "holdings": {"RELIANCE": 10, "HDFCBANK": 20, "ZOMATO": 40},
        "holdings_market_value": {"RELIANCE": 24_800, "HDFCBANK": 30_600, "ZOMATO": 6_720},
        "total_value": 500_000,
        "total_trades": 6,
        "winning_trades": 2,
        "losing_trades": 1,
        "closed_trades": 3,
        "sebi_violations": 0,
    }
    medium_score = grade_medium(
        portfolio_snapshot=medium_snapshot,
        price_matrix=price_matrix_medium,
        nifty_series=[22000.0, 22100.0, 22250.0, 22300.0],
        starting_capital=500_000,
        sebi_violations=0,
    )
    assert 0.0 < medium_score < 1.0, f"Medium score not strict-open: {medium_score}"

    hard_snapshot = {
        "return_pct": 1.5,
        "max_drawdown": 9.5,
        "sebi_violations": 1,
    }
    hard_score = grade_hard(
        portfolio_snapshot=hard_snapshot,
        price_matrix={
            "RELIANCE": [2500.0, 2480.0, 2520.0, 2550.0],
            "TCS": [3500.0, 3460.0, 3520.0, 3570.0],
            "INFY": [1500.0, 1470.0, 1490.0, 1520.0],
            "ONGC": [180.0, 176.0, 182.0, 185.0],
            "TATASTEEL": [120.0, 115.0, 118.0, 122.0],
        },
        nifty_series=[18000.0, 17800.0, 17950.0, 18100.0],
        starting_capital=1_000_000,
        sebi_violations=1,
        daily_values=[1_000_000, 995_000, 1_005_000, 1_015_000],
        crash_dates_hit=[True, False, True],
    )
    assert 0.0 < hard_score < 1.0, f"Hard score not strict-open: {hard_score}"

    print("Strict-open grader bounds PASSED ✅\n")


def test_grader_step_reward():
    print("Testing step reward...")
    # Good step — made money, no violations
    r = calc_step_reward(
        daily_pnl=1000.0,
        starting_capital=100_000,
        sebi_violations_now=0,
        n_holdings=2,
        total_trades=3,
        action_type="buy",
    )
    assert -0.1 <= r <= 0.1, f"Step reward out of range: {r}"
    assert r > 0, f"Good step should have positive reward: {r}"
    print(f"  ✅ Good step reward: {r:.6f}")

    # Bad step — lost money, SEBI violation
    r_bad = calc_step_reward(
        daily_pnl=-2000.0,
        starting_capital=100_000,
        sebi_violations_now=2,
        n_holdings=1,
        total_trades=20,
        action_type="sell",
    )
    assert r_bad < r, "Bad step should score lower than good step"
    print(f"  ✅ Bad step reward: {r_bad:.6f} (correctly lower)")

    print("Step reward tests PASSED ✅\n")


def test_sebi_rules():
    print("Testing SEBI rule checks...")
    p = Portfolio(starting_capital=100_000)

    # Buy 90% in one stock — should trigger concentration violation
    p.buy("RELIANCE", 30, 2500.0, day=1, date_str="2024-01-02")

    rules = {"max_single_stock_pct": 0.40, "min_cash_reserve_pct": 0.20}
    prices = {"RELIANCE": 2500.0}
    violations = p.check_sebi_rules(rules, prices)

    assert len(violations) > 0, "Should have SEBI violations"
    print(f"  ✅ Correctly detected {len(violations)} SEBI violation(s)")
    for v in violations:
        print(f"     → {v}")

    print("SEBI rule tests PASSED ✅\n")


def test_symbol_fallback_candidates():
    print("Testing symbol fallback candidates...")
    f = MarketDataFetcher()
    cands = f._get_symbol_candidates("ZOMATO")
    assert "ZOMATO.NS" in cands, "Primary ZOMATO symbol missing"
    assert "ETERNAL.NS" in cands, "Fallback alias for ZOMATO missing"
    assert cands[0] == "ZOMATO.NS", f"Unexpected candidate ordering: {cands}"
    print(f"  ✅ ZOMATO candidates: {cands}")
    print("Symbol fallback tests PASSED ✅\n")


def test_validate_action_buy_clamp():
    print("Testing buy action clamping for SEBI/cash...")
    obs = {
        "portfolio": {"RELIANCE": 100},
        "current_prices": {"RELIANCE": 2500.0, "TCS": 3500.0},
        "cash_balance": 20_000.0,
        "portfolio_value": 400_000.0,
        "unrealized_pnl": {"RELIANCE": 1_000.0},
        "current_date": "2022-02-20",
    }
    action = {
        "action_type": "buy",
        "stock_symbol": "RELIANCE",
        "quantity": 1_000,
        "reasoning": "go big",
    }
    out = validate_action(action, obs, "hard")
    assert out["action_type"] in {"buy", "hold"}, f"Unexpected action: {out}"
    if out["action_type"] == "buy":
        assert out["quantity"] < 1_000, "Expected quantity clamp for risk controls"
    print(f"  ✅ Buy action validated: {out}")
    print("Buy clamp tests PASSED ✅\n")


def test_validate_action_forced_stop_loss():
    print("Testing forced stop-loss sell behavior...")
    obs = {
        "portfolio": {"INFY": 10},
        "current_prices": {"INFY": 900.0},
        "cash_balance": 200_000.0,
        "portfolio_value": 209_000.0,
        # pnl=-2000 for qty=10 implies avg=1100, so loss ~18.2% (>10%)
        "unrealized_pnl": {"INFY": -2_000.0},
        "current_date": "2022-05-10",
    }
    action = {
        "action_type": "hold",
        "stock_symbol": "",
        "quantity": 0,
        "reasoning": "wait",
    }
    out = validate_action(action, obs, "hard")
    assert out["action_type"] == "sell", f"Expected forced sell, got: {out}"
    assert out["stock_symbol"] == "INFY", f"Expected INFY forced sell, got: {out}"
    assert out["quantity"] == 10, f"Expected full exit quantity, got: {out}"
    print(f"  ✅ Forced stop-loss action: {out}")
    print("Stop-loss behavior tests PASSED ✅\n")


if __name__ == "__main__":
    print("=" * 55)
    print("🇮🇳 Indian Stock Trading Environment — Unit Tests")
    print("=" * 55 + "\n")

    test_portfolio_buy_sell()
    test_grader_easy()
    test_graders_strict_open_interval()
    test_grader_step_reward()
    test_sebi_rules()
    test_symbol_fallback_candidates()
    test_validate_action_buy_clamp()
    test_validate_action_forced_stop_loss()

    print("=" * 55)
    print("✅ All tests passed!")
    print("=" * 55)
