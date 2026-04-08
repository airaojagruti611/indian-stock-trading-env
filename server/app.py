"""
app.py — FastAPI server for Indian Stock Trading Environment.

Wires the environment factory with action/observation types for OpenEnv's HTTP API:
/ws, /reset, /step, /state, /health, /docs
"""

import os
import sys

from openenv.core.env_server import create_fastapi_app
try:
    from models import IndianTradingAction, MarketObservation
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models import IndianTradingAction, MarketObservation

try:
    from .environment import IndianStockEnvironment
except ImportError:
    from environment import IndianStockEnvironment

app = create_fastapi_app(
    IndianStockEnvironment,
    IndianTradingAction,
    MarketObservation,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
