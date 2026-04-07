"""
app.py — FastAPI server for Indian Stock Trading Environment.

Wires the environment factory with action/observation types for OpenEnv's HTTP API:
/ws, /reset, /step, /state, /health, /docs
"""

from openenv.core.env_server import create_fastapi_app
from models import IndianTradingAction, MarketObservation

from .environment import IndianStockEnvironment

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
