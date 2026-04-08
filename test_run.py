from client import IndianStockEnv
from models import IndianTradingAction

with IndianStockEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy")
    print(result.observation.current_prices)
    # Should show REAL NSE prices like:
    # {'RELIANCE': 2431.25, 'TCS': 3834.70}