import pandas as pd
from collections import deque


def run(df: pd.DataFrame, strategy_id: str = "StopLoss") -> pd.DataFrame:
    result = pd.DataFrame({'Date': pd.to_datetime(df['Date'])})
    result.set_index('Date', inplace=True)

    result['Signal'] = 0
    result['Profit'] = 0.0
    result['OrderPlaced'] = 0
    result['Executed'] = 0
    result['LossCutTriggered'] = 0

    positions = deque()
    active_stop_orders = {}

    for i in range(len(df)):
        current = df.iloc[i]
        now_time = pd.to_datetime(current['Date'])
        price = current['Close']

        # === 1. エントリー: 成行BUY ===
        positions.append({
            'entry_time': now_time,
            'entry_price': price,
            'quantity': 1
        })
        result.at[now_time, 'Signal'] = 1
        result.at[now_time, 'OrderPlaced'] = 1  # ← 注文を出したタイミング
        result.at[now_time, 'Executed'] = 1     # ← 即成行で約定された

        # === 2. ロスカット逆指値注文を同時に発行（100円下） ===
        stop_price = price - 100
        active_stop_orders[now_time] = {
            'trigger_price': stop_price,
            'side': 'SELL',
            'triggered': False,
            'executed_time': None,
            'exit_price': None
        }

        # === 3. ロスカット約定判定 ===
        to_remove = []
        for entry_time, stop in active_stop_orders.items():
            if not stop['triggered']:
                if current['Low'] <= stop['trigger_price']:
                    stop['triggered'] = True
                    stop['executed_time'] = now_time
                    stop['exit_price'] = stop['trigger_price']

                    entry_price = positions[0]['entry_price']
                    quantity = positions[0]['quantity']
                    pnl = (stop['exit_price'] - entry_price) * quantity

                    result.at[now_time, 'Profit'] = pnl
                    result.at[now_time, 'Executed'] = 1
                    result.at[now_time, 'LossCutTriggered'] = 1
                    to_remove.append(entry_time)

        for entry_time in to_remove:
            positions.popleft()
            del active_stop_orders[entry_time]

    return result