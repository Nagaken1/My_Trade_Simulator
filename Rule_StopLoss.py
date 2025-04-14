import pandas as pd
from collections import deque


def run(df: pd.DataFrame, strategy_id: str = "StopLoss") -> pd.DataFrame:
    result = pd.DataFrame({'Date': pd.to_datetime(df['Date'])})
    result.set_index('Date', inplace=True)
    result['Signal'] = 0
    result['Profit'] = 0.0
    result['IsStopLoss'] = False

    positions = deque()
    active_stop_orders = {}

    for i in range(len(df)):
        current = df.iloc[i]
        now_time = pd.to_datetime(current['Date'])
        price = current['Close']

        # 新規成行BUY（エントリー）
        positions.append({
            'entry_time': now_time,
            'entry_price': price,
            'quantity': 1
        })
        result.at[now_time, 'Signal'] = 1

        # ロスカット価格設定（100円下）
        stop_price = price - 100
        active_stop_orders[now_time] = {
            'trigger_price': stop_price,
            'side': 'SELL',
            'triggered': False,
            'executed_time': None,
            'exit_price': None
        }

        # チェック＆約定判定（エントリーから順に）
        to_remove = []
        for entry_time, stop in active_stop_orders.items():
            if not stop['triggered']:
                if current['Low'] <= stop['trigger_price']:
                    stop['triggered'] = True
                    stop['executed_time'] = now_time
                    stop['exit_price'] = stop['trigger_price']

                    # 利益計算
                    entry_price = positions[0]['entry_price']
                    quantity = positions[0]['quantity']
                    pnl = (stop['exit_price'] - entry_price) * quantity

                    result.at[now_time, 'Profit'] = pnl
                    result.at[now_time, 'IsStopLoss'] = True
                    to_remove.append(entry_time)

        # 決済済みの注文削除
        for entry_time in to_remove:
            positions.popleft()
            del active_stop_orders[entry_time]

    return result