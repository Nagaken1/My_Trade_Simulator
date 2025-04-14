import pandas as pd
from collections import deque
#from Order import Order  # 必要に応じて

def run(df: pd.DataFrame, strategy_id: str = 'RuleA') -> pd.DataFrame:
    """
    毎分 BUY → 2分後に決済（SELL）する単純戦略。
    Signal: 1=買い, -1=売り, 0=何もしない
    Profit: 決済時に損益を記録（それ以外は0）
    """
    result = pd.DataFrame({'Date': pd.to_datetime(df['Date'])})
    result.set_index('Date', inplace=True)
    result['Signal'] = 0
    result['Profit'] = 0.0

    positions = deque()

    for i in range(len(df)):
        now = df.iloc[i]
        now_time = pd.to_datetime(df.iloc[i]['Date'])
        price = now['Close']

        # 新規買いポジション
        positions.append({'entry_time': now_time, 'entry_price': price, 'quantity': 1})
        result.at[now_time, 'Signal'] = 1

        # 2分後に決済
        if len(positions) > 0 and i >= 2:
            pos = positions.popleft()
            pnl = (price - pos['entry_price']) * pos['quantity']
            result.at[now_time, 'Profit'] = pnl

    return result