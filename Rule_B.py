import pandas as pd
from collections import deque
#from Order import Order  # 必要に応じて

def run(df: pd.DataFrame, strategy_id: str = 'RuleB') -> pd.DataFrame:
    """
    毎分 SELL → 3分後に決済（BUY）する単純な逆張り型戦略。
    Signal: -1=売り, 1=買戻し, 0=何もしない
    Profit: 決済時に損益を記録（それ以外は0）
    """
    result = pd.DataFrame(index=df['Date'])
    result['Signal'] = 0
    result['Profit'] = 0.0

    positions = deque()

    for i in range(len(df)):
        now = df.iloc[i]
        now_time = now['Date']
        price = now['Close']

        # 新規売りポジション
        positions.append({'entry_time': now_time, 'entry_price': price, 'quantity': 1})
        result.at[now_time, 'Signal'] = -1

        # 3分後に決済
        if len(positions) > 0 and i >= 3:
            pos = positions.popleft()
            pnl = (pos['entry_price'] - price) * pos['quantity']
            result.at[now_time, 'Profit'] = pnl

    return result