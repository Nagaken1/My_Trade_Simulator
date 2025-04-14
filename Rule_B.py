import pandas as pd
from collections import deque
#from Order import Order  # 必要に応じて

def run(df: pd.DataFrame, strategy_id: str = 'RuleB') -> pd.DataFrame:
    """
    毎分 SELL → 3分後に決済（BUY）する単純な逆張り型戦略。
    Signal: -1=売り, 1=買戻し, 0=何もしない
    Profit: 決済時に損益を記録（それ以外は0）
    """
    result = pd.DataFrame({'Date': pd.to_datetime(df['Date'])})
    result.set_index('Date', inplace=True)

    result['Signal'] = 0
    result['Profit'] = 0.0

    # ✅ マーカー列（描画のために必須）
    result['OrderPlaced'] = 0
    result['Executed'] = 0
    result['LossCutTriggered'] = 0

    positions = deque()

    for i in range(len(df)):
        now = df.iloc[i]
        now_time = pd.to_datetime(now['Date'])
        price = now['Close']

        # 5分おきにBUY（例: indexが5の倍数）
        if i % 5 == 0:
            positions.append({'entry_time': now_time, 'entry_price': price, 'quantity': 1})
            result.at[now_time, 'Signal'] = 1
            result.at[now_time, 'OrderPlaced'] = 1
            result.at[now_time, 'Executed'] = 1

        # 3分後に決済（あくまで例です）
        if len(positions) > 0 and i >= 3:
            pos = positions.popleft()
            pnl = (price - pos['entry_price']) * pos['quantity']
            result.at[now_time, 'Profit'] = pnl

    return result