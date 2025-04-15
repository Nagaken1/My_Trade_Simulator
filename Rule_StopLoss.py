import pandas as pd
from My_Trade_Simulator import Order

def run(current_ohlc, positions_df, order_history, strategy_id='Rule_StopLoss'):
    """
    成行BUYでエントリーし、100円下に逆指値SELLを出すストップロス戦略。
    逆指値は Low がトリガー価格以下になったときに即時発動・約定。
    """
    new_orders = []
    time = current_ohlc.time
    close = current_ohlc.close
    low = current_ohlc.low

    # === 1. 成行BUYでエントリー ===
    entry_id = f"{strategy_id}_{time:%Y%m%d%H%M%S}"
    order_entry = Order(
        strategy_id=strategy_id,
        side='BUY',
        price=close,
        quantity=1,
        order_time=time,
        order_type='market',
        position_effect='open'
    )
    order_entry.order_id = entry_id
    new_orders.append(order_entry)

    # === 2. ストップロス注文を同時に出す（50円下の逆指値SELL） ===
    stop_price = close - 50

    order_stop = Order(
        strategy_id=strategy_id,
        side='SELL',
        price=stop_price,
        quantity=1,
        order_time=time,  # 発注タイミング
        order_type='stop',
        trigger_price=stop_price,
        position_effect='close'
    )
    order_stop.order_id = f"{entry_id}_close"
    new_orders.append(order_stop)

    return new_orders