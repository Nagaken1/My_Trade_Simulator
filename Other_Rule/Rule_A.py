import pandas as pd
from collections import deque
from datetime import datetime, timedelta, time as dtime
from My_Trade_Simulator import Order,get_trade_date


# --- 毎分呼び出される戦略関数 ---
def run(current_ohlc, positions_df, order_history, strategy_id='Rule_A'):
    """
    毎分 BUY → 1分後に SELL する単純な時間ベース戦略。
    EntryOrderIDを使ってProfitを後からマッピング可能にします。
    """
    new_orders = []
    time = current_ohlc.time
    close = current_ohlc.close

    trade_date_str = get_trade_date(time).strftime("%Y%m%d")
    entry_id = f"{strategy_id}_{trade_date_str}{time:%H%M%S}"

    # --- 新規建玉（BUY） ---
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

    # --- 決済（SELL） 前回の建玉があれば1つだけ決済 ---
    past_open_orders = order_history[
        (order_history['strategy_id'] == strategy_id) &
        (order_history['position_effect'] == 'open') &
        (order_history['status'] == 'executed')
    ]

    past_close_orders = order_history[
        (order_history['strategy_id'] == strategy_id) &
        (order_history['position_effect'] == 'close') &
        (order_history['status'] == 'executed')
    ]

    open_executed_ids = set(past_open_orders['order_id'])
    close_executed_ids = set(past_close_orders['order_id'].str.replace('_close', ''))

    # 未決済のopenがあれば決済
    remaining = sorted(list(open_executed_ids - close_executed_ids))
    if remaining:
        entry_oid = remaining[0]
        order_close = Order(
            strategy_id=strategy_id,
            side='SELL',
            price=close,
            quantity=1,
            order_time=time,
            order_type='market',
            position_effect='close'
        )
        order_close.order_id = f"{entry_oid}_close"
        new_orders.append(order_close)

    return new_orders