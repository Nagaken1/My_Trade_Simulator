import pandas as pd
from collections import deque
from My_Trade_Simulator import Order
# 過去価格保持用（戦略ごとにグローバルに持っても良い）


def run(current_ohlc, positions_df, order_history, strategy_id='Rule_B'):
    """
    ロジック: 終値が前回より上昇 → 買い建て、下降 → 売り建て。
    2分前の未決済ポジションがあれば決済。
    """
    new_orders = []
    time = current_ohlc.time
    close = current_ohlc.close

    # --- 前回の終値を記録する static-like 変数 ---
    if not hasattr(run, "prev_close"):
        run.prev_close = close

    # --- シグナル検出 ---
    signal = 0
    if close > run.prev_close:
        signal = 1  # BUY
    elif close < run.prev_close:
        signal = -1  # SELL
    run.prev_close = close

    if signal != 0:
        entry_id = f"{strategy_id}_{time:%Y%m%d%H%M%S}"
        order_entry = Order(
            strategy_id=strategy_id,
            side='BUY' if signal == 1 else 'SELL',
            price=close,
            quantity=1,
            order_time=time,
            order_type='market',
            position_effect='open'
        )
        order_entry.order_id = entry_id
        new_orders.append(order_entry)

    # --- 2分前のエントリー注文を決済（order_historyから照合） ---
    two_min_ago = time - pd.Timedelta(minutes=2)

    open_orders = order_history[
        (order_history['strategy_id'] == strategy_id) &
        (order_history['position_effect'] == 'open') &
        (order_history['status'] == 'executed') &
        (order_history['order_time'] == two_min_ago)
    ]

    for _, row in open_orders.iterrows():
        entry_oid = row['order_id']
        close_order = Order(
            strategy_id=strategy_id,
            side='SELL' if row['side'] == 'BUY' else 'BUY',
            price=close,
            quantity=row['quantity'],
            order_time=time,
            order_type='market',
            position_effect='close'
        )
        close_order.order_id = f"{entry_oid}_close"
        new_orders.append(close_order)

    return new_orders