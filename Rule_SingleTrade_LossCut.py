import pandas as pd
from My_Trade_Simulator  import Order

def run(current_ohlc, positions_df, order_history, strategy_id='Rule_SingleTrade_LossCut'):
    orders = []

    time = current_ohlc.time
    close = current_ohlc.close

    # ✅ 「一度でもエントリー済み」かどうかを strategy_id 単位で判断
    entry_orders = order_history[
        (order_history['strategy_id'] == strategy_id) &
        (order_history['position_effect'] == 'open')
    ]

    if entry_orders.empty:
        # ✅ 初回エントリーと同時に逆指値ロスカットを出す
        entry_order_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}"
        stop_order_id = f"{entry_order_id}_close"

        # エントリー（成行）
        entry_order = Order(
            strategy_id=strategy_id,
            side='BUY',
            price=close,
            quantity=1,
            order_time=time,
            order_type='market',
            position_effect='open'
        )
        entry_order.order_id = entry_order_id
        orders.append(entry_order)
        print(f"[DEBUG] エントリー発注: {entry_order_id}")

        # ✅ ストップロス（逆指値、trigger_price指定）
        losscut_price = close - 50
        stop_order = Order(
            strategy_id=strategy_id,
            side='SELL',
            price=losscut_price,               # 実行価格（成行にするなら close でもOK）
            quantity=1,
            order_time=time,
            order_type='stop',
            trigger_price=losscut_price,       # ✅ この価格に達したら発動
            position_effect='close'
        )
        stop_order.order_id = stop_order_id
        orders.append(stop_order)
        print(f"[DEBUG] 逆指値ロスカット発注: {stop_order_id} @ trigger={losscut_price}")

        return orders
    return orders