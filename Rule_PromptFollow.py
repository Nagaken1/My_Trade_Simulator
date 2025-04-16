import pandas as pd
from My_Trade_Simulator import Order

support_lines = []
resistance_lines = []
last_entry_index = -9999  # 重複エントリー防止用
LOOKBACK_MINUTES = 3

def run(current_ohlc, positions_df, order_history, strategy_id="Rule_PromptFollow", ohlc_history=None):
    global support_lines, resistance_lines, last_entry_index
    orders = []

    time = current_ohlc.time
    index = int(time.strftime("%Y%m%d%H%M"))

    # ✅ ohlc_history から3本取得し、current_ohlcと合わせて4本の履歴を作る
    if ohlc_history is None or len(ohlc_history) < 3:
        return orders, None, None

    combined = ohlc_history.tail(3).copy()
    combined = pd.concat([combined, pd.DataFrame([vars(current_ohlc)])], ignore_index=True)

    highs = combined["high"].values
    lows = combined["low"].values
    opens = combined["open"].values
    open4 = opens[3]

    print(f"[DEBUG] 検査: Highs = {highs}, time={time}")
    print(f"[DEBUG] 検査: Lows = {lows}, time={time}")

    # --- 抵抗線（山型）
    if highs[0] < highs[1] > highs[2]:
        resistance_lines.append(highs[1])
        print(f"[DETECTED] 抵抗線: {highs[1]} @ {time}")

    # --- 支持線（谷型）
    if lows[0] > lows[1] < lows[2]:
        support_lines.append(lows[1])
        print(f"[DETECTED] 支持線: {lows[1]} @ {time}")

    # 重複エントリー防止
    if index == last_entry_index:
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    # 建玉保有状況
    open_pos = positions_df[positions_df['exit_time'].isna()]
    has_buy = not open_pos[open_pos['side'] == 'BUY'].empty
    has_sell = not open_pos[open_pos['side'] == 'SELL'].empty

    # --- 買いシグナル ---
    if resistance_lines and open4 > resistance_lines[-1] and not has_buy:
        entry_price = open4
        entry_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}"
        stop_id = f"{entry_id}_close"

        entry_order = Order(
            strategy_id=strategy_id,
            side="BUY",
            price=entry_price,
            quantity=1,
            order_time=time,
            order_type="market",
            position_effect="open"
        )
        entry_order.order_id = entry_id
        orders.append(entry_order)

        stop_price = entry_price - 25
        stop_order = Order(
            strategy_id=strategy_id,
            side="SELL",
            price=stop_price,
            quantity=1,
            order_time=time,
            order_type="stop",
            trigger_price=stop_price,
            position_effect="close"
        )
        stop_order.order_id = stop_id
        orders.append(stop_order)

        print(f"[ENTRY] 抵抗線ブレイク買い: {open4} > {resistance_lines[-1]} @ {time}")
        last_entry_index = index

    # --- 売りシグナル ---
    elif support_lines and open4 < support_lines[-1] and not has_sell:
        entry_price = open4
        entry_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}"
        stop_id = f"{entry_id}_close"

        entry_order = Order(
            strategy_id=strategy_id,
            side="SELL",
            price=entry_price,
            quantity=1,
            order_time=time,
            order_type="market",
            position_effect="open"
        )
        entry_order.order_id = entry_id
        orders.append(entry_order)

        stop_price = entry_price + 25
        stop_order = Order(
            strategy_id=strategy_id,
            side="BUY",
            price=stop_price,
            quantity=1,
            order_time=time,
            order_type="stop",
            trigger_price=stop_price,
            position_effect="close"
        )
        stop_order.order_id = stop_id
        orders.append(stop_order)

        print(f"[ENTRY] 支持線ブレイク売り: {open4} < {support_lines[-1]} @ {time}")
        last_entry_index = index

    return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None