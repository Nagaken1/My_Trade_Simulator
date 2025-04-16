import pandas as pd
from My_Trade_Simulator import Order
import logging

support_lines = []
resistance_lines = []
last_entry_time = None  # 最後のエントリー発注時刻
LOOKBACK_MINUTES = 3

def run(current_ohlc, positions, order_book, strategy_id="Rule_PromptFollow", ohlc_history=None):
    global support_lines, resistance_lines, last_entry_time
    orders = []

    time = current_ohlc.time

    # --- 3本以上の履歴がないと判定不能 ---
    if ohlc_history is None or len(ohlc_history) < 3:
        return orders, None, None

    combined = ohlc_history.tail(3).copy()
    combined = pd.concat([combined, pd.DataFrame([vars(current_ohlc)])], ignore_index=True)

    highs = combined["high"].values
    lows = combined["low"].values
    opens = combined["open"].values
    open4 = opens[3]

    logging.debug(f"[検査] Highs = {highs}, time = {time}")
    logging.debug(f"[検査] Lows = {lows}, time = {time}")

    # --- 抵抗線（山型） ---
    if highs[0] < highs[1] > highs[2]:
        resistance_lines.append(highs[1])
        logging.debug(f"[DETECTED] 抵抗線: {highs[1]} @ {time}")

    # --- 支持線（谷型） ---
    if lows[0] > lows[1] < lows[2]:
        support_lines.append(lows[1])
        logging.debug(f"[DETECTED] 支持線: {lows[1]} @ {time}")

    # --- 同一戦略の未決済ポジションが存在する場合、発注しない ---
    open_positions = [p for p in positions if getattr(p, "strategy_id", strategy_id) == strategy_id and not p.is_closed()]
    if open_positions:
        logging.debug(f"[SKIP] 同戦略の未決済ポジションが {len(open_positions)} 件存在 @ {time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    # --- 同一戦略の未約定の新規注文が存在する場合、発注しない ---
    has_pending_entry_order = any(
        o.strategy_id == strategy_id and
        o.position_effect == "open" and
        o.status == "pending"
        for o in order_book.pending_orders
    )
    if has_pending_entry_order:
        logging.debug(f"[SKIP] 未約定の建玉注文ありのためスキップ @ {time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    # --- 前回の発注から1分以内ならスキップ（datetimeベース） ---
    if last_entry_time and (time - last_entry_time).total_seconds() < 90:
        logging.debug(f"[SKIP] 前回の注文が近いためスキップ: last={last_entry_time}, now={time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    # --- 買いシグナル ---
    if resistance_lines and open4 > resistance_lines[-1]:
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

        logging.debug(f"[ENTRY] 抵抗線ブレイク買い: {open4} > {resistance_lines[-1]} @ {time}")
        last_entry_time = time

    # --- 売りシグナル ---
    elif support_lines and open4 < support_lines[-1]:
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

        logging.debug(f"[ENTRY] 支持線ブレイク売り: {open4} < {support_lines[-1]} @ {time}")
        last_entry_time = time

    return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None