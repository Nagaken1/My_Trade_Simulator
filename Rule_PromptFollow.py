import pandas as pd
from My_Trade_Simulator import Order
import logging

support_lines = []
resistance_lines = []
previous_support = None
previous_resistance = None
support_touch_count = 0
resistance_touch_count = 0
last_entry_time = None
LOOKBACK_MINUTES = 3

def run(current_ohlc, positions, order_book, strategy_id="Rule_PromptFollow", ohlc_history=None):
    global support_lines, resistance_lines, previous_support, previous_resistance
    global support_touch_count, resistance_touch_count, last_entry_time

    orders = []
    time = current_ohlc.time
    low = current_ohlc.low
    high = current_ohlc.high

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

    if highs[0] < highs[1] > highs[2]:
        previous_resistance = highs[1]
        resistance_lines.append(previous_resistance)
        resistance_touch_count = 0
        logging.debug(f"[DETECTED] 抵抗線: {previous_resistance} @ {time}")

    if lows[0] > lows[1] < lows[2]:
        previous_support = lows[1]
        support_lines.append(previous_support)
        support_touch_count = 0
        logging.debug(f"[DETECTED] 支持線: {previous_support} @ {time}")

    if previous_support and low <= previous_support:
        support_touch_count += 1
        logging.debug(f"[TOUCH] 支持線接触: {support_touch_count} 回 @ {time}")

    if previous_resistance and high >= previous_resistance:
        resistance_touch_count += 1
        logging.debug(f"[TOUCH] 抵抗線接触: {resistance_touch_count} 回 @ {time}")

    for p in positions:
        if not p.is_closed() and p.strategy_id == strategy_id:
            if p.side == "BUY" and support_touch_count >= 2 and previous_support and not p.has_limit_order:
                close_order = Order(
                    strategy_id=strategy_id,
                    side="SELL",
                    price=previous_support,
                    quantity=p.quantity,
                    order_time=time,
                    order_type="limit",
                    position_effect="close",
                    target_entry_id=p.entry_order_id
                )
                close_order.order_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_limit_close"
                orders.append(close_order)
                p.has_limit_order = True
                logging.debug(f"[EXIT] 支持線2回接触 → 支持線で利確SELL @ {previous_support} | target_entry_id={p.entry_order_id}")
                support_touch_count = 0

            elif p.side == "SELL" and resistance_touch_count >= 2 and previous_resistance and not p.has_limit_order:
                close_order = Order(
                    strategy_id=strategy_id,
                    side="BUY",
                    price=previous_resistance,
                    quantity=p.quantity,
                    order_time=time,
                    order_type="limit",
                    position_effect="close",
                    target_entry_id=p.entry_order_id
                )
                close_order.order_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_limit_close"
                orders.append(close_order)
                p.has_limit_order = True
                logging.debug(f"[EXIT] 抵抗線2回接触 → 抵抗線で利確BUY @ {previous_resistance} | target_entry_id={p.entry_order_id}")
                resistance_touch_count = 0

    open_positions = [p for p in positions if getattr(p, "strategy_id", strategy_id) == strategy_id and not p.is_closed()]
    if open_positions:
        logging.debug(f"[SKIP] 同戦略の未決済ポジションが {len(open_positions)} 件存在 @ {time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    has_pending_entry_order = any(
        o.strategy_id == strategy_id and
        o.position_effect == "open" and
        o.status == "pending"
        for o in order_book.pending_orders
    )
    if has_pending_entry_order:
        logging.debug(f"[SKIP] 未約定の建玉注文ありのためスキップ @ {time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    if last_entry_time and (time - last_entry_time).total_seconds() < 90:
        logging.debug(f"[SKIP] 前回の注文が近いためスキップ: last={last_entry_time}, now={time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

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
            position_effect="close",
            target_entry_id=entry_id
        )
        stop_order.order_id = stop_id
        orders.append(stop_order)

        logging.debug(f"[ENTRY] 抵抗線ブレイク買い: {open4} > {resistance_lines[-1]} @ {time}")
        last_entry_time = time

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
            position_effect="close",
            target_entry_id=entry_id
        )
        stop_order.order_id = stop_id
        orders.append(stop_order)

        logging.debug(f"[ENTRY] 支持線ブレイク売り: {open4} < {support_lines[-1]} @ {time}")
        last_entry_time = time

    return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None
