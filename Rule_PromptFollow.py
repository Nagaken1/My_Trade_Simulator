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

    # === 既存ポジションに対して、OCO（ロスカット・利確）を出す（build_time が現在時刻より前のもののみ）===
    for p in positions:
        if not p.is_settlement() and p.strategy_id == strategy_id:
            logging.debug(f"[OCO対象確認] p.order_id={p.order_id}, build_time={p.build_time}, now={time}")
            if p.stoploss_order_id or p.profitfixed_order_id:
                continue  # すでにOCOが出ているならスキップ

            if p.build_time >= time:  # ✅ 同じ分ならまだ未約定（simulate_strategy の都合）
                continue  # 翌分以降に処理する

            build_price = p.build_price
            quantity = p.build_quantity
            side = p.position_side

            stop_price = build_price - 25 if side == "BUY" else build_price + 25
            profit_price = build_price + 25 if side == "BUY" else build_price - 25

            stop_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_stop"
            profit_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_profit"

            stop_order = Order(
                order_id=stop_id,
                strategy_id=strategy_id,
                order_side="SELL" if side == "BUY" else "BUY",
                order_price=stop_price,
                order_quantity=quantity,
                order_time=time,
                order_type="limit",
                order_effect="settlement",
                target_id=p.order_id,
                trigger_price=stop_price
            )
            profit_order = Order(
                order_id=profit_id,
                strategy_id=strategy_id,
                order_side="SELL" if side == "BUY" else "BUY",
                order_price=profit_price,
                order_quantity=quantity,
                order_time=time,
                order_type="limit",
                order_effect="settlement",
                target_id=p.order_id,
                trigger_price=profit_price
            )

            orders.extend([stop_order, profit_order])

            logging.debug(f"[OCO] run()内で発注: {p.order_id} → STOP={stop_price}, PROFIT={profit_price}")

    open_positions = [p for p in positions if getattr(p, "strategy_id", strategy_id) == strategy_id and not p.is_settlement()]
    if open_positions:
        logging.debug(f"[SKIP] 同戦略の未決済ポジションが {len(open_positions)} 件存在 @ {time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    has_pending_entry_order = any(
        o.strategy_id == strategy_id and
        o.order_effect == "newie" and
        o.status == "pending"
        for o in order_book.pending_orders
    )
    if has_pending_entry_order:
        logging.debug(f"[SKIP] 未約定の建玉注文ありのためスキップ @ {time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None

    if last_entry_time and (time - last_entry_time).total_seconds() < 90:
        logging.debug(f"[SKIP] 前回の注文が近いためスキップ: last={last_entry_time}, now={time}")
        return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None


    # --- 抵抗線ブレイク（BUYエントリー + ロスカットSELL） ---
    if resistance_lines and open4 > resistance_lines[-1]:
        build_price = open4
        newie_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_newie"
        stop_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_stop"

        newie_order = Order(
            order_id=newie_id,
            strategy_id=strategy_id,
            order_side="BUY",
            order_price=build_price,
            order_quantity=1,
            order_time=time,
            order_type="market",
            order_effect="newie"  # ✅ エントリー明示
        )
        orders.append(newie_order)

        logging.debug(f"[ENTRY] 抵抗線ブレイク買い: {build_price} > {resistance_lines[-1]} @ {time}")
        last_entry_time = time

    # --- 支持線ブレイク（SELLエントリー + ロスカットBUY） ---
    elif support_lines and open4 < support_lines[-1]:
        build_price = open4
        newie_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_newie"
        stop_id = f"{strategy_id}_{time.strftime('%Y%m%d%H%M%S')}_stop"

        newie_order = Order(
            order_id=newie_id,
            strategy_id=strategy_id,
            order_side="SELL",
            order_price=build_price,
            order_quantity=1,
            order_time=time,
            order_type="market",
            order_effect="newie"  # ✅ エントリー明示
        )
        orders.append(newie_order)

        logging.debug(f"[ENTRY] 支持線ブレイク売り: {build_price} < {support_lines[-1]} @ {time}")
        last_entry_time = time

    return orders, support_lines[-1] if support_lines else None, resistance_lines[-1] if resistance_lines else None
