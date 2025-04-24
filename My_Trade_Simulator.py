import os
import sys
import glob
import pandas as pd
import numpy as np
import importlib.util
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timedelta
import time
import logging

# --- OHLC クラス ---
class OHLC:
    def __init__(self, time, open_, high, low, close):
        self.time = time      # datetime オブジェクト（例: 2025-04-11 09:01:00）
        self.open = open_     # 始値（float）
        self.high = high      # 高値（float）
        self.low = low        # 安値（float）
        self.close = close    # 終値（float）

# --- Order クラス ---
class Order:
    def __init__(
        self,
        order_id: str,
        strategy_id: str,
        order_side: str,        # 'BUY' または 'SELL'
        order_price: float,     # 指値注文の価格（マーケットなら None）
        order_quantity: int,
        order_time,             # datetimeオブジェクト
        order_type: str = 'limit',  # 'limit', 'market'
        trigger_price: float = None,  # 逆指値発動価格
        order_effect: str = 'newie',  # 'newie' or 'settlement'
        target_id: str = None         # 決済注文の対象建玉ID
    ):
        self.order_id = order_id
        self.strategy_id = strategy_id
        self.order_side = order_side  # 'BUY' または 'SELL'
        self.order_price = order_price  # 注文価格
        self.order_quantity = order_quantity  # 注文数量
        self.order_time = order_time  # 注文時刻
        self.order_type = order_type  # 'limit', 'market'
        self.trigger_price = trigger_price  # 逆指値注文の発動価格
        self.order_effect = order_effect
        self.target_id = target_id  # 建玉指定（決済用）

        self.triggered = False  # ストップ注文の発動フラグ
        self.status = 'pending'  # 'pending', 'executed', 'canceled'
        self.execution_price = None  # 約定価格
        self.execution_time = None  # 約定時刻

        self.order_category = None  # "New", "profitfixed", "Stop" など

        self.cancel_time = None # キャンセルされた時刻を記録する

# --- Position クラス ---
class Position:
    def __init__(
        self,
        order_id,                   # 新規建玉に紐づく注文ID
        strategy_id,               # 戦略(Rule)の名前
        position_side,             # 'BUY' または 'SELL'
        build_price,               # エントリー価格
        build_quantity,            # 保有数量
        build_time,                # 建玉作成時間（datetime）
        settlement_time=None,      # 決済時間（まだ決済していなければ None）
        settlement_price=None      # 決済価格（同上）
    ):
        self.order_id = order_id
        self.strategy_id = strategy_id
        self.position_side = position_side

        self.build_price = build_price
        self.build_quantity = build_quantity
        self.build_time = build_time

        self.settlement_order_id = None
        self.settlement_time = settlement_time
        self.settlement_price = settlement_price

        self.profitfixed_order_id = None  # 利益確定注文ID
        self.stoploss_order_id = None     # ロスカット注文ID

        self.realized_profit = None  # ✅ 損益（決済完了時に記録される）

    def is_settlement(self):
        """ポジションが決済済みかどうかを返す"""
        return self.settlement_time is not None

    def profit(self):
        """ポジションが決済されていれば損益を返す、されていなければ0"""
        if not self.is_settlement():
            return 0
        if self.position_side == 'BUY':
            return (self.settlement_price - self.build_price) * self.build_quantity
        else:  # 'SELL'
            return (self.build_price - self.settlement_price) * self.build_quantity

# --- OrderBook クラス ---
class OrderBook:
    def __init__(self):
        self.orders: list[Order] = []             # 全注文（分析/記録用に保持）
        self.pending_orders: list[Order] = []     # マッチ対象の未約定注文のみ
        self.executed_orders: list[Order] = []    # 約定済みのみ保持
        self.canceled_orders: list[Order] = []    # キャンセルされた注文を保持
        self.positions: list[Position] = []       # 建玉情報（OrderBookに保持される想定）

    def add_order(self, order: Order, positions: list[Position]):
        logging.debug(f"[DEBUG][add_order] 受け取り: {order.order_id}, effect={order.order_effect}, category={getattr(order, 'order_category', '?')}")

        if order.order_effect == 'newie':
            order.order_category = "New"

        elif order.order_effect == 'settlement':
            matched_pos = next((p for p in positions if not p.is_settlement() and p.order_id == order.target_id), None)

            if matched_pos:
                logging.debug(f"[DEBUG][matched_pos] found: {matched_pos.order_id}, stoploss_id={matched_pos.stoploss_order_id}, profitfixed_id={matched_pos.profitfixed_order_id}")

                if order.trigger_price is not None:
                    is_stop = (
                        matched_pos.position_side == 'BUY' and order.order_price < matched_pos.build_price
                    ) or (
                        matched_pos.position_side == 'SELL' and order.order_price > matched_pos.build_price
                    )

                    if is_stop:
                        if matched_pos.stoploss_order_id is not None:
                            logging.warning(f"[SKIP] ロスカット注文はすでに存在: pos_id={matched_pos.order_id}")
                            return
                        order.order_category = "Stop"
                        matched_pos.stoploss_order_id = order.order_id
                        logging.debug(f"[LINK] ロスカット注文を登録: pos_id={matched_pos.order_id} → order_id={order.order_id}")

                    else:
                        if matched_pos.profitfixed_order_id is not None:
                            logging.warning(f"[SKIP] 利確注文はすでに存在: pos_id={matched_pos.order_id}")
                            return
                        order.order_category = "Profitfixed"
                        matched_pos.profitfixed_order_id = order.order_id
                        logging.debug(f"[LINK] 利確注文を登録: pos_id={matched_pos.order_id} → order_id={order.order_id}")
                else:
                    order.order_category = "Settlement"
                    logging.debug(f"[INFO] 成行による決済注文: pos_id={matched_pos.order_id} → order_id={order.order_id}")

            else:
                order.order_category = "Settlement"
                logging.warning(f"[CATEGORY] Settlement注文に対するポジションが見つかりませんでした: {order.order_id}")

        # 最終登録
        self.orders.append(order)
        self.pending_orders.append(order)
        logging.debug(f"[ADD_ORDER] {order.order_id}: effect={order.order_effect}, category={order.order_category}, target={order.target_id}")
        logging.debug(f"[DEBUG][add_order] 追加成功: {order.order_id} → pending_orders")

    def match_orders(self, ohlc, positions: list, current_index=None, time_index_map=None):
        executed = []
        still_pending = []

        # ---- パス①：新規（newie）注文処理 ----
        for order in self.pending_orders:
            order_index = time_index_map.get(order.order_time, -1)
            if order_index > current_index:
                logging.debug(f"[SKIP] 時系列不一致: {order.order_id} は future の注文扱いとしてスキップされました")
                still_pending.append(order)
                continue

            if order.order_effect != 'newie':
                continue

            executed_flag = False

            # --- 成行注文 ---
            if order.order_type == 'market':
                if order.order_side == 'BUY':
                    order.execution_price = ohlc['open'] + 5
                else:
                    order.execution_price = ohlc['open'] - 5
                executed_flag = True

            # --- 指値注文 ---
            elif order.order_type == 'limit':
                if (order.order_side == 'BUY' and ohlc['low'] <= order.order_price) or \
                   (order.order_side == 'SELL' and ohlc['high'] >= order.order_price):
                    order.execution_price = order.order_price
                    executed_flag = True

            if executed_flag:
                order.execution_time = ohlc['time']
                order.status = 'executed'
                self.executed_orders.append(order)
                executed.append(order)

                new_position = Position(
                    order_id=order.order_id,
                    strategy_id=order.strategy_id,
                    position_side=order.order_side,
                    build_price=order.execution_price,
                    build_quantity=order.order_quantity,
                    build_time=order.execution_time,
                )
                positions.append(new_position)
                logging.debug(f"[CREATE POSITION] order_id={order.order_id}, side={order.order_side}, entry_time={order.execution_time}, closed={new_position.is_settlement()}")
            else:
                still_pending.append(order)

        # ---- パス②：決済（settlement）注文処理 ----
        for order in self.pending_orders:
            order_index = time_index_map.get(order.order_time, -1)
            if order_index > current_index or order.order_effect != 'settlement':
                continue

            executed_flag = False
            order_id = order.target_id if hasattr(order, "target_id") else order.order_id.replace("_settlement", "")

            # --- 決済注文の指値判定（Stop: スリッページなし、Profitfixed: スリッページあり） ---
            if order.order_type == 'limit':
                if order.order_category == "Stop":
                    logging.debug(f"[CHECK][STOP] {order.order_id} | Side={order.order_side} | "
                                  f"Low={ohlc['low']}, High={ohlc['high']}, "
                                  f"Price={order.order_price}, Category={order.order_category}, "
                                  f"OrderTime={order.order_time}, CurrentOHLC={ohlc['time']}")

                    # ロスカット：スリッページなし
                    logging.debug(f"[CHECK][STOP] {order.order_id}: High={ohlc['high']} vs Price={order.order_price}")
                    if order.order_side == 'BUY':
                        if ohlc['high'] >= order.order_price:
                            order.execution_price = order.order_price
                            executed_flag = True
                    elif order.order_side == 'SELL':
                        if ohlc['low'] <= order.order_price:
                            order.execution_price = order.order_price
                            executed_flag = True

                elif order.order_category == "Profitfixed":
                    # 利確：スリッページ5円必要
                    logging.debug(f"[CHECK][PROFIT] {order.order_id}: High={ohlc['high']} vs Price+5={order.order_price + 5}")
                    logging.debug(f"[CHECK][CATEGORY DISPATCH] {order.order_id}: category={order.order_category}")

                    if order.order_side == 'BUY':
                        if ohlc['low'] <= order.order_price - 5:
                            order.execution_price = order.order_price
                            executed_flag = True
                    elif order.order_side == 'SELL':
                        if ohlc['high'] >= order.order_price + 5:
                            order.execution_price = order.order_price
                            executed_flag = True

            # --- スリッページ付き成行決済判定 ---
            elif order.order_type == 'market':
                if order.order_side == 'BUY':
                    order.execution_price = ohlc['open'] + 5
                elif order.order_side == 'SELL':
                    order.execution_price = ohlc['open'] - 5
                executed_flag = True

            if executed_flag:
                order.execution_time = ohlc['time']
                order.status = 'executed'
                self.executed_orders.append(order)
                executed.append(order)

                order_id = getattr(order, "target_id", order.order_id.replace("_settlement", ""))

                matched = False
                for pos in positions:
                    if not pos.is_settlement():
                        # 精密に照合（完全一致）＋ デバッグログ
                        if pos.order_id == order_id:
                            logging.debug(f"[MATCH] order {order.order_id} → matched with pos {pos.order_id}")
                            pos.settlement_price = order.execution_price
                            pos.settlement_time = order.execution_time
                            pos.settlement_order_id = order.order_id

                            # --- 即時損益計算 ---
                            if pos.position_side == 'BUY':
                                gross_profit = pos.settlement_price - pos.build_price
                            else:
                                gross_profit = pos.build_price - pos.settlement_price

                            # 日経225mini → 100倍レバレッジ、手数料は固定77円
                            net_profit = gross_profit * 100 - 77
                            pos.realized_profit = net_profit

                            # OCOキャンセル処理
                            if order.order_category == "Stop" and pos.profitfixed_order_id:
                                self.cancel_order_by_id(pos.profitfixed_order_id, triggered_by=order.order_id, cancel_time=ohlc['time'])
                            elif order.order_category == "Profitfixed" and pos.stoploss_order_id:
                                self.cancel_order_by_id(pos.stoploss_order_id, triggered_by=order.order_id, cancel_time=ohlc['time'])

                            matched = True
                            logging.debug(f"[CLOSE MATCHED] {order.order_id} が建玉 {pos.order_id} を決済しました（価格={order.execution_price}, 時刻={order.execution_time}）")
                            break
                        else:
                            logging.debug(f"[MISMATCH] order {order.order_id} vs pos {pos.order_id}")

                if not matched:
                    logging.warning(f"[WARNING] 決済注文 {order.order_id} は約定されたが、対象建玉が見つかりませんでした (target_id={order_id})")
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        # --- すべての処理が終わったあと、まだ pending に残っている注文を記録 ---
        for order in still_pending:
            logging.debug(f"[PENDING] 注文未約定のまま継続中: {order.order_id}, category={order.order_category}, time={order.order_time}, status={order.status}")
        return executed

    def cancel_order_by_id(self, cancel_id: str, triggered_by: str = "",cancel_time: datetime= None):
        """
        指定した order_id の注文をキャンセル処理する。
        pending_orders から削除し、canceled_orders に移動する。
        """
        canceled = False
        for order in self.pending_orders:
            if order.order_id == cancel_id:
                order.cancel_time = cancel_time
                order.status = 'canceled'
                self.canceled_orders.append(order)
                self.pending_orders.remove(order)
                canceled = True
                logging.info(f"[CANCEL] OCOキャンセル: {cancel_id} が {triggered_by} によってキャンセルされました")
                break
        if not canceled:
            logging.warning(f"[CANCEL] キャンセル対象の注文が見つかりませんでした: {cancel_id}")

# --- 統計計算クラス ---
class TradeStatisticsCalculator:
    """取引損益に基づく戦略指標を時系列で計算"""

    @staticmethod
    def total_profit(profit_list):
        """累計損益"""
        return np.cumsum(profit_list).tolist()

    @staticmethod
    def winning_rate(profit_list):
        """勝率の推移（= 勝ち数 / 総取引数）"""
        profit_array = np.array(profit_list)
        wins = (profit_array > 0).astype(int)
        trades = (profit_array != 0).astype(int)
        cum_wins = np.cumsum(wins)
        cum_trades = np.cumsum(trades)
        with np.errstate(divide='ignore', invalid='ignore'):
            win_rate = np.where(cum_trades > 0, cum_wins / cum_trades, 0.0)
        return np.round(win_rate, 4).tolist()

    @staticmethod
    def payoff_ratio(profit_list):
        """ペイオフレシオの推移（= 平均利益 / 平均損失）"""
        win_total = 0
        loss_total = 0
        win_count = 0
        loss_count = 0
        ratios = []

        for p in profit_list:
            if p > 0:
                win_total += p
                win_count += 1
            elif p < 0:
                loss_total += p
                loss_count += 1
            if win_count > 0 and loss_count > 0:
                avg_win = win_total / win_count
                avg_loss = abs(loss_total / loss_count)
                ratios.append(round(avg_win / avg_loss, 4))
            else:
                ratios.append(0.0)
        return ratios

    @staticmethod
    def expected_value(winning_rate_list, payoff_ratio_list):
        """期待値の推移（= 勝率×ペイオフ − 負け率）"""
        win_rate = np.array(winning_rate_list)
        payoff = np.array(payoff_ratio_list)
        expected = win_rate * payoff - (1 - win_rate)
        return np.round(expected, 4).tolist()

    @staticmethod
    def drawdown(profit_list):
        profits = np.array(profit_list)
        peak = np.maximum.accumulate(profits)
        dd = peak - profits
        return np.round(dd, 4).tolist()

    @staticmethod
    def max_drawdown(drawdown_list):
        drawdowns = np.array(drawdown_list)
        max_dd = np.maximum.accumulate(drawdowns)
        return max_dd.tolist()

# ====== OrderBook内の約定情報を集約して辞書化 ======

def build_orderbook_price_map(order_book):
    """
    約定済みの Order オブジェクトを {order_id: Order instance} で返す
    """
    order_map = {}
    for order in order_book.orders:
        if order.status == 'executed' and order.order_id is not None:
            order_map[order.order_id] = order
    return order_map

# --- メイン処理 ---
def simulate_strategy(strategy_id, strategy_func, ohlc_list):
    start_time_total = time.perf_counter()

    state = {
        'order_book': OrderBook(),
        'positions': [],
        'log': []
    }

    strategy_module = importlib.import_module(strategy_func.__module__)
    lookback_minutes = getattr(strategy_module, "LOOKBACK_MINUTES", 0)
    lookback_ohlc_buffer = deque(maxlen=lookback_minutes)
    time_index_map = {ohlc.time: i for i, ohlc in enumerate(ohlc_list)}

    for i, current_ohlc in enumerate(ohlc_list):
        if i >= lookback_minutes:
            ohlc_df_history = pd.DataFrame([vars(ohlc) for ohlc in lookback_ohlc_buffer])
        else:
            ohlc_df_history = pd.DataFrame()

        log_entry = {"Date": current_ohlc.time}
        state['log'].append(log_entry)

        # 戦略実行
        t0 = time.perf_counter()

        try:
            new_orders, support, resistance = strategy_func(
                current_ohlc=current_ohlc,
                positions=state['positions'],
                order_book=state['order_book'],
                strategy_id=strategy_id,
                ohlc_history=ohlc_df_history
            )
        except ValueError:
            new_orders = strategy_func(
                current_ohlc=current_ohlc,
                positions=state['positions'],
                order_book=state['order_book'],
                strategy_id=strategy_id,
                ohlc_history=ohlc_df_history
            )
            support = resistance = None

        t1 = time.perf_counter()
        logging.debug(f"[TIME] {strategy_id} 戦略実行: {t1 - t0:.4f} 秒")

        log_entry["SupportLine"] = support
        log_entry["ResistanceLine"] = resistance

        if i > 0:
            t2 = time.perf_counter()
            executed_now = state['order_book'].match_orders(
                vars(current_ohlc),
                state['positions'],
                current_index=i,
                time_index_map=time_index_map
            )
            t3 = time.perf_counter()
            logging.debug(f"[TIME] {strategy_id} 約定処理: {t3 - t2:.4f} 秒")

            for exec_order in executed_now:
                if exec_order.order_effect == "newie":
                    kind = "New"
                elif exec_order.order_effect == "settlement":
                    kind = exec_order.order_category

                side = "Buy" if exec_order.order_side == "BUY" else "Sell"
                t = pd.to_datetime(exec_order.execution_time).floor("T")
                match = next((r for r in state['log'] if r["Date"] == t), None)
                if match:
                    match[f"{side}_{kind}_ExecID"] = exec_order.order_id
                    match[f"{side}_{kind}_ExecTime"] = exec_order.execution_time
                    match[f"{side}_{kind}_ExecPrice"] = exec_order.execution_price

                    # --- Profit を即時ログに記録（settlement のみ）---
                    if exec_order.order_effect == "settlement":
                        matched_pos = next((p for p in state["positions"]
                                            if p.settlement_order_id == exec_order.order_id), None)

                        if matched_pos:
                            logging.debug(f"[PROFIT_CHECK] matched_pos found: {matched_pos.order_id}, profit={matched_pos.realized_profit}")
                            if matched_pos.realized_profit is not None:
                                match[f"{strategy_id}_Profit"] = matched_pos.realized_profit
                                logging.debug(f"[PROFIT_LOGGED] Profit logged: {matched_pos.realized_profit} @ {t}")
                            else:
                                logging.warning(f"[PROFIT_MISSING] matched_pos {matched_pos.order_id} has no realized_profit")
                        else:
                            logging.warning(f"[PROFIT_MISSING] No position matched for order {exec_order.order_id}")

                        if matched_pos and matched_pos.realized_profit is not None:
                            match[f"{strategy_id}_Profit"] = matched_pos.realized_profit

            # 発注登録（newie → settlement の順に登録）
            seen = set()
            t4 = time.perf_counter()

            # ✅ 先に newie を登録して Position を確保
            for order in new_orders:
                if order.order_id in seen:
                    logging.warning(f"[DUPLICATE] 同じ order_id が複数登録されようとしています: {order.order_id}")
                seen.add(order.order_id)

                if order.order_effect == "newie":
                    state['order_book'].add_order(order, state['positions'])

                    logging.debug(
                        f"[ORDER ISSUED] {order.order_id} | strategy={order.strategy_id} | "
                        f"{order.order_effect.upper()} {order.order_side} | "
                        f"price={order.order_price}, type={order.order_type}, time={order.order_time}"
                    )

                    kind = "New"
                    side = "Buy" if order.order_side == "BUY" else "Sell"
                    log_entry[f"{side}_{kind}_OrderID"] = order.order_id
                    log_entry[f"{side}_{kind}_OrderTime"] = order.order_time
                    log_entry[f"{side}_{kind}_OrderPrice"] = order.order_price

            # ✅ 次に settlement を登録（positions に依存するため）
            for order in new_orders:
                if order.order_effect == "settlement":
                    state['order_book'].add_order(order, state['positions'])

                    logging.debug(
                        f"[ORDER ISSUED] {order.order_id} | strategy={order.strategy_id} | "
                        f"{order.order_effect.upper()} {order.order_side} | "
                        f"price={order.order_price}, type={order.order_type}, time={order.order_time}"
                    )

                    kind = order.order_category
                    side = "Buy" if order.order_side == "BUY" else "Sell"
                    log_entry[f"{side}_{kind}_OrderID"] = order.order_id
                    log_entry[f"{side}_{kind}_OrderTime"] = order.order_time
                    log_entry[f"{side}_{kind}_OrderPrice"] = order.order_price

            t5 = time.perf_counter()
            logging.debug(f"[TIME] {strategy_id} 注文登録: {t5 - t4:.4f} 秒")

        # ✅ キャンセル注文をログに記録
        for canceled_order in state['order_book'].canceled_orders:
            t = pd.to_datetime(canceled_order.cancel_time).floor("T")
            match = next((r for r in state['log'] if r["Date"] == t), None)
            if match is not None:
                if canceled_order.order_effect == "newie":
                    kind = "New"
                elif canceled_order.order_effect == "settlement":
                    kind = canceled_order.order_category

                side = "Buy" if canceled_order.order_side == "BUY" else "Sell"
                match[f"{side}_{kind}_CancelID"] = canceled_order.order_id
                match[f"{side}_{kind}_CancelTime"] = canceled_order.cancel_time

        lookback_ohlc_buffer.append(current_ohlc)

    # 最終補完
    dummy = {
        'time': ohlc_list[-1].time + pd.Timedelta(minutes=1),
        'open': ohlc_list[-1].close,
        'high': ohlc_list[-1].close,
        'low': ohlc_list[-1].close,
        'close': ohlc_list[-1].close
    }
    state['order_book'].match_orders(dummy, state['positions'], len(ohlc_list), time_index_map)

    # DataFrame化と後処理
    df_result = pd.DataFrame(state['log'])
    df_result["Date"] = pd.to_datetime(df_result["Date"])
    df_result.set_index("Date", inplace=True)
    #df_result = apply_execution_prices(df_result, build_orderbook_price_map(state['order_book']), strategy_id)
    df_result = apply_statistics(df_result)
    df_result.columns = [
                            col if col.startswith(f"{strategy_id}_") else f"{strategy_id}_{col}"
                            for col in df_result.columns
                        ]

    end_time_total = time.perf_counter()

    logging.debug(f"[TIME] {strategy_id} simulate_strategy 総時間: {end_time_total - start_time_total:.2f} 秒")

    return df_result

def run_multi_strategy_simulation(df, strategies):
    ohlc_list = [OHLC(row.Date, row.Open, row.High, row.Low, row.Close)
                 for row in df.itertuples(index=False)]

    with ThreadPoolExecutor() as executor:
        futures = {
            strategy_id: executor.submit(simulate_strategy, strategy_id, strategy_func, ohlc_list)
            for strategy_id, strategy_func in strategies.items()
        }

        result_dfs = {sid: f.result() for sid, f in futures.items()}

    # Dateをキーにしてマージ
    combined = df[["Date"]].copy().set_index("Date")
    for result_df in result_dfs.values():
        combined = combined.join(result_df, how="left")

    return combined.reset_index()

# --- 戦略を読み込む関数 ---
def load_strategies():
    strategies = {}
    for filepath in glob.glob("Rule_*.py"):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strategies[module_name] = module.run
    return strategies

# --- Strategy Metrics Applier ---
def apply_statistics(result_df: pd.DataFrame) -> pd.DataFrame:
    calc = TradeStatisticsCalculator()
    result_df = result_df.copy()

    # 戦略名（prefix）をすべて抽出
    profit_cols = [col for col in result_df.columns if col.endswith("_Profit")]

    for profit_col in profit_cols:
        prefix = profit_col.replace("_Profit", "")
        profits = result_df[profit_col].fillna(0.0).tolist()

        result_df[f"{prefix}_TotalProfit"] = calc.total_profit(profits)
        result_df[f"{prefix}_WinningRate"] = calc.winning_rate(profits)
        result_df[f"{prefix}_PayoffRatio"] = calc.payoff_ratio(profits)
        result_df[f"{prefix}_ExpectedValue"] = calc.expected_value(
            result_df[f"{prefix}_WinningRate"],
            result_df[f"{prefix}_PayoffRatio"]
        )
        result_df[f"{prefix}_DrawDown"] = calc.drawdown(
            result_df[f"{prefix}_TotalProfit"]
        )
        result_df[f"{prefix}_MaxDrawDown"] = calc.max_drawdown(
            result_df[f"{prefix}_DrawDown"]
        )

    return result_df

def get_fixed_column_order(strategy_id: str) -> list:
    base = ['Date', 'Open', 'High', 'Low', 'Close']
    exec_columns_flat = [
        f"{strategy_id}_Buy_New_OrderID", f"{strategy_id}_Buy_New_OrderTime", f"{strategy_id}_Buy_New_OrderPrice",
        f"{strategy_id}_Buy_New_ExecID", f"{strategy_id}_Buy_New_ExecTime", f"{strategy_id}_Buy_New_ExecPrice",

        f"{strategy_id}_Buy_Settlement_OrderID", f"{strategy_id}_Buy_Settlement_OrderTime", f"{strategy_id}_Buy_Settlement_OrderPrice",
        f"{strategy_id}_Buy_Settlement_ExecID", f"{strategy_id}_Buy_Settlement_ExecTime", f"{strategy_id}_Buy_Settlement_ExecPrice",

        f"{strategy_id}_Buy_Stop_OrderID", f"{strategy_id}_Buy_Stop_OrderTime", f"{strategy_id}_Buy_Stop_OrderPrice",
        f"{strategy_id}_Buy_Stop_ExecID", f"{strategy_id}_Buy_Stop_ExecTime", f"{strategy_id}_Buy_Stop_ExecPrice",

        f"{strategy_id}_Buy_Profitfixed_OrderID", f"{strategy_id}_Buy_Profitfixed_OrderTime", f"{strategy_id}_Buy_Profitfixed_OrderPrice",
        f"{strategy_id}_Buy_Profitfixed_ExecID", f"{strategy_id}_Buy_Profitfixed_ExecTime", f"{strategy_id}_Buy_Profitfixed_ExecPrice",

        f"{strategy_id}_Sell_New_OrderID", f"{strategy_id}_Sell_New_OrderTime", f"{strategy_id}_Sell_New_OrderPrice",
        f"{strategy_id}_Sell_New_ExecID", f"{strategy_id}_Sell_New_ExecTime", f"{strategy_id}_Sell_New_ExecPrice",

        f"{strategy_id}_Sell_Settlement_OrderID", f"{strategy_id}_Sell_Settlement_OrderTime", f"{strategy_id}_Sell_Settlement_OrderPrice",
        f"{strategy_id}_Sell_Settlement_ExecID", f"{strategy_id}_Sell_Settlement_ExecTime", f"{strategy_id}_Sell_Settlement_ExecPrice",

        f"{strategy_id}_Sell_Stop_OrderID", f"{strategy_id}_Sell_Stop_OrderTime", f"{strategy_id}_Sell_Stop_OrderPrice",
        f"{strategy_id}_Sell_Stop_ExecID", f"{strategy_id}_Sell_Stop_ExecTime", f"{strategy_id}_Sell_Stop_ExecPrice",

        f"{strategy_id}_Sell_Profitfixed_OrderID", f"{strategy_id}_Sell_Profitfixed_OrderTime", f"{strategy_id}_Sell_Profitfixed_OrderPrice",
        f"{strategy_id}_Sell_Profitfixed_ExecID", f"{strategy_id}_Sell_Profitfixed_ExecTime", f"{strategy_id}_Sell_Profitfixed_ExecPrice",

        f"{strategy_id}_Buy_Stop_CancelID", f"{strategy_id}_Buy_Stop_CancelTime",
        f"{strategy_id}_Buy_Profitfixed_CancelID", f"{strategy_id}_Buy_Profitfixed_CancelTime",

        f"{strategy_id}_Sell_Stop_CancelID", f"{strategy_id}_Sell_Stop_CancelTime",
        f"{strategy_id}_Sell_Profitfixed_CancelID", f"{strategy_id}_Sell_Profitfixed_CancelTime",

        f"{strategy_id}_Profit", f"{strategy_id}_TotalProfit", f"{strategy_id}_WinningRate",
        f"{strategy_id}_PayoffRatio", f"{strategy_id}_ExpectedValue", f"{strategy_id}_DrawDown", f"{strategy_id}_MaxDrawDown"
    ]
    return base + exec_columns_flat

def apply_execution_prices(result: pd.DataFrame, orderbook_dict: dict, strategy_id: str) -> pd.DataFrame:
    result = result.copy()
    result[f"{strategy_id}_Profit"] = 0.0
    result["ExecMatchKey"] = result.index.floor("T")

    used_pairs = set()

    # ✅ 約定ID列のパターン（Profitfixedを含む）
    exec_columns = [
        (f"{strategy_id}_Buy_New_ExecID", f"{strategy_id}_Buy_New_ExecTime", f"{strategy_id}_Buy_New_ExecPrice"),
        (f"{strategy_id}_Buy_Settlement_ExecID", f"{strategy_id}_Buy_Settlement_ExecTime", f"{strategy_id}_Buy_Settlement_ExecPrice"),
        (f"{strategy_id}_Buy_Stop_ExecID", f"{strategy_id}_Buy_Stop_ExecTime", f"{strategy_id}_Buy_Stop_ExecPrice"),
        (f"{strategy_id}_Buy_Profitfixed_ExecID", f"{strategy_id}_Buy_Profitfixed_ExecTime", f"{strategy_id}_Buy_Profitfixed_ExecPrice"),

        (f"{strategy_id}_Sell_New_ExecID", f"{strategy_id}_Sell_New_ExecTime", f"{strategy_id}_Sell_New_ExecPrice"),
        (f"{strategy_id}_Sell_Settlement_ExecID", f"{strategy_id}_Sell_Settlement_ExecTime", f"{strategy_id}_Sell_Settlement_ExecPrice"),
        (f"{strategy_id}_Sell_Stop_ExecID", f"{strategy_id}_Sell_Stop_ExecTime", f"{strategy_id}_Sell_Stop_ExecPrice"),
        (f"{strategy_id}_Sell_Profitfixed_ExecID", f"{strategy_id}_Sell_Profitfixed_ExecTime", f"{strategy_id}_Sell_Profitfixed_ExecPrice"),
    ]

    for exec_id_col, time_col, price_col in exec_columns:
        for idx in result.index:
            exec_id = result.at[idx, exec_id_col] if exec_id_col in result.columns else None
            if pd.notna(exec_id) and exec_id in orderbook_dict:
                order = orderbook_dict[exec_id]
                exec_time_floor = pd.to_datetime(order.execution_time).floor("T")
                match_rows = result[result["ExecMatchKey"] == exec_time_floor]
                if not match_rows.empty:
                    mi = match_rows.index[0]
                    result.at[mi, time_col] = order.execution_time
                    result.at[mi, price_col] = order.execution_price

    # ✅ 損益計算
    for idx in result.index:
        for suffix in ["Settlement", "Stop", "Profitfixed"]:
            for side in ["Buy", "Sell"]:
                exit_oid_col = f"{strategy_id}_{side}_{suffix}_ExecID"
                if exit_oid_col not in result.columns:
                    continue
                exit_oid = result.at[idx, exit_oid_col]
                if pd.isna(exit_oid):
                    continue

                exit_order = orderbook_dict.get(exit_oid)
                if not exit_order:
                    continue

                # ✅ Exit Order に対応する Entry Order を推定
                if exit_oid.endswith("_stop"):
                    entry_oid = exit_oid.replace("_stop", "_newie")
                elif exit_oid.endswith("_settlement"):
                    entry_oid = exit_oid.replace("_settlement", "_newie")
                elif exit_oid.endswith("_profitfixed"):
                    entry_oid = exit_oid.replace("_profitfixed", "_newie")
                else:
                    continue

                entry_order = orderbook_dict.get(entry_oid)
                if not entry_order:
                    continue

                pair_key = (entry_order.order_id, exit_order.order_id)
                if pair_key in used_pairs:
                    continue

                if entry_order.execution_price is not None and exit_order.execution_price is not None:
                    if entry_order.order_side == 'BUY':
                        profit = exit_order.execution_price - entry_order.execution_price
                    else:
                        profit = entry_order.execution_price - exit_order.execution_price

                    result.at[idx, f"{strategy_id}_Profit"] = profit
                    used_pairs.add(pair_key)
                    break  # 1件で十分

    result.drop(columns=["ExecMatchKey"], inplace=True)
    return result


def get_trade_date(now: datetime) -> datetime.date:
    # ナイトセッションは17:00以降で、取引日は翌営業日
    if now.time() >= dtime(17, 0):
        trade_date = now.date() + timedelta(days=1)
    else:
        trade_date = now.date()

    # 営業日に補正（先に進める）
    while trade_date.weekday() >= 5:  # 土曜(5) or 日曜(6)
        trade_date += timedelta(days=1)

    return trade_date

def get_trade_datetime(now: datetime) -> datetime:
    """取引日付＋時刻で比較可能なdatetimeを返す"""
    trade_date = get_trade_date(now)
    return datetime.combine(trade_date, now.time())

def order_list_to_dataframe(order_list: list) -> pd.DataFrame:
    """
    List[Order] → DataFrame に変換する
    """
    if not order_list:
        return pd.DataFrame()

    records = []
    for order in order_list:
        records.append({
            'order_id': order.order_id,
            'strategy_id': order.strategy_id,
            'side': order.side,
            'price': order.price,
            'quantity': order.quantity,
            'order_time': order.order_time,
            'order_type': order.order_type,
            'trigger_price': order.trigger_price,
            'triggered': order.triggered,
            'status': order.status,
            'execution_price': order.execution_price,
            'execution_time': order.execution_time,
            'position_effect': order.position_effect
        })

    return pd.DataFrame(records)
# --- 実行ブロック ---
def main():


    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s][%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("debug_log.txt", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    log_file_path = "debug_log.txt"
    sys.stdout = open(log_file_path, "w", encoding="utf-8")  # 以降すべての print がファイルに出力される

    # 📁 最新ファイル取得
    csv_files = glob.glob(os.path.join("Input_csv", "*.csv"))
    if not csv_files:
        logging.info("Input_csv フォルダに CSV ファイルが見つかりません。")
        return

    latest_file = max(csv_files, key=os.path.getmtime)
    # ✅ 最新ファイルからDateとOHLC列だけ抽出
    df_input = pd.read_csv(latest_file, parse_dates=['Date'])
    base_columns = df_input[['Date', 'Open', 'High', 'Low', 'Close']]

    # ✅ シミュレーション実行
    strategies = load_strategies()
    combined_df = run_multi_strategy_simulation(df_input, strategies)

    # ✅ Dateでの整合を取りつつ横に合体（再index化不要）
    final_df = pd.merge(base_columns, combined_df, on="Date", how="left")

    fixed_columns = get_fixed_column_order("Rule_PromptFollow")
    for col in fixed_columns:
        if col not in final_df.columns:
            final_df[col] = None  # 空欄で追加

    final_df = final_df[fixed_columns]

    # 💾 書き出し（index=False）
    final_df.to_csv("result_stats.csv", index=False)
    logging.info("シミュレーション結果を 'result_stats.csv' に出力しました。")

if __name__ == "__main__":
    main()