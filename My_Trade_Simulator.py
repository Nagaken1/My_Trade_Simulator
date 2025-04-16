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
import importlib

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
    def __init__(self, side, price, quantity, order_time, order_type='limit', trigger_price=None,position_effect='open', strategy_id='default'):
        self.strategy_id = strategy_id
        self.side = side  # 'BUY' または 'SELL'
        self.price = price  # 注文価格
        self.quantity = quantity  # 注文数量
        self.order_time = order_time  # 注文時刻
        self.order_type = order_type  # 'limit', 'market', または 'stop'
        self.trigger_price = trigger_price  # 逆指値注文の発動価格（order_type が 'stop' の場合）
        self.triggered = False  # 逆指値注文が発動されたかどうか
        self.status = 'pending'  # 'pending' または 'executed'
        self.execution_price = None  # 約定価格
        self.execution_time = None  # 約定時刻
        self.position_effect = position_effect  # 'open'（新規）または 'close'（決済）

# --- Position クラス ---
class Position:
    def __init__(self, side, price, quantity, entry_time, exit_time=None, exit_price=None):
        self.side = side              # 'BUY' または 'SELL'
        self.price = price            # エントリー価格
        self.quantity = quantity      # 保有数量
        self.entry_time = entry_time  # 建玉作成時間（datetime）

        self.exit_time = exit_time    # 決済時間（まだ決済していなければ None）
        self.exit_price = exit_price  # 決済価格（同上）

    def is_closed(self):
        """ポジションが決済済みかどうかを返す"""
        return self.exit_time is not None

    def profit(self):
        """ポジションが決済されていれば損益を返す、されていなければ0"""
        if not self.is_closed():
            return 0
        if self.side == 'BUY':
            return (self.exit_price - self.price) * self.quantity
        else:  # 'SELL'
            return (self.price - self.exit_price) * self.quantity

# --- OrderBook クラス ---
class OrderBook:
    def __init__(self):
        self.orders: list[Order] = []             # 全注文（未約定・約定を含む）
        self.executed_orders: list[Order] = []    # 約定済みのみ

    def add_order(self, order: Order):
        self.orders.append(order)

    def match_orders(self, ohlc, positions: list, current_index=None, time_index_map=None):
        executed = []

        for order in self.orders:
            if order.status != 'pending':
                continue

            order_index = time_index_map.get(order.order_time, -1)
            if order_index >= current_index:
                continue

            executed_flag = False

            # 成行注文
            if order.order_type == 'market':
                order.execution_price = ohlc['high'] if order.side == 'BUY' else ohlc['low']
                order.execution_time = ohlc['time']
                order.status = 'executed'
                executed_flag = True

            # 指値注文
            elif order.order_type == 'limit':
                if order.position_effect == 'open':
                    if (order.side == 'BUY' and ohlc['low'] <= order.price) or \
                       (order.side == 'SELL' and ohlc['high'] >= order.price):
                        order.execution_price = order.price
                        order.execution_time = ohlc['time']
                        order.status = 'executed'
                        executed_flag = True
                else:
                    has_opposite = any(
                        not p.is_closed() and p.side != order.side and p.entry_time <= order.order_time
                        for p in positions
                    )
                    if has_opposite:
                        if (order.side == 'BUY' and ohlc['low'] - 5 <= order.price) or \
                           (order.side == 'SELL' and ohlc['high'] + 5 >= order.price):
                            order.execution_price = order.price
                            order.execution_time = ohlc['time']
                            order.status = 'executed'
                            executed_flag = True

            # ストップ注文（逆指値）
            elif order.order_type == 'stop':
                if not order.triggered:
                    if (order.side == 'BUY' and ohlc['high'] >= order.trigger_price) or \
                       (order.side == 'SELL' and ohlc['low'] <= order.trigger_price):
                        order.triggered = True

                if order.triggered:
                    order.execution_price = ohlc['high'] if order.side == 'BUY' else ohlc['low']
                    order.execution_time = ohlc['time']
                    order.status = 'executed'
                    executed_flag = True

            if executed_flag:
                self.executed_orders.append(order)
                executed.append(order)

        return executed

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
    import time
    import importlib
    start_time_total = time.perf_counter()

    state = {
        'order_book': OrderBook(),
        'positions': [],
        'log': []
    }

    strategy_module = importlib.import_module(strategy_func.__module__)
    lookback_minutes = getattr(strategy_module, "LOOKBACK_MINUTES", 0)
    time_index_map = {ohlc.time: i for i, ohlc in enumerate(ohlc_list)}

    for i, current_ohlc in enumerate(ohlc_list):
        if i >= lookback_minutes:
            hist = [vars(ohlc_list[j]) for j in range(i - lookback_minutes, i)]
            ohlc_df_history = pd.DataFrame(hist)
        else:
            ohlc_df_history = pd.DataFrame()

        log_entry = {"Date": current_ohlc.time}
        state['log'].append(log_entry)

        # 戦略実行
        order_history_df = order_list_to_dataframe(state['order_book'].orders)
        try:
            new_orders, support, resistance = strategy_func(
                current_ohlc=current_ohlc,
                positions=state['positions'],
                order_history=order_history_df,
                strategy_id=strategy_id,
                ohlc_history=ohlc_df_history
            )
        except ValueError:
            new_orders = strategy_func(
                current_ohlc=current_ohlc,
                positions=state['positions'],
                order_history=order_history_df,
                strategy_id=strategy_id,
                ohlc_history=ohlc_df_history
            )
            support = resistance = None

        log_entry["SupportLine"] = support
        log_entry["ResistanceLine"] = resistance

        if i > 0:
            executed_now = state['order_book'].match_orders(
                vars(current_ohlc),
                state['positions'],
                current_index=i,
                time_index_map=time_index_map
            )
            for exec_order in executed_now:
                kind = "New" if exec_order.position_effect == "open" else \
                       "Stop" if exec_order.order_type == "stop" else "Close"
                side = "Buy" if exec_order.side == "BUY" else "Sell"
                t = pd.to_datetime(exec_order.execution_time).floor("T")
                match = next((r for r in state['log'] if r["Date"] == t), None)
                if match:
                    match[f"{side}_{kind}_ExecID"] = exec_order.order_id
                    match[f"{side}_{kind}_ExecTime"] = exec_order.execution_time
                    match[f"{side}_{kind}_ExecPrice"] = exec_order.execution_price

        # 発注登録
        for order in new_orders:
            state['order_book'].add_order(order)
            kind = "New" if order.position_effect == "open" else \
                   "Stop" if order.order_type == "stop" else "Close"
            side = "Buy" if order.side == "BUY" else "Sell"
            log_entry[f"{side}_{kind}_OrderID"] = order.order_id
            log_entry[f"{side}_{kind}_OrderTime"] = order.order_time
            log_entry[f"{side}_{kind}_OrderPrice"] = order.price

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
    df_result = apply_execution_prices(df_result, build_orderbook_price_map(state['order_book']), strategy_id)
    df_result = apply_statistics(df_result)
    df_result.columns = [f"{strategy_id}_{col}" for col in df_result.columns]

    end_time_total = time.perf_counter()
    print(f"[TIME] {strategy_id} simulate_strategy: {end_time_total - start_time_total:.2f} sec")

    return df_result


def run_multi_strategy_simulation(df, strategies, orderbook_prices=None):

    base_df = df[['Date']].copy()
    ohlc_list = [OHLC(row.Date, row.Open, row.High, row.Low, row.Close)
                 for row in df.itertuples(index=False)]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_strategy, strategy_id, strategy_func, ohlc_list)
            for strategy_id, strategy_func in strategies.items()
        ]
        result_dfs = [f.result() for f in futures]

    combined_df = pd.concat([base_df.set_index("Date")] + result_dfs, axis=1).reset_index()
    return combined_df

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
    profits = result_df['Profit'].tolist()
    result_df['TotalProfit'] = calc.total_profit(profits)
    result_df['WinningRate'] = calc.winning_rate(profits)
    result_df['PayoffRatio'] = calc.payoff_ratio(profits)
    result_df['ExpectedValue'] = calc.expected_value(result_df['WinningRate'], result_df['PayoffRatio'])
    result_df['DrawDown'] = calc.drawdown(result_df['TotalProfit'])
    result_df['MaxDrawDown'] = calc.max_drawdown(result_df['DrawDown'])
    return result_df

def apply_execution_prices(result: pd.DataFrame, orderbook_dict: dict, strategy_id: str) -> pd.DataFrame:
    result = result.copy()
    result["Profit"] = 0.0
    result["ExecMatchKey"] = result.index.floor("T")

    used_pairs = set()

    # 約定情報を記録する対象（ExecID → 時刻・価格を書き込むカラム）
    exec_columns = [
        ("Buy_New_ExecID", "Buy_New_ExecTime", "Buy_New_ExecPrice"),
        ("Buy_Close_ExecID", "Buy_Close_ExecTime", "Buy_Close_ExecPrice"),
        ("Buy_Stop_ExecID", "Buy_Stop_ExecTime", "Buy_Stop_ExecPrice"),
        ("Sell_New_ExecID", "Sell_New_ExecTime", "Sell_New_ExecPrice"),
        ("Sell_Close_ExecID", "Sell_Close_ExecTime", "Sell_Close_ExecPrice"),
        ("Sell_Stop_ExecID", "Sell_Stop_ExecTime", "Sell_Stop_ExecPrice"),
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

    # --- Profit 計算対象の Exit ExecID 列（BuyのExitはClose or Stop、Sellも同様）
    exit_exec_columns = [
        ("Buy", "Close"), ("Buy", "Stop"),
        ("Sell", "Close"), ("Sell", "Stop")
    ]

    for idx in result.index:
        for side, kind in exit_exec_columns:
            exit_oid_col = f"{side}_{kind}_ExecID"
            exit_oid = result.at[idx, exit_oid_col] if exit_oid_col in result.columns else None
            if pd.isna(exit_oid):
                continue

            exit_order = orderbook_dict.get(exit_oid)
            if not exit_order:
                continue

            # ✅ Exit Order ID から Entry ID を明示的に逆算
            entry_oid = exit_oid.replace("_close", "")
            entry_order = orderbook_dict.get(entry_oid)
            if not entry_order:
                continue

            pair_key = (entry_order.order_id, exit_order.order_id)
            if pair_key in used_pairs:
                continue

            if entry_order.execution_price is not None and exit_order.execution_price is not None:
                if entry_order.side == 'BUY':
                    profit = exit_order.execution_price - entry_order.execution_price
                else:
                    profit = entry_order.execution_price - exit_order.execution_price

                result.at[idx, "Profit"] = profit
                used_pairs.add(pair_key)
                break

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
    List[Order] → DataFrame 変換（vars()ベース）
    """
    return pd.DataFrame([vars(o) for o in order_list])

# --- 実行ブロック ---
def main():


    log_file_path = "debug_log.txt"
    sys.stdout = open(log_file_path, "w", encoding="utf-8")  # 以降すべての print がファイルに出力される

    # 📁 最新ファイル取得
    csv_files = glob.glob(os.path.join("Input_csv", "*.csv"))
    if not csv_files:
        print("Input_csv フォルダに CSV ファイルが見つかりません。")
        return

    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"最新のファイルを読み込みます: {latest_file}")
    df = pd.read_csv(latest_file, parse_dates=['Date'])

    # 🔄 戦略の読み込みと実行
    strategies = load_strategies()
    combined_df = run_multi_strategy_simulation(df, strategies, orderbook_prices={})

    # ✅ 日付とOHLCを input から復元（時刻を完全に維持）
    df_input = pd.read_csv(latest_file, parse_dates=['Date'])
    date_series = df_input['Date']
    ohlc_df = df_input[['Open', 'High', 'Low', 'Close']].reset_index(drop=True)

    combined_df.reset_index(drop=True, inplace=True)

    # 💾 Date, OHLC列を先頭に挿入
    final_df = pd.concat([date_series, ohlc_df, combined_df], axis=1)
    final_df.to_csv("result_stats.csv", index=False)

    print("シミュレーション結果を 'result_stats.csv' に出力しました。")

if __name__ == "__main__":
    main()