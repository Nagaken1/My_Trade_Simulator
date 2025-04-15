import os
import sys
import glob
import pandas as pd
import numpy as np
import importlib.util
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timedelta

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
        self.orders = pd.DataFrame(columns=[
            'order_id','strategy_id','side', 'price', 'quantity', 'order_time', 'order_type',
            'trigger_price', 'triggered', 'status',
            'execution_price', 'execution_time', 'position_effect'
        ])
        self.executed_orders = []  # ✅ 追加：Orderオブジェクトを記録

    def add_order(self, order):
        self.orders.loc[len(self.orders)] = {
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
        }

    def match_orders(self, ohlc, positions_df, current_index=None, time_index_map=None):
        executed = []
        active_orders = self.orders[self.orders['status'] == 'pending']

        for idx, order in active_orders.iterrows():
            executed_flag = False

            order_index = time_index_map.get(order['order_time'], -1)
            if order_index >= current_index:
                continue

            if order['order_type'] == 'market':
                exec_price = ohlc['high'] if order['side'] == 'BUY' else ohlc['low']
                self.orders.loc[idx, ['status', 'execution_price', 'execution_time']] = [
                    'executed', exec_price, ohlc['time']
                ]
                executed_flag = True

            elif order['order_type'] == 'limit':
                if order['position_effect'] == 'open':
                    if (order['side'] == 'BUY' and ohlc['low'] <= order['price']) or \
                       (order['side'] == 'SELL' and ohlc['high'] >= order['price']):
                        self.orders.loc[idx, ['status', 'execution_price', 'execution_time']] = [
                            'executed', order['price'], ohlc['time']
                        ]
                        executed_flag = True
                else:
                    has_opposite = not positions_df[
                        (positions_df['exit_time'].isna()) &
                        (positions_df['side'] != order['side']) &
                        (positions_df['strategy_id'] == order['strategy_id'])
                    ].empty
                    if has_opposite:
                        if (order['side'] == 'BUY' and ohlc['low'] - 5 <= order['price']) or \
                           (order['side'] == 'SELL' and ohlc['high'] + 5 >= order['price']):
                            self.orders.loc[idx, ['status', 'execution_price', 'execution_time']] = [
                                'executed', order['price'], ohlc['time']
                            ]
                            executed_flag = True

            elif order['order_type'] == 'stop':
                triggered = self.orders.at[idx, 'triggered']
                status = self.orders.at[idx, 'status']
                if not triggered:
                    if (order['side'] == 'BUY' and ohlc['high'] >= order['trigger_price']) or \
                       (order['side'] == 'SELL' and ohlc['low'] <= order['trigger_price']):
                        self.orders.at[idx, 'triggered'] = True
                        triggered = True
                if triggered and status == 'pending':
                    exec_price = ohlc['high'] if order['side'] == 'BUY' else ohlc['low']
                    self.orders.loc[idx, ['status', 'execution_price', 'execution_time']] = [
                        'executed', exec_price, ohlc['time']
                    ]
                    executed_flag = True

            if executed_flag:
                exec_order = Order(
                    strategy_id=order['strategy_id'],
                    side=order['side'],
                    price=order['price'],
                    quantity=order['quantity'],
                    order_time=order['order_time'],
                    order_type=order['order_type'],
                    trigger_price=order['trigger_price'],
                    position_effect=order['position_effect']
                )
                exec_order.status = 'executed'
                exec_order.execution_price = self.orders.loc[idx, 'execution_price']
                exec_order.execution_time = self.orders.loc[idx, 'execution_time']
                exec_order.order_id = order['order_id']
                self.executed_orders.append(exec_order)  # ✅ ここで保持
                executed.append(exec_order)

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
    for _, row in order_book.orders.iterrows():
        if row['status'] == 'executed' and pd.notna(row['order_id']):
            order = Order(
                strategy_id=row['strategy_id'],
                side=row['side'],
                price=row['price'],
                quantity=row['quantity'],
                order_time=row['order_time'],
                order_type=row['order_type'],
                trigger_price=row['trigger_price'],
                position_effect=row['position_effect']
            )
            order.status = row['status']
            order.execution_price = row['execution_price']
            order.execution_time = row['execution_time']
            order.order_id = row['order_id']
            order_map[row['order_id']] = order
    return order_map

# --- メイン処理 ---
def simulate_strategy(strategy_id, strategy_func, ohlc_list):
    state = {
        'order_book': OrderBook(),
        'positions_df': pd.DataFrame(columns=[
            'side', 'entry_price', 'quantity', 'entry_time', 'exit_price', 'exit_time', 'strategy_id'
        ]),
        'log': []
    }

    time_index_map = {ohlc.time: i for i, ohlc in enumerate(ohlc_list)}

    for i in range(len(ohlc_list)):
        current_ohlc = ohlc_list[i]

        log_entry = {
            'Date': current_ohlc.time,
            'NewBuy_OrderID': None, 'NewBuy_ExecTime': None, 'NewBuy_ExecPrice': None,
            'NewSell_OrderID': None, 'NewSell_ExecTime': None, 'NewSell_ExecPrice': None,
            'CloseBuy_OrderID': None, 'CloseBuy_ExecTime': None, 'CloseBuy_ExecPrice': None,
            'CloseSell_OrderID': None, 'CloseSell_ExecTime': None, 'CloseSell_ExecPrice': None,
            'StopBuy_OrderID': None, 'StopBuy_ExecTime': None, 'StopBuy_ExecPrice': None,
            'StopSell_OrderID': None, 'StopSell_ExecTime': None, 'StopSell_ExecPrice': None,

            # ✅ 約定専用のログ欄（Profit対象はこちら）
            'NewBuyExec_OrderID': None, 'NewSellExec_OrderID': None,
            'CloseBuyExec_OrderID': None, 'CloseSellExec_OrderID': None,
            'StopBuyExec_OrderID': None, 'StopSellExec_OrderID': None,
        }

        state['log'].append(log_entry)

        # 約定処理（2回目以降のみ）
        if i > 0:
            executed_now = state['order_book'].match_orders(
                vars(current_ohlc),
                state['positions_df'],
                current_index=i,
                time_index_map=time_index_map
            )

            for exec_order in executed_now:
                key_prefix = "New" if exec_order.position_effect == "open" else \
                             "Stop" if exec_order.order_type == "stop" else "Close"
                side_key = "Buy" if exec_order.side == "BUY" else "Sell"
                match_time = pd.to_datetime(exec_order.execution_time).floor("T")

                matched_log = next((row for row in state['log'] if row["Date"] == match_time), None)

                if matched_log is not None:
                    # ✅ 約定情報を Exec 専用フィールドに記録
                    exec_key = f"{key_prefix}{side_key}Exec_OrderID"
                    already_logged = any(row.get(exec_key) == exec_order.order_id for row in state['log'])
                    if not already_logged:
                        matched_log[exec_key] = exec_order.order_id

        # 発注処理（strategy 関数を呼ぶ）
        new_orders = strategy_func(
            current_ohlc=current_ohlc,
            positions_df=state['positions_df'],
            order_history=state['order_book'].orders,
            strategy_id=strategy_id
        )

        for order in new_orders:
            state['order_book'].add_order(order)

            key_prefix = "New" if order.position_effect == "open" else \
                         "Stop" if order.order_type == "stop" else "Close"
            side_key = "Buy" if order.side == "BUY" else "Sell"
            log_entry[f"{key_prefix}{side_key}_OrderID"] = order.order_id

    # ダミー処理
    dummy_ohlc = {
        'time': ohlc_list[-1].time + pd.Timedelta(minutes=1),
        'open': ohlc_list[-1].close,
        'high': ohlc_list[-1].close,
        'low': ohlc_list[-1].close,
        'close': ohlc_list[-1].close
    }
    state['order_book'].match_orders(dummy_ohlc, state['positions_df'], len(ohlc_list), time_index_map)

    # DataFrame化と処理
    df_result = pd.DataFrame(state['log'])
    df_result["Date"] = pd.to_datetime(df_result["Date"])
    df_result.set_index("Date", inplace=True)

    # ✅ Exec系 OrderID のみを対象に Profit計算
    df_result = apply_execution_prices(df_result, build_orderbook_price_map(state['order_book']), strategy_id)
    df_result = apply_statistics(df_result)
    df_result.columns = [f"{strategy_id}_{col}" for col in df_result.columns]

    return df_result

def run_multi_strategy_simulation(df, strategies, orderbook_prices):
    base_df = df[['Date']].copy()
    ohlc_list = [OHLC(row.Date, row.Open, row.High, row.Low, row.Close) for row in df.itertuples(index=False)]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_strategy, strategy_id, strategy_func, ohlc_list)
            for strategy_id, strategy_func in strategies.items()
        ]
        result_dfs = [f.result() for f in futures]

    combined_df = pd.concat([base_df.set_index('Date')] + result_dfs, axis=1).reset_index()
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

    # --- 約定情報を記録する列マッピング
    order_columns = [
        ('NewBuy_OrderID', 'NewBuy_ExecTime', 'NewBuy_ExecPrice'),
        ('NewSell_OrderID', 'NewSell_ExecTime', 'NewSell_ExecPrice'),
        ('CloseBuy_OrderID', 'CloseBuy_ExecTime', 'CloseBuy_ExecPrice'),
        ('CloseSell_OrderID', 'CloseSell_ExecTime', 'CloseSell_ExecPrice'),
        ('StopBuy_OrderID', 'StopBuy_ExecTime', 'StopBuy_ExecPrice'),
        ('StopSell_OrderID', 'StopSell_ExecTime', 'StopSell_ExecPrice'),
    ]

    for oid_col, time_col, price_col in order_columns:
        for idx in result.index:
            order_id = result.at[idx, oid_col] if oid_col in result.columns else None
            if pd.notna(order_id) and order_id in orderbook_dict:
                order = orderbook_dict[order_id]
                exec_time_floor = pd.to_datetime(order.execution_time).floor("T")
                match_rows = result[result["ExecMatchKey"] == exec_time_floor]
                if not match_rows.empty:
                    mi = match_rows.index[0]
                    result.at[mi, time_col] = order.execution_time
                    result.at[mi, price_col] = order.execution_price

    # --- Profit 計算（entry と exit が別行でも対応）
    for idx in result.index:
        for exit_type, exit_oid_col in [('BUY', 'StopSell_OrderID'), ('BUY', 'CloseSell_OrderID'),
                                        ('SELL', 'StopBuy_OrderID'), ('SELL', 'CloseBuy_OrderID')]:
            exit_oid = result.at[idx, exit_oid_col] if exit_oid_col in result.columns else None
            if pd.isna(exit_oid):
                continue

            # ✅ Exitが _close 付きなら対応するEntry IDに変換
            if isinstance(exit_oid, str) and exit_oid.endswith('_close'):
                entry_oid = exit_oid.replace('_close', '')
            else:
                continue

            entry_order = orderbook_dict.get(entry_oid)
            exit_order = orderbook_dict.get(exit_oid)

            # ✅ すでにProfitが記録されていればスキップ（2重書き込み防止）
            if result.at[idx, "Profit"] != 0.0:
                continue

            if entry_order and exit_order and entry_order.execution_price is not None and exit_order.execution_price is not None:
                if exit_type == 'BUY':
                    profit = exit_order.execution_price - entry_order.execution_price
                else:
                    profit = entry_order.execution_price - exit_order.execution_price

                result.at[idx, "Profit"] = profit
                break  # ✅ 1組見つけたら終了

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