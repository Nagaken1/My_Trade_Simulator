import os
import glob
import pandas as pd

# --- OHLC クラス ---
class OHLC:
    def __init__(self, time, open_, high, low, close):
        self.time = time      # datetime オブジェクト（例: 2025-04-11 09:01:00）
        self.open = open_     # 始値（float）
        self.high = high      # 高値（float）
        self.low = low        # 安値（float）
        self.close = close    # 終値（float）

# --- Position クラス ---
class Order:
    def __init__(self, side, price, quantity, order_time, order_type='limit', trigger_price=None, position_effect='open'):
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
            'side', 'price', 'quantity', 'order_time', 'order_type',
            'trigger_price', 'triggered', 'status',
            'execution_price', 'execution_time', 'position_effect'
        ])

    def add_order(self, order):
        self.orders.loc[len(self.orders)] = {
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

    def match_orders(self, ohlc, positions_df):
        """OHLCに対して注文を評価し、約定注文のリストを返す"""
        executed = []

        # 対象時刻の未約定注文のみ抽出
        active_orders = self.orders[
            (self.orders['status'] == 'pending') &
            (self.orders['order_time'] == ohlc['time'])
        ]

        for idx, order in active_orders.iterrows():
            executed_flag = False

            if order['order_type'] == 'market':
                self.orders.at[idx, 'status'] = 'executed'
                self.orders.at[idx, 'execution_price'] = ohlc['close']
                self.orders.at[idx, 'execution_time'] = ohlc['time']
                executed_flag = True

            elif order['order_type'] == 'limit':
                if order['position_effect'] == 'open':
                    if (order['side'] == 'BUY' and ohlc['low'] <= order['price']) or \
                    (order['side'] == 'SELL' and ohlc['high'] >= order['price']):
                        self.orders.at[idx, 'status'] = 'executed'
                        self.orders.at[idx, 'execution_price'] = order['price']
                        self.orders.at[idx, 'execution_time'] = ohlc['time']
                        executed_flag = True
                else:  # position_effect == 'close'
                    has_opposite = not positions_df[
                        (positions_df['exit_time'].isna()) &
                        (positions_df['side'] != order['side'])
                    ].empty
                    if has_opposite:
                        if (order['side'] == 'BUY' and ohlc['low'] - 5 <= order['price']) or \
                        (order['side'] == 'SELL' and ohlc['high'] + 5 >= order['price']):
                            self.orders.at[idx, 'status'] = 'executed'
                            self.orders.at[idx, 'execution_price'] = order['price']
                            self.orders.at[idx, 'execution_time'] = ohlc['time']
                            executed_flag = True

            elif order['order_type'] == 'stop':
                if not order['triggered']:
                    if (order['side'] == 'BUY' and ohlc['high'] >= order['trigger_price']) or \
                    (order['side'] == 'SELL' and ohlc['low'] <= order['trigger_price']):
                        self.orders.at[idx, 'triggered'] = True
                        self.orders.at[idx, 'status'] = 'executed'
                        self.orders.at[idx, 'execution_price'] = ohlc['close']
                        self.orders.at[idx, 'execution_time'] = ohlc['time']
                        executed_flag = True

            # 約定したら、元の Order オブジェクトとして返す
            if executed_flag:
                executed.append(Order(
                    side=order['side'],
                    price=order['price'],
                    quantity=order['quantity'],
                    order_time=order['order_time'],
                    order_type=order['order_type'],
                    trigger_price=order['trigger_price'],
                    position_effect=order['position_effect']
                ))

        return executed

    def _is_settlement(self, order, positions):#注文が現在のポジションの反対方向である場合、決済注文と判断
        for pos in positions:
            if not pos.is_closed() and pos.side != order.side:
                return True
        return False

    def _can_execute_new(self, order, ohlc):
        if order.side == 'BUY':
            return ohlc['low'] <= order.price
        else:
            return ohlc['high'] >= order.price

    def _can_execute_settlement(self, order, ohlc):
        if order.side == 'BUY':
            return ohlc['low'] - 5 <= order.price
        else:
            return ohlc['high'] + 5 >= order.price

    def _should_trigger(self, order, ohlc):
        if order.side == 'BUY':
            return ohlc['high'] >= order.trigger_price
        else:
            return ohlc['low'] <= order.trigger_price

# --- 統計計算クラス ---
class TradeStatisticsCalculator:
    @staticmethod
    def total_profit(profit_array):
        """累計損益（TotalProfit）の推移"""
        total = 0
        output_array = []
        for p in profit_array:
            total += p
            output_array.append(total)
        return output_array

    @staticmethod
    def winning_rate(profit_array):
        """勝率（勝ちトレード数 ÷ トレード数）"""
        win_count = 0
        trade_count = 0
        output_array = []

        for i, profit in enumerate(profit_array):
            if profit > 0:
                win_count += 1
            if profit != 0:
                trade_count += 1

            if trade_count != 0:
                output_array.append(round(win_count / trade_count, 4))
            else:
                output_array.append(output_array[i-1] if i > 0 else 0)

        return output_array

    @staticmethod
    def payoff_ratio(profit_array):
        """ペイオフレシオ（平均利益 ÷ 平均損失）"""
        total_win_profit = 0
        total_lose_profit = 0
        win_count = 0
        lose_count = 0
        output_array = []

        for profit in profit_array:
            if profit > 0:
                total_win_profit += profit
                win_count += 1
            elif profit < 0:
                total_lose_profit += profit
                lose_count += 1

            if win_count != 0 and lose_count != 0:
                payoff = (total_win_profit / win_count) / abs(total_lose_profit / lose_count)
                output_array.append(round(payoff, 4))
            else:
                output_array.append(0)

        return output_array

    @staticmethod
    def expected_value(winning_rate_array, payoff_ratio_array):
        """期待値（勝率×ペイオフレシオ − 負け率）"""
        output_array = []
        for w, p in zip(winning_rate_array, payoff_ratio_array):
            ev = w * p - (1 - w)
            output_array.append(round(ev, 4))
        return output_array

    @staticmethod
    def draw_down(profit_array):
        """ドローダウン（最大益からの下落）"""
        output_array = []
        for i in range(len(profit_array)):
            draw_down = max(profit_array[:i+1]) - profit_array[i]
            output_array.append(draw_down)
        return output_array

    @staticmethod
    def max_draw_down(draw_down_array):
        """最大ドローダウンの推移"""
        output_array = []
        max_dd = 0
        for dd in draw_down_array:
            max_dd = max(max_dd, dd)
            output_array.append(max_dd)
        return output_array

# --- メイン処理 ---
def run_simulation_with_stats(df):

    positions_df = pd.DataFrame(columns=[
    'side', 'entry_price', 'quantity', 'entry_time', 'exit_price', 'exit_time'
    ])

    order_book = OrderBook()

    # 戦略による注文発行（order_bookに注文が溜まる）
    sample_strategy(df, order_book, positions_df)

    # OHLC処理（1行ずつ処理）
    for row in df.itertuples(index=False):
        ohlc = {
            'time': row.Date,
            'open': row.Open,
            'high': row.High,
            'low': row.Low,
            'close': row.Close
        }

        # 約定チェック
        executed_orders = order_book.match_orders(ohlc, positions_df)

        for order in executed_orders:
            if order.position_effect == 'open':
                # 新規建玉をDataFrameに追加
                positions_df.loc[len(positions_df)] = {
                    'side': order.side,
                    'entry_price': order.execution_price,
                    'quantity': order.quantity,
                    'entry_time': order.execution_time,
                    'exit_price': None,
                    'exit_time': None
                }
            elif order.position_effect == 'close':
                # 未決済 & 反対側のポジション1つ取得
                mask = (positions_df['exit_time'].isna()) & (positions_df['side'] != order.side)
                idx = positions_df[mask].index.min()

                if pd.notna(idx):
                    positions_df.at[idx, 'exit_price'] = order.execution_price
                    positions_df.at[idx, 'exit_time'] = order.execution_time


    # 決済済みポジションの損益配列を作成
    closed = positions_df.dropna(subset=['exit_price'])

    # Profit計算
    closed['Profit'] = closed.apply(
        lambda row: (row['exit_price'] - row['entry_price']) * row['quantity']
        if row['side'] == 'BUY'
        else (row['entry_price'] - row['exit_price']) * row['quantity'],
        axis=1
    )

    profits = closed['Profit'].tolist()

    # 統計指標の計算
    calc = TradeStatisticsCalculator()
    total_profit = calc.total_profit(profits)
    win_rate = calc.winning_rate(profits)
    payoff = calc.payoff_ratio(profits)
    expected = calc.expected_value(win_rate, payoff)
    dd = calc.draw_down(profits)
    max_dd = calc.max_draw_down(dd)

    # 結果をDataFrameにまとめて返す
    df_out = pd.DataFrame({
        'EntryTime': closed['entry_time'],
        'ExitTime': closed['exit_time'],
        'Side': closed['side'],
        'Profit': profits,
        'TotalProfit': total_profit,
        'WinningRate': win_rate,
        'PayoffRatio': payoff,
        'ExpectedValue': expected,
        'DrawDown': dd,
        'MaxDrawDown': max_dd
    })

    return df_out


# --- 戦略---
def sample_strategy(df, order_book, positions):
    """
    毎分 BUY → 2分後に SELL 決済 という確実に利益/損益が発生する単純戦略
    全て成行注文で確実に約定させる
    """
    for i in range(len(df)):
        current = df.iloc[i]

        # 新規建玉
        order = Order(
            side='BUY',
            price=current['Close'],
            quantity=1,
            order_time=current['Date'],
            order_type='market',
            position_effect='open'
        )
        order_book.add_order(order)

        if i % 100 == 0:
            print(f"[INFO] Order発行中: {current['Date']}")

        # 2分後に決済（SELL）
        if i + 2 < len(df):
            close = df.iloc[i + 2]
            order_close = Order(
                side='SELL',
                price=close['Close'] - 5 ,
                quantity=1,
                order_time=close['Date'],
                order_type='market',
                position_effect='close'
            )
            order_book.add_order(order_close)
            print(f"[CLOSE] SELL @ {close['Date']}")

# --- 実行 ---
if __name__ == '__main__':
    # 📁 Input_csv フォルダ内の最新の CSV ファイルを取得
    csv_files = glob.glob(os.path.join("Input_csv", "*.csv"))
    if not csv_files:
        print("Input_csv フォルダに CSV ファイルが見つかりません。")
        exit()

    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"最新のファイルを読み込みます: {latest_file}")

    # 📄 CSV ファイルを読み込む
    df = pd.read_csv(latest_file, parse_dates=['Date'])

    # 🧠 シミュレーション実行
    result = run_simulation_with_stats(df)
    print(result)

    # 💾 結果を CSV ファイルとして保存
    result.to_csv("result_stats.csv", index=False)
    print("シミュレーション結果を 'result_stats.csv' に出力しました。")