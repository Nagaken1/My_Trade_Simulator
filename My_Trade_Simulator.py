import os
import glob
import pandas as pd
from collections import deque

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
            'strategy_id','side', 'price', 'quantity', 'order_time', 'order_type',
            'trigger_price', 'triggered', 'status',
            'execution_price', 'execution_time', 'position_effect'
        ])

    def add_order(self, order):
        self.orders.loc[len(self.orders)] = {
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
                        (positions_df['side'] != order['side'])&
                        (positions_df['strategy_id'] == order['strategy_id'])
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
                exec_order.execution_price = self.orders.at[idx, 'execution_price']
                exec_order.execution_time = self.orders.at[idx, 'execution_time']
                executed.append(exec_order)

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
    """取引損益に基づく戦略指標を時系列で計算"""

    @staticmethod
    def total_profit(profit_list):
        """累計損益"""
        total = 0
        result = []
        for p in profit_list:
            total += p
            result.append(total)
        return result

    @staticmethod
    def winning_rate(profit_list):
        """勝率の推移（= 勝ち数 / 総取引数）"""
        wins = 0
        trades = 0
        rates = []

        for p in profit_list:
            if p > 0:
                wins += 1
            if p != 0:
                trades += 1

            if trades > 0:
                rates.append(round(wins / trades, 4))
            else:
                rates.append(0.0)

        return rates

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
        expected = []

        for win_rate, payoff in zip(winning_rate_list, payoff_ratio_list):
            ev = win_rate * payoff - (1 - win_rate)
            expected.append(round(ev, 4))

        return expected

    @staticmethod
    def drawdown(profit_list):
        dd = []
        max_profit = 0
        for p in profit_list:
            max_profit = max(max_profit, p)
            dd.append(round(max_profit - p, 4))
        return dd

    @staticmethod
    def max_drawdown(drawdown_list):
        max_dd = 0
        max_list = []
        for d in drawdown_list:
            max_dd = max(max_dd, d)
            max_list.append(max_dd)
        return max_list


# --- メイン処理 ---
def run_multi_strategy_simulation(df, strategies):
    """
    複数の戦略を時系列で横並びにシミュレーションし、統合結果をDataFrameで返す。

    Parameters:
        df: pandas.DataFrame
            元の1分足OHLCデータ（Date列がdatetime）
        strategies: list of tuples
            (strategy_func, strategy_id) のリスト

    Returns:
        df_result: pandas.DataFrame
            戦略別指標が横並びになった結果
    """
    # 共通のOHLC部分を保持
    df_result = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 各戦略を順番に実行
    for strategy_func, strategy_id in strategies:
        df_strategy = strategy_func(df.copy(), strategy_id=strategy_id)

        # 各列名に strategy_id を付加して区別（例: RuleA_Profit, RuleA_Signal）
        df_strategy = df_strategy.add_prefix(f"{strategy_id}_")

        # インデックスが Date なら join、そうでなければマージ
        if 'Date' in df_strategy.columns:
            df_result = df_result.merge(df_strategy, on='Date', how='left')
        else:
            df_result = df_result.merge(df_strategy, left_on='Date', right_index=True, how='left')

    return df_result


# --- 戦略---
def sample_strategy_rule_a(df, strategy_id='RuleA'):
    """
    シンプルな戦略A: 毎分 BUY → 2分後に成行SELLで確定。
    各分足ごとのSignal, Profit, TotalProfitを出力。
    """
    from collections import deque

    # 出力用DataFrame
    result = pd.DataFrame(index=df['Date'])
    result['Signal'] = 0
    result['Profit'] = 0.0

    # 建玉管理（FIFOキュー）
    positions = deque()

    for i in range(len(df)):
        now = df.iloc[i]
        now_time = now['Date']
        price = now['Close']

        # --- ① 新規買いエントリー（毎分）
        order = Order(
            side='BUY',
            price=price,
            quantity=1,
            order_time=now_time,
            order_type='market',
            position_effect='open',
            strategy_id=strategy_id
        )
        # 新規建玉を保持
        positions.append({
            'entry_time': now_time,
            'entry_price': price,
            'quantity': 1
        })
        result.at[now_time, 'Signal'] = 1  # 1 = エントリー

        # --- ② 2分後に成行で決済
        if len(positions) > 0 and i >= 2:
            pos = positions.popleft()
            exit_time = now_time
            exit_price = price
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
            result.at[exit_time, 'Profit'] = pnl

    # --- 累積損益
    calc = TradeStatisticsCalculator()
    profits = result['Profit'].tolist()
    result['TotalProfit'] = calc.total_profit(profits)
    result['WinningRate'] = calc.winning_rate(profits)
    result['PayoffRatio'] = calc.payoff_ratio(profits)
    result['ExpectedValue'] = calc.expected_value(result['WinningRate'], result['PayoffRatio'])
    result['DrawDown'] = calc.drawdown(profits)
    result['MaxDrawDown'] = calc.max_drawdown(result['DrawDown'])
    return result


# --- Sample Strategy Rule B ---
def sample_strategy_rule_b(df, strategy_id='RuleB'):
    result = pd.DataFrame(index=df['Date'])
    result['Signal'] = 0
    result['Profit'] = 0.0

    positions = deque()

    for i in range(len(df)):
        now = df.iloc[i]
        now_time = now['Date']
        price = now['Close']

        # SELL エントリー
        order = Order(
            side='SELL',
            price=price,
            quantity=1,
            order_time=now_time,
            order_type='market',
            position_effect='open',
            strategy_id=strategy_id
        )
        positions.append({
            'entry_time': now_time,
            'entry_price': price,
            'quantity': 1
        })
        result.at[now_time, 'Signal'] = -1

        # 3分後にBUYで決済
        if len(positions) > 0 and i >= 3:
            pos = positions.popleft()
            exit_time = now_time
            exit_price = price
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']  # 売りから入って利益
            result.at[exit_time, 'Profit'] = pnl

    calc = TradeStatisticsCalculator()
    profits = result['Profit'].tolist()
    result['TotalProfit'] = calc.total_profit(profits)
    result['WinningRate'] = calc.winning_rate(profits)
    result['PayoffRatio'] = calc.payoff_ratio(profits)
    result['ExpectedValue'] = calc.expected_value(result['WinningRate'], result['PayoffRatio'])
    result['DrawDown'] = calc.drawdown(profits)
    result['MaxDrawDown'] = calc.max_drawdown(result['DrawDown'])
    return result



# --- 実行ブロック ---
def run_multi_strategy_simulation(df, strategies):
    df_result = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    for strategy_func, strategy_id in strategies:
        df_strategy = strategy_func(df.copy(), strategy_id=strategy_id)
        df_strategy = df_strategy.add_prefix(f"{strategy_id}_")

        if 'Date' in df_strategy.columns:
            df_result = df_result.merge(df_strategy, on='Date', how='left')
        else:
            df_result = df_result.merge(df_strategy, left_on='Date', right_index=True, how='left')

    return df_result


if __name__ == '__main__':
    import os
    import glob

    csv_files = glob.glob(os.path.join("Input_csv", "*.csv"))
    if not csv_files:
        print("Input_csv フォルダに CSV ファイルが見つかりません。")
        exit()

    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"最新のファイルを読み込みます: {latest_file}")

    df = pd.read_csv(latest_file, parse_dates=['Date'])

    strategies = [
        (sample_strategy_rule_a, 'RuleA'),
        (sample_strategy_rule_b, 'RuleB')
    ]

    result = run_multi_strategy_simulation(df, strategies)
    print(result.head())

    result.to_csv("result_multi_strategy.csv", index=False)
    print("シミュレーション結果を 'result_multi_strategy.csv' に出力しました。")