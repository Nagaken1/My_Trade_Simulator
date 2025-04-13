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
    result['TotalProfit'] = result['Profit'].cumsum()

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

    result['TotalProfit'] = result['Profit'].cumsum()
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