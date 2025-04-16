import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import CheckButtons,RadioButtons
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf

def plot_candle_with_markers(df, title="ローソク足＋マーカー（Indexベース）"):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons, RadioButtons
    from mplfinance.original_flavor import candlestick_ohlc

    marker_color_map = {
        'Buy_New_OrderTime':    ('o', 'blue'),
        'Buy_New_ExecTime':     ('^', 'blue'),
        'Buy_Close_OrderTime':  ('s', 'blue'),
        'Buy_Close_ExecTime':   ('D', 'blue'),
        'Buy_Stop_OrderTime':   ('+', 'blue'),
        'Buy_Stop_ExecTime':    ('x', 'blue'),
        'Sell_New_OrderTime':   ('o', 'red'),
        'Sell_New_ExecTime':    ('^', 'red'),
        'Sell_Close_OrderTime': ('s', 'red'),
        'Sell_Close_ExecTime':  ('D', 'red'),
        'Sell_Stop_OrderTime':  ('+', 'red'),
        'Sell_Stop_ExecTime':   ('x', 'red')
    }

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Label'] = df['Date'].dt.strftime("%m-%d %H:%M")
    df.reset_index(drop=True, inplace=True)

    # OHLCデータ（X:整数Index）
    ohlc_data = [
        [i, row.Open, row.High, row.Low, row.Close]
        for i, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='g', colordown='r', alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Price")

    # X軸に文字列ラベルを付ける
    label_interval = max(1, len(df) // 10)
    ax.set_xticks(range(0, len(df), label_interval))
    ax.set_xticklabels(df['Label'][::label_interval], rotation=45, ha='right')

    # マーカー描画
    marker_lines_by_strategy = {}
    all_strategies = set()
    marker_types = set()
    matched_cols = {}

    for col in df.columns:
        for suffix in marker_color_map:
            if col.endswith(suffix):
                strategy = col[:-(len(suffix) + 1)]
                all_strategies.add(strategy)
                marker_types.add(suffix)
                matched_cols[(strategy, suffix)] = col

    all_strategies = sorted(all_strategies)
    marker_types = sorted(marker_types)

    for strategy in all_strategies:
        marker_lines_by_strategy[strategy] = {}
        for mtype in marker_types:
            full_col = matched_cols.get((strategy, mtype))
            if full_col and mtype in marker_color_map:
                marker, color = marker_color_map[mtype]
                mask = df[full_col].notna()

                if mask.sum() == 0:
                    continue

                xvals = df.index[mask]
                yvals = df['High'][mask] + 30

                line = ax.scatter(
                    xvals, yvals,
                    marker=marker, color=color,
                    s=80, label=full_col,
                    visible=False
                )
                marker_lines_by_strategy[strategy][mtype] = line

    # UI：戦略とマーカー切替
    selected_strategy = [all_strategies[0]]
    marker_flags = {m: True for m in marker_types}

    def update_visibility():
        for strategy, lines in marker_lines_by_strategy.items():
            for mtype, line in lines.items():
                visible = (strategy == selected_strategy[0]) and marker_flags[mtype]
                line.set_visible(visible)
        fig.canvas.draw_idle()

    # ラジオボタン：戦略選択
    rax = plt.axes([0.85, 0.6, 0.13, 0.2])
    radio = RadioButtons(rax, all_strategies)
    radio.on_clicked(lambda label: (selected_strategy.__setitem__(0, label), update_visibility()))

    # チェックボタン：マーカー種類切替
    cax = plt.axes([0.85, 0.3, 0.13, 0.2])
    check = CheckButtons(cax, marker_types, [True] * len(marker_types))
    check.on_clicked(lambda label: (marker_flags.__setitem__(label, not marker_flags[label]), update_visibility()))

    update_visibility()
    plt.tight_layout()
    plt.show()