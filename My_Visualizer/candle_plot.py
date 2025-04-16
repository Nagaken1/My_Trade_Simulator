import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import CheckButtons,RadioButtons
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
from collections import defaultdict

def plot_candle_with_markers(df, title="ローソク足＋マーカー（重なり段差調整）"):


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

    ohlc_data = [
        [i, row.Open, row.High, row.Low, row.Close]
        for i, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='g', colordown='r', alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Price")

    # X軸表示ラベル
    label_interval = max(1, len(df) // 10)
    ax.set_xticks(range(0, len(df), label_interval))
    ax.set_xticklabels(df['Label'][::label_interval], rotation=45, ha='right')

    # === Support / Resistance ライン描画 ===
    if "SupportLine" in df.columns:
        support_mask = df["SupportLine"].notna()
        ax.plot(df.index[support_mask], df["SupportLine"][support_mask],
                linestyle="--", color="cyan", label="Support")

    if "ResistanceLine" in df.columns:
        resistance_mask = df["ResistanceLine"].notna()
        ax.plot(df.index[resistance_mask], df["ResistanceLine"][resistance_mask],
                linestyle="--", color="magenta", label="Resistance")

    # マーカーの位置情報保持
    marker_lines_by_strategy = {}
    all_strategies = set()
    marker_types = set()
    matched_cols = {}

    # === ステップ1：マーカー位置集計（index単位）
    marker_index_map = defaultdict(list)  # {index: [ (strategy, mtype) ]}

    for col in df.columns:
        for suffix in marker_color_map:
            if col.endswith(suffix):
                strategy = col[:-(len(suffix) + 1)]
                all_strategies.add(strategy)
                marker_types.add(suffix)
                matched_cols[(strategy, suffix)] = col

                mask = df[col].notna()
                for idx in df[mask].index:
                    marker_index_map[idx].append((strategy, suffix))

    all_strategies = sorted(all_strategies)
    marker_types = sorted(marker_types)

    # === ステップ2：マーカー描画（段差オフセット付き）
    base_offset = 10
    offset_step = 10

    for strategy in all_strategies:
        marker_lines_by_strategy[strategy] = {}
        for mtype in marker_types:
            full_col = matched_cols.get((strategy, mtype))
            if full_col and mtype in marker_color_map:
                marker, color = marker_color_map[mtype]
                mask = df[full_col].notna()

                if mask.sum() == 0:
                    continue

                xvals = []
                yvals = []
                for idx in df[mask].index:
                    row_high = df.at[idx, 'High']
                    # そのインデックスにあるマーカーの並び順を取得
                    key_list = marker_index_map[idx]
                    order = key_list.index((strategy, mtype))  # このマーカーが何番目か
                    offset = base_offset + offset_step * order
                    xvals.append(idx)
                    yvals.append(row_high + offset)

                line = ax.scatter(
                    xvals, yvals,
                    marker=marker, color=color,
                    s=80, label=full_col,
                    visible=False
                )
                marker_lines_by_strategy[strategy][mtype] = line

    # === UIコントロール ===
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

    # チェックボタン：マーカー種類切り替え
    cax = plt.axes([0.85, 0.3, 0.13, 0.2])
    check = CheckButtons(cax, marker_types, [True] * len(marker_types))
    check.on_clicked(lambda label: (marker_flags.__setitem__(label, not marker_flags[label]), update_visibility()))

    update_visibility()
    plt.tight_layout()
    plt.show()