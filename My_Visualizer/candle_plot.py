import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import CheckButtons,RadioButtons
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
from collections import defaultdict
import mplcursors

def plot_candle_with_markers(df, title="ローソク足＋マーカー＋支持線/抵抗線"):

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

    label_interval = max(1, len(df) // 10)
    ax.set_xticks(range(0, len(df), label_interval))
    ax.set_xticklabels(df['Label'][::label_interval], rotation=45, ha='right')

    # === 支持線・抵抗線（複数）をリストから描画 ===
    support_yvals = set()
    resistance_yvals = set()

    for col in df.columns:
        if col.endswith("SupportLines"):
            for raw in df[col].dropna():
                try:
                    values = eval(raw) if isinstance(raw, str) else raw
                    support_yvals.update(values)
                except:
                    continue
        elif col.endswith("ResistanceLines"):
            for raw in df[col].dropna():
                try:
                    values = eval(raw) if isinstance(raw, str) else raw
                    resistance_yvals.update(values)
                except:
                    continue

    # 描画：ラインをローソク足全体の期間に渡して引く
    x_start = 0
    x_end = len(df) - 1

    for y in support_yvals:
        ax.hlines(y, x_start, x_end, linestyles="--", colors="cyan", label="Support")

    for y in resistance_yvals:
        ax.hlines(y, x_start, x_end, linestyles="--", colors="magenta", label="Resistance")

    # --- 単一値型の SupportLine 描画（値が連続する区間ごとに線を引く） ---
    def draw_flat_lines(series: pd.Series, color: str, label: str):
        last_val = None
        start_idx = None

        for i, val in enumerate(series):
            if pd.isna(val):
                if last_val is not None and start_idx is not None:
                    ax.hlines(last_val, start_idx, i - 1, linestyles="--", colors=color, label=label)
                    last_val = None
                    start_idx = None
                continue

            if val != last_val:
                if last_val is not None and start_idx is not None:
                    ax.hlines(last_val, start_idx, i - 1, linestyles="--", colors=color, label=label)
                last_val = val
                start_idx = i

        if last_val is not None and start_idx is not None:
            ax.hlines(last_val, start_idx, len(series) - 1, linestyles="--", colors=color, label=label)

    # ✅ カラム名の後方一致で柔軟に対応
    for col in df.columns:
        if col.endswith("SupportLine"):
            draw_flat_lines(df[col], color="cyan", label=col)
        elif col.endswith("ResistanceLine"):
            draw_flat_lines(df[col], color="magenta", label=col)

    # === マーカー処理 ===
    marker_lines_by_strategy = {}
    all_strategies = set()
    marker_types = set()
    matched_cols = {}

    marker_index_map = defaultdict(list)

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
                    key_list = marker_index_map[idx]
                    order = key_list.index((strategy, mtype))
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

    rax = plt.axes([0.85, 0.6, 0.13, 0.2])
    radio = RadioButtons(rax, all_strategies)
    radio.on_clicked(lambda label: (selected_strategy.__setitem__(0, label), update_visibility()))

    cax = plt.axes([0.85, 0.3, 0.13, 0.2])
    check = CheckButtons(cax, marker_types, [True] * len(marker_types))
    check.on_clicked(lambda label: (marker_flags.__setitem__(label, not marker_flags[label]), update_visibility()))


    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = int(sel.target[0])
        if 0 <= index < len(df):
            row = df.iloc[index]
            label = (
                f"{row['Date'].strftime('%Y/%m/%d %H:%M')}\n"
                f"O: {row['Open']:.1f}  H: {row['High']:.1f}  "
                f"L: {row['Low']:.1f}  C: {row['Close']:.1f}"
            )
            sel.annotation.set(text=label)


    update_visibility()
    plt.tight_layout()
    plt.show()