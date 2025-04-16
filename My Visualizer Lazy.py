import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import CheckButtons, RadioButtons
from mplfinance.original_flavor import candlestick_ohlc
import os
from collections import defaultdict

# --- CSV 読み込み ---
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "result_stats.csv")
df = pd.read_csv(csv_path, parse_dates=['Date'])
df['Index'] = range(len(df))

# --- 描画パラメータ ---
window_size = 200
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

# --- GUI構築（先に root を作る必要あり） ---
root = tk.Tk()
root.title("ローソク足ビューア＋マーカー＋スクロール")

# --- 状態変数 ---
selected_strategy = [None]
marker_flags = {}
marker_lines_by_strategy = {}
strategy_var = tk.StringVar()
marker_vars = {}

fig, ax = plt.subplots(figsize=(12, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# --- 描画関数 ---
def plot_candles(start_index):
    end_index = start_index + window_size
    sub_df = df.iloc[start_index:end_index].copy()

    ohlc_data = [
        [i, row.Open, row.High, row.Low, row.Close]
        for i, row in sub_df.iterrows()
    ]

    ax.clear()
    candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='g', colordown='r', alpha=0.8)
    label_interval = max(1, len(sub_df) // 10)
    ax.set_xticks(range(0, len(sub_df), label_interval))
    ax.set_xticklabels(sub_df['Date'].dt.strftime("%m-%d %H:%M")[::label_interval], rotation=45, ha='right')
    ax.set_title(f"ローソク足: {start_index}〜{end_index}")
    ax.set_ylabel("Price")

    marker_lines_by_strategy.clear()
    marker_index_map = defaultdict(list)
    all_strategies = set()
    marker_types = set()
    matched_cols = {}

    for col in sub_df.columns:
        for suffix in marker_color_map:
            if col.endswith(suffix):
                strategy = col[:-(len(suffix) + 1)]
                all_strategies.add(strategy)
                marker_types.add(suffix)
                matched_cols[(strategy, suffix)] = col

                mask = sub_df[col].notna()
                for idx in sub_df[mask].index:
                    marker_index_map[idx].append((strategy, suffix))

    base_offset = 10
    offset_step = 10

    for strategy in sorted(all_strategies):
        marker_lines_by_strategy[strategy] = {}
        for mtype in sorted(marker_types):
            full_col = matched_cols.get((strategy, mtype))
            if full_col and mtype in marker_color_map:
                marker, color = marker_color_map[mtype]
                mask = sub_df[full_col].notna()
                if mask.sum() == 0:
                    continue
                xvals = []
                yvals = []
                for idx in sub_df[mask].index:
                    row_high = sub_df.at[idx, 'High']
                    key_list = marker_index_map[idx]
                    order = key_list.index((strategy, mtype))
                    offset = base_offset + offset_step * order
                    xvals.append(idx - start_index)
                    yvals.append(row_high + offset)
                line = ax.scatter(xvals, yvals, marker=marker, color=color, s=80, label=full_col, visible=False)
                marker_lines_by_strategy[strategy][mtype] = line

    update_visibility()
    canvas.draw()

# --- 表示切替 ---
def update_visibility():
    selected = strategy_var.get()
    for strategy, lines in marker_lines_by_strategy.items():
        for mtype, line in lines.items():
            visible = (strategy == selected) and marker_vars.get(mtype, tk.BooleanVar()).get()
            line.set_visible(visible)
    fig.canvas.draw_idle()

# --- スクロールバー ---
max_index = len(df) - window_size
scroll = tk.Scale(root, from_=0, to=max_index, orient=tk.HORIZONTAL, label="表示開始位置", length=800, command=lambda i: plot_candles(int(i)))
scroll.pack()

# --- ラジオボタン：戦略切替 ---
frame_strategy = tk.LabelFrame(root, text="戦略選択")
frame_strategy.pack(fill=tk.X, padx=10)

strategies = sorted({col.split('_')[0] for col in df.columns if any(sfx in col for sfx in marker_color_map)})
if strategies:
    strategy_var.set(strategies[0])
for s in strategies:
    rb = tk.Radiobutton(frame_strategy, text=s, value=s, variable=strategy_var, command=update_visibility)
    rb.pack(side=tk.LEFT)

# --- チェックボックス：マーカー種別切替 ---
frame_marker = tk.LabelFrame(root, text="マーカー表示")
frame_marker.pack(fill=tk.X, padx=10)

for mtype in sorted(set(marker_color_map.keys())):
    var = tk.BooleanVar(value=True)
    marker_vars[mtype] = var
    cb = tk.Checkbutton(frame_marker, text=mtype, variable=var, command=update_visibility)
    cb.pack(side=tk.LEFT)

# --- 初回描画 ---
plot_candles(0)

root.mainloop()