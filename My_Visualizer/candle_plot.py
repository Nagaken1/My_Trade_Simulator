import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import CheckButtons,RadioButtons

def plot_candle_with_markers(df, title="Candle Chart with Strategy and Marker Toggle"):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ローソク足描画
    ohlc = df[['Open', 'High', 'Low', 'Close']].copy()
    ohlc['Date'] = mdates.date2num(ohlc.index)
    ohlc_values = ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values

    from mplfinance.original_flavor import candlestick_ohlc
    candlestick_ohlc(ax, ohlc_values, width=0.0008, colorup='g', colordown='r', alpha=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()

    # ==== マーカー描画 ====
    strategies = sorted(set('_'.join(col.split('_')[:2]) for col in df.columns if col.endswith('OrderPlaced')))
    marker_types = ['OrderPlaced', 'Executed', 'LossCutTriggered']
    colors = {'OrderPlaced': 'gray', 'Executed': 'green', 'LossCutTriggered': 'red'}
    markers = {'OrderPlaced': 'o', 'Executed': '*', 'LossCutTriggered': 'x'}

    marker_lines = {s: {} for s in strategies}

    for strategy in strategies:
        for mtype in marker_types:
            col_name = f"{strategy}_{mtype}"
            if col_name in df.columns:
                yvals = df['High'].where(df[col_name] == 1) + 20
                line = ax.scatter(
                    mdates.date2num(df.index), yvals,
                    marker=markers.get(mtype, 'o'),
                    color=colors.get(mtype, 'blue'),
                    label=f"{strategy}_{mtype}",
                    visible=False
                )
                marker_lines[strategy][mtype] = line

    # === 状態管理 ===
    selected_strategy = [strategies[0]]
    marker_flags = {m: True for m in marker_types}

    def update_visibility():
        for s in strategies:
            for m in marker_types:
                line = marker_lines[s].get(m)
                if line:
                    line.set_visible(s == selected_strategy[0] and marker_flags[m])
        fig.canvas.draw_idle()

    # === ラジオボタン：戦略選択 ===
    rax = plt.axes([0.85, 0.6, 0.13, 0.2])
    radio = RadioButtons(rax, strategies)

    def on_strategy_select(label):
        selected_strategy[0] = label
        update_visibility()

    radio.on_clicked(on_strategy_select)

    # === チェックボタン：マーカー種別切り替え ===
    cax = plt.axes([0.85, 0.3, 0.13, 0.2])
    check = CheckButtons(cax, marker_types, [True] * len(marker_types))

    def on_marker_toggle(label):
        marker_flags[label] = not marker_flags[label]
        update_visibility()

    check.on_clicked(on_marker_toggle)

    # 初回表示
    update_visibility()

    plt.tight_layout()
    plt.show()
