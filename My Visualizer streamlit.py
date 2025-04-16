import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# データ読み込み
df = pd.read_csv("result_stats.csv", parse_dates=["Date"])
st.title("\U0001F4C8 ローソク足＋マーカー ビューア（Streamlit）")

# 表示範囲を選ぶスライダー
window_size = st.slider("表示足本数", min_value=100, max_value=1000, value=300, step=50)
start_idx = st.slider("開始位置", min_value=0, max_value=len(df)-window_size, value=0, step=50)
df_window = df.iloc[start_idx:start_idx + window_size].copy()

# チャート描画
fig = go.Figure(data=[
    go.Candlestick(
        x=df_window['Date'],
        open=df_window['Open'],
        high=df_window['High'],
        low=df_window['Low'],
        close=df_window['Close'],
        name='ローソク足'
    )
])

# マーカーを自動検出して追加
marker_color_map = {
    'Buy_New_ExecTime':     ('triangle-up', 'blue'),
    'Buy_Close_ExecTime':   ('diamond', 'blue'),
    'Buy_Stop_ExecTime':    ('x', 'blue'),
    'Sell_New_ExecTime':    ('triangle-up', 'red'),
    'Sell_Close_ExecTime':  ('diamond', 'red'),
    'Sell_Stop_ExecTime':   ('x', 'red')
}

for col, (symbol, color) in marker_color_map.items():
    for strategy in sorted({c.split('_')[0] for c in df.columns if c.endswith(col)}):
        full_col = f"{strategy}_{col}"
        if full_col in df_window.columns:
            mask = df_window[full_col].notna()
            fig.add_trace(go.Scatter(
                x=df_window.loc[mask, 'Date'],
                y=df_window.loc[mask, 'High'] + 30,
                mode='markers',
                marker=dict(symbol=symbol, size=10, color=color),
                name=full_col
            ))

fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)