import os
import pandas as pd
from My_Visualizer.candle_plot import plot_candle_with_markers

base_dir = os.path.dirname(__file__)  # このスクリプトがあるフォルダ
csv_path = os.path.join(base_dir, "result_stats.csv")  # 一つ上の階層にある想定

df = pd.read_csv(csv_path, parse_dates=['Date'])

# ✅ 必要なカラムが存在するかチェック
required_cols = ['Open', 'High', 'Low', 'Close']
if all(col in df.columns for col in required_cols):
    plot_candle_with_markers(
        df,
        title="Rule_StopLoss マーカー付きローソク足"
    )
else:
    print("❗ OHLC 列が不足しています")