import pandas as pd
from pathlib import Path

# Test reading the CSV file
csv_path = Path('runs/train/vigilai_yolov11/results.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print('CSV found and loaded successfully!')
    print(f'Shape: {df.shape}')
    print('Columns:', df.columns.tolist())
    print('\nFinal epoch data:')
    final = df.iloc[-1]
    print(f'Epoch: {final["epoch"]}')
    print(f'mAP50: {final["metrics/mAP50(B)"]}')
    print(f'Box Loss: {final["train/box_loss"]}')
    print(f'Class Loss: {final["train/cls_loss"]}')
else:
    print('CSV file not found!')
