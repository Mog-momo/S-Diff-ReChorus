# -*- coding: UTF-8 -*-
import pandas as pd
from ast import literal_eval
import os

def safe_literal_eval(x):
    if isinstance(x, str) and ('[' in x or ',' in x):
        try:
            return literal_eval(x)
        except Exception as e:
            print(f"⚠️ Warning: Failed to parse as list: {str(x)[:50]}... Error: {e}")
            return x
    return x

def inspect_dataset(file_path, name):
    if not os.path.exists(file_path):
        print(f"❌ {name}: File not found - {file_path}")
        return

    try:
        # 尝试读取，自动解析 list 字段
        df = pd.read_csv(file_path, sep='\t', nrows=3)
        # 手动解析可能未被正确加载的 list 列
        for col in ['item_id', 'neg_items']:
            if col in df.columns:
                df[col] = df[col].apply(safe_literal_eval)
    except Exception as e:
        print(f"💥 Error loading {file_path}: {e}")
        return

    print(f"\n🔍 Inspecting {name} ({file_path})")
    print("-" * 60)
    print("First row sample:")
    for col in df.columns:
        val = df[col].iloc[0]
        print(f"  {col:<12}: {type(val).__name__} → {str(val)[:70]}{'...' if len(str(val)) > 70 else ''}")

    # 检查关键字段
    if 'item_id' in df.columns:
        item_val = df['item_id'].iloc[0]
        if isinstance(item_val, list):
            print(f"\n✅ 'item_id' is a LIST (length={len(item_val)}) — GOOD for evaluation!")
        else:
            print(f"\n❌ 'item_id' is NOT a list (type={type(item_val)}) — May cause issues in SDiff/BPR!")

    print("=" * 60)

if __name__ == "__main__":
    # ✅ 关键修改：直接在当前目录找文件！
    train_file = "train.csv"
    dev_file   = "dev.csv"
    test_file  = "test.csv"

    inspect_dataset(train_file, "Training Set")
    inspect_dataset(dev_file,   "Validation Set")
    inspect_dataset(test_file,  "Test Set")