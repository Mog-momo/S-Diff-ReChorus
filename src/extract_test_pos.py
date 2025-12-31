# -*- coding: UTF-8 -*-
"""
从带负样本的测试集中提取正样本，生成标准 test.csv
输入: 原始测试集（含 user_id, item_id（正样本）, neg_xxx（负样本））
输出: test_only_pos.csv (仅 user_id, item_id)
"""

import pandas as pd

# 🔧 配置路径
RAW_TEST_FILE = "data/Grocery_and_Gourmet_Food/test.csv"
OUTPUT_TEST_FILE = "data/Grocery_and_Gourmet_Food/test_only_pos.csv"
SEP = "\t"  # 根据实际分隔符调整（可能是 ','）

# 📥 加载数据
df = pd.read_csv(RAW_TEST_FILE, sep=SEP)

# ✅ 直接提取正样本对（假设 item_id 就是正样本）
pos_df = df[['user_id', 'item_id']]

# 💾 保存
pos_df.to_csv(OUTPUT_TEST_FILE, sep=SEP, index=False)
print(f"✅ 正样本已提取并保存至: {OUTPUT_TEST_FILE}")
print("前5行示例:")
print(pos_df.head())