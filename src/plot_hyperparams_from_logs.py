# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 优先 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题
sns.set(style="whitegrid")
# ========== 复用你的解析函数 ==========
def parse_args_table(lines, start_idx):
    args_dict = {}
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("===========================================") and i > start_idx:
            break
        if '|' in line and not line.startswith("Arguments"):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                if not key or not value:
                    i += 1
                    continue
                if key in ['T', 'K_eig', 'emb_size', 'epoch', 'batch_size']:
                    try:
                        args_dict[key] = int(value)
                    except ValueError:
                        args_dict[key] = value
                elif key in ['alpha_min', 'guidance_s', 'lr', 'l2', 'sigma_max']:
                    try:
                        args_dict[key] = float(value)
                    except ValueError:
                        args_dict[key] = value
                else:
                    args_dict[key] = value
        i += 1
    return args_dict

def extract_metrics_from_parentheses(line):
    match = re.search(r'\((.*)\)', line)
    if not match:
        return {}
    metrics_str = match.group(1)
    metrics = {}
    for item in metrics_str.split(','):
        if ':' in item:
            k, v = item.split(':', 1)
            try:
                metrics[k.strip()] = float(v.strip())
            except ValueError:
                continue
    return metrics

def parse_log_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            print(f"❌ 编码错误，跳过: {filepath}")
            return None

    args_dict = None
    dev_metrics = None
    test_metrics = None
    args_start_line = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if "Arguments" in stripped and "Values" in stripped:
            args_start_line = i + 2
        elif "Dev  After Training:" in stripped:
            dev_metrics = extract_metrics_from_parentheses(stripped)
        elif "Test After Training:" in stripped:
            test_metrics = extract_metrics_from_parentheses(stripped)

    if args_start_line != -1:
        args_dict = parse_args_table(lines, args_start_line)

    if args_dict is not None and dev_metrics is not None and test_metrics is not None:
        return {
            'file': str(filepath),
            'params': args_dict,
            'dev': dev_metrics,
            'test': test_metrics
        }
    return None

# ========== 绘图主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="从 ReChorus 日志直接绘图")
    parser.add_argument('--log_dir', type=str, default='../log/SDiff', help='日志目录路径')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    # 解析所有日志
    all_results = []
    for log_file in log_dir.rglob("*.txt"):
        result = parse_log_file(log_file)
        if result:
            all_results.append(result)

    if not all_results:
        print("❌ 未找到有效日志文件！")
        return

    print(f"✅ 成功加载 {len(all_results)} 个实验结果")

    # 提取绘图所需数据
    T_list = []
    HR5_test = []
    alpha_list = []
    sigma_list = []
    HR5_for_heatmap = []

    K_eig_list = []
    HR5_K = []

    for r in all_results:
        p = r['params']
        t = r['test']

        # 图1: T vs HR@5
        if 'T' in p and 'HR@5' in t:
            T_list.append(p['T'])
            HR5_test.append(t['HR@5'])

        # 图2: α_min vs σ_max
        if 'alpha_min' in p and 'sigma_max' in p and 'HR@5' in t:
            alpha_list.append(p['alpha_min'])
            sigma_list.append(p['sigma_max'])
            HR5_for_heatmap.append(t['HR@5'])

        # 图3: K_eig vs HR@5
        if 'K_eig' in p and 'HR@5' in t:
            K_eig_list.append(p['K_eig'])
            HR5_K.append(t['HR@5'])

    # ========== 开始绘图 ==========
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="whitegrid")

    # 图1: T vs Test HR@5
    if T_list:
        plt.figure(figsize=(8, 5))
        # 按 T 排序
        combined = sorted(zip(T_list, HR5_test))
        T_sorted, HR_sorted = zip(*combined)
        plt.plot(T_sorted, HR_sorted, marker='o', linewidth=2, markersize=8)
        plt.title('扩散步数 T 对 Test HR@5 的影响')
        plt.xlabel('T (Diffusion Steps)')
        plt.ylabel('Test HR@5')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('T_vs_HR@5.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 图2: α_min vs σ_max 热力图
    if alpha_list:
        # 构建热力图矩阵（用字典聚合）
        heatmap_data = defaultdict(lambda: defaultdict(list))
        for a, s, hr in zip(alpha_list, sigma_list, HR5_for_heatmap):
            # 四舍五入以合并相近值
            a_key = round(a, 4)
            s_key = round(s, 2)
            heatmap_data[a_key][s_key].append(hr)

        # 计算平均值
        rows = sorted(heatmap_data.keys(), reverse=True)  # 小 alpha 在上方
        cols = sorted(set(s_key for row in heatmap_data.values() for s_key in row.keys()))
        matrix = []
        for a in rows:
            row = []
            for s in cols:
                vals = heatmap_data[a][s]
                avg = sum(vals) / len(vals) if vals else float('nan')
                row.append(avg)
            matrix.append(row)

        plt.figure(figsize=(9, 6))
        ax = sns.heatmap(
            matrix,
            xticklabels=[f"{s:.2f}" for s in cols],
            yticklabels=[f"{a:.4f}" for a in rows],
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={'label': 'Test HR@5'}
        )
        plt.title('α_min 与 σ_max 对 Test HR@5 的联合影响')
        plt.xlabel('σ_max')
        plt.ylabel('α_min')
        plt.tight_layout()
        plt.savefig('alpha_sigma_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 图3: K_eig vs HR@5
    if K_eig_list:
        plt.figure(figsize=(8, 5))
        plt.scatter(K_eig_list, HR5_K, alpha=0.7, s=80)
        plt.title('谱维度 K_eig 对 Test HR@5 的影响')
        plt.xlabel('K_eig')
        plt.ylabel('Test HR@5')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('K_eig_vs_HR@5.png', dpi=300, bbox_inches='tight')
        plt.show()

    print("✅ 所有图表已保存到当前目录！")

if __name__ == '__main__':
    main()