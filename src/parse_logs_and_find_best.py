# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from pathlib import Path

def parse_args_table(lines, start_idx):
    """从 Arguments 表格中提取超参数字典"""
    args_dict = {}
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("===========================================") and i > start_idx:
            break  # 表格结束
        if '|' in line and not line.startswith("Arguments"):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                if not key or not value:
                    i += 1
                    continue
                # 转换类型
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
    """从 (HR@5:0.3430,NDCG@5:0.2330,...) 提取指标字典"""
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
    """解析单个日志文件，返回 dict 或 None（若无效）"""
    try:
        # 尝试多种编码，优先 utf-8-sig（兼容 Windows BOM）
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

    in_args_table = False
    args_start_line = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 检测 Arguments 表格开始
        if "Arguments" in stripped and "Values" in stripped:
            in_args_table = True
            args_start_line = i + 2  # 跳过分隔线
            continue
        
        # 解析指标
        if "Dev  After Training:" in stripped:
            dev_metrics = extract_metrics_from_parentheses(stripped)
        elif "Test After Training:" in stripped:
            test_metrics = extract_metrics_from_parentheses(stripped)

    # 如果检测到表格，解析它
    if args_start_line != -1:
        args_dict = parse_args_table(lines, args_start_line)

    if args_dict is not None and dev_metrics is not None and test_metrics is not None:
        return {
            'file': str(filepath),
            'params': args_dict,
            'dev': dev_metrics,
            'test': test_metrics
        }
    else:
        missing = []
        if args_dict is None: missing.append("args")
        if dev_metrics is None: missing.append("dev")
        if test_metrics is None: missing.append("test")
        print(f"❌ 跳过不完整日志 ({', '.join(missing)}): {filepath}")
        return None

def main():
    parser = argparse.ArgumentParser(description="从 ReChorus 日志中找出最佳超参数组合")
    parser.add_argument('--log_dir', type=str, default='log/SDiff', help='日志目录路径')
    parser.add_argument('--metric', type=str, default='HR@5', help='用于选择的主指标（如 HR@5, NDCG@10）')
    parser.add_argument('--phase', type=str, default='dev', choices=['dev', 'test'], help='在哪个阶段选优（通常用 dev）')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    all_results = []
    for log_file in log_dir.rglob("*.txt"):
        result = parse_log_file(log_file)
        if result:
            all_results.append(result)

    if not all_results:
        print("未找到有效日志文件！")
        return

    # 过滤出包含目标指标的结果
    valid_results = [
        r for r in all_results
        if args.metric in r[args.phase]
    ]

    if not valid_results:
        print(f"没有日志包含指标 {args.phase}.{args.metric}")
        return

    # 找出最佳（指标值最大）
    best = max(valid_results, key=lambda x: x[args.phase][args.metric])

    print("\n" + "="*60)
    print(f"🏆 最佳超参数组合（基于 {args.phase}.{args.metric}）")
    print("="*60)
    print(f"日志文件: {best['file']}")
    print(f"{args.phase}.{args.metric} = {best[args.phase][args.metric]:.4f}")
    print("\n🔍 超参数:")
    param_keys = ['T', 'K_eig', 'emb_size', 'alpha_min', 'guidance_s', 'lr', 'l2', 'sigma_max']
    for k in param_keys:
        if k in best['params']:
            print(f"  {k}: {best['params'][k]}")

    print("\n🧪 测试集性能:")
    for metric in sorted(best['test'].keys()):
        print(f"  {metric}: {best['test'][metric]:.4f}")

    # 保存最佳结果到 JSON
    output_path = log_dir / "best_config.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'file': best['file'],
            'params': best['params'],
            'dev': best['dev'],
            'test': best['test'],
            'selection_metric': f"{args.phase}.{args.metric}",
            'value': best[args.phase][args.metric]
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 最佳配置已保存至: {output_path}")

if __name__ == '__main__':
    main()