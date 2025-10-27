#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Okwinds DPO数据集处理工具

该模块用于处理Human-Like-DPO-Dataset数据集，包括数据加载、清洗和格式转换
为DPO训练所需的标准格式。
"""

import json
import os
from typing import Dict, List, Optional


def load_okwinds_dataset(data_path: str = "./data.json") -> List[Dict]:
    """
    加载Okwinds DPO数据集
    
    Args:
        data_path: 数据集文件路径，默认为当前目录下的data.json
        
    Returns:
        数据集列表，每个元素包含prompt、chosen和rejected字段
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        
        print(f"正在加载Okwinds DPO数据集...")
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"成功加载Okwinds数据集，共包含{len(dataset)}个样本")
        return dataset
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        raise
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        raise


def clean_and_validate_dataset(raw_dataset: List[Dict]) -> List[Dict]:
    """
    清洗并验证数据集
    
    Args:
        raw_dataset: 原始数据集列表
        
    Returns:
        清洗后的数据集列表
    """
    print("开始清洗和验证数据集...")
    
    cleaned_dataset = []
    valid_count = 0
    invalid_count = 0
    
    for idx, item in enumerate(raw_dataset):
        # 检查必需字段是否存在
        if not all(key in item for key in ['prompt', 'chosen', 'rejected']):
            invalid_count += 1
            print(f"样本 {idx} 缺少必需字段，跳过")
            continue
        
        # 检查字段是否为空
        if not item['prompt'].strip() or not item['chosen'].strip() or not item['rejected'].strip():
            invalid_count += 1
            print(f"样本 {idx} 包含空字段，跳过")
            continue
        
        # 移除多余的空白字符
        cleaned_item = {
            'prompt': item['prompt'].strip(),
            'chosen': item['chosen'].strip(),
            'rejected': item['rejected'].strip()
        }
        
        cleaned_dataset.append(cleaned_item)
        valid_count += 1
    
    print(f"数据集清洗完成: 有效样本 {valid_count}, 无效样本 {invalid_count}")
    return cleaned_dataset


def split_dataset(dataset: List[Dict], train_ratio: float = 0.9, 
                 random_seed: Optional[int] = 42) -> Dict[str, List[Dict]]:
    """
    将数据集分割为训练集和验证集
    
    Args:
        dataset: 清洗后的数据集
        train_ratio: 训练集比例
        random_seed: 随机种子，用于可重复分割
        
    Returns:
        包含'train'和'validation'键的字典
    """
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # 随机打乱数据集
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    
    # 分割数据集
    train_size = int(len(shuffled_dataset) * train_ratio)
    
    split_data = {
        'train': shuffled_dataset[:train_size],
        'validation': shuffled_dataset[train_size:]
    }
    
    print(f"数据集分割完成: 训练集 {len(split_data['train'])} 样本, "
          f"验证集 {len(split_data['validation'])} 样本")
    
    return split_data


def save_processed_dataset(dataset: Dict[str, List[Dict]], 
                          output_dir: str = ".\processed_data") -> None:
    """
    保存处理后的数据集
    
    Args:
        dataset: 处理后的数据集（包含train和validation）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in dataset.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存{split_name}集至 {output_path}")


def process_okwinds_dataset(data_path: str = "./data.json",
                           output_dir: str = "./processed_data",
                           split: bool = True,
                           train_ratio: float = 0.9) -> Dict[str, List[Dict]]:
    """
    完整的Okwinds数据集处理流程
    
    Args:
        data_path: 数据集文件路径，默认为当前目录下的data.json
        output_dir: 处理后数据保存目录，使用正斜杠以兼容不同操作系统
        split: 是否分割训练集和验证集
        train_ratio: 训练集比例
        
    Returns:
        处理后的数据集，包含'train'键，可选包含'validation'键
    """
    print("开始处理Okwinds DPO数据集...")
    
    # 1. 加载原始数据
    raw_data = load_okwinds_dataset(data_path)
    
    # 2. 清洗和验证数据
    cleaned_data = clean_and_validate_dataset(raw_data)
    
    # 3. 分割数据集（如果需要）
    if split:
        processed_data = split_dataset(cleaned_data, train_ratio)
    else:
        processed_data = {'train': cleaned_data}
    
    # 4. 保存处理后的数据
    save_processed_dataset(processed_data, output_dir)
    
    print("Okwinds DPO数据集处理完成！")
    return processed_data


def analyze_dataset(dataset: List[Dict]) -> Dict:
    """
    分析数据集统计信息
    
    Args:
        dataset: 数据集列表
        
    Returns:
        数据集统计信息
    """
    stats = {
        'total_samples': len(dataset),
        'prompt_lengths': [],
        'chosen_lengths': [],
        'rejected_lengths': []
    }
    
    for item in dataset:
        stats['prompt_lengths'].append(len(item['prompt']))
        stats['chosen_lengths'].append(len(item['chosen']))
        stats['rejected_lengths'].append(len(item['rejected']))
    
    # 计算基本统计
    import statistics
    
    stats_summary = {
        'total_samples': stats['total_samples'],
        'prompt_avg_length': statistics.mean(stats['prompt_lengths']),
        'prompt_max_length': max(stats['prompt_lengths']),
        'chosen_avg_length': statistics.mean(stats['chosen_lengths']),
        'chosen_max_length': max(stats['chosen_lengths']),
        'rejected_avg_length': statistics.mean(stats['rejected_lengths']),
        'rejected_max_length': max(stats['rejected_lengths'])
    }
    
    return stats_summary


if __name__ == "__main__":
    # 示例用法
    try:
        import argparse
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description="处理Okwinds DPO数据集")
        parser.add_argument("--data_path", type=str, default="./data.json", 
                          help="数据集文件路径")
        parser.add_argument("--output_dir", type=str, default="./processed_data", 
                          help="处理后数据保存目录")
        parser.add_argument("--no_split", action="store_true", 
                          help="不分割训练集和验证集")
        parser.add_argument("--train_ratio", type=float, default=0.9, 
                          help="训练集比例")
        
        args = parser.parse_args()
        
        # 处理数据集
        processed_data = process_okwinds_dataset(
            data_path=args.data_path,
            output_dir=args.output_dir,
            split=not args.no_split,
            train_ratio=args.train_ratio
        )
        
        # 分析训练集
        train_stats = analyze_dataset(processed_data['train'])
        print("\n数据集统计信息:")
        for key, value in train_stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()