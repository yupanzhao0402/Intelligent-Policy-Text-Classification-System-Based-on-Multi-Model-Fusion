#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
# 设置中文字体和负号显示
plt.rcParams["font.family"] = "SimHei"  # 设置默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
import seaborn as sns
from transformers import BertTokenizer
import random
import os

# 导入自定义分割函数
try:
    from custom_split import stratified_train_val_test_split
    CUSTOM_SPLIT_AVAILABLE = True
except ImportError:
    CUSTOM_SPLIT_AVAILABLE = False
    print("未找到custom_split模块，将使用标准分割方法")

class PolicyDataset(Dataset):
    """政策文本数据集类"""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class DataProcessor:
    """数据处理类，用于处理政策文本数据"""
    def __init__(self, filepath, sample_size=1000, random_state=42):
        """
        初始化数据处理器
        
        参数:
        filepath (str): CSV文件路径
        sample_size (int): 采样大小，默认为None表示使用全部数据
        random_state (int): 随机种子
        """
        self.filepath = filepath
        self.sample_size = sample_size
        self.random_state = random_state
        self.label_encoder = None
        
        # 尝试加载tokenizer
        try:
            # 尝试从本地加载
            print("尝试从本地加载BERT模型...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
            print("成功从本地加载BERT模型")
        except Exception as e:
            print(f"无法从本地加载模型，错误: {e}")
            print("尝试从在线源下载模型...")
            # 尝试在线下载
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                print("成功从在线源下载BERT模型")
            except Exception as e2:
                print(f"在线下载失败，错误: {e2}")
                print("正在尝试使用备选方案...")
                
                # 备选方案：创建一个简单的中文分词器
                from transformers import BasicTokenizer
                self.tokenizer = BasicTokenizer(do_lower_case=True)
                print("已使用BasicTokenizer作为备用")
        
        # 加载数据
        self.data = self._load_data()
        
        # 编码标签
        self._encode_labels()
        
        # 划分数据集
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._split_data()
        
    def _load_data(self):
        """加载CSV数据"""
        # 读取CSV文件
        df = pd.read_csv(self.filepath)
        
        # 如果指定了样本大小，则随机抽样
        if self.sample_size is not None:
            if self.sample_size < len(df):
                df = df.sample(n=self.sample_size, random_state=self.random_state)
        
        return df
    
    def _encode_labels(self):
        """编码分类标签"""
        self.label_encoder = LabelEncoder()
        self.data['label_encoded'] = self.label_encoder.fit_transform(self.data['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
    def _split_data(self):
        """将数据集分为训练集、验证集和测试集"""
        if CUSTOM_SPLIT_AVAILABLE:
            # 使用自定义分割函数
            print("使用自定义分层抽样分割函数...")
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
                self.data['text'].values,
                self.data['label_encoded'].values,
                val_size=0.1,
                test_size=0.1,
                random_state=self.random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # 使用标准方法
            # 先划分训练集和临时测试集（80%训练，20%测试+验证）
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.data['text'].values, 
                self.data['label_encoded'].values,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.data['label_encoded'].values
            )
            
            # 检查临时测试集中每个类别的样本数
            unique_labels, counts = np.unique(y_temp, return_counts=True)
            min_samples = np.min(counts)
            
            # 再将临时测试集划分为验证集和测试集（验证和测试各占10%）
            if min_samples >= 4:  # 如果每个类别至少有4个样本，可以使用分层抽样
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=0.5,
                    random_state=self.random_state,
                    stratify=y_temp
                )
            else:  # 否则不使用分层抽样
                print(f"警告: 某些类别样本数量过少(最少{min_samples}个样本)，无法使用分层抽样进行第二次划分")
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=0.5,
                    random_state=self.random_state
                )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_dataloaders(self, batch_size=16):
        """创建DataLoader实例"""
        # 创建数据集实例
        train_dataset = PolicyDataset(self.X_train, self.y_train)
        val_dataset = PolicyDataset(self.X_val, self.y_val)
        test_dataset = PolicyDataset(self.X_test, self.y_test)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch):
        """整理批次数据"""
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # 使用BertTokenizer编码文本
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 获取输入IDs和注意力掩码
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # 将标签转换为张量
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def visualize_label_distribution(self):
        """可视化标签分布"""
        plt.figure(figsize=(12, 6))
        sns.countplot(y='label', data=self.data, order=self.data['label'].value_counts().index)
        plt.title('标签分布情况')
        plt.xlabel('数量')
        plt.ylabel('标签类别')
        plt.tight_layout()
        return plt
    
    def visualize_text_length_distribution(self):
        """可视化文本长度分布"""
        self.data['text_length'] = self.data['text'].apply(len)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['text_length'], bins=50)
        plt.title('文本长度分布')
        plt.xlabel('文本长度')
        plt.ylabel('频率')
        plt.axvline(x=128, color='r', linestyle='--', label='BERT最大长度(128)')
        plt.legend()
        plt.tight_layout()
        return plt
    
    def get_class_names(self):
        """获取类别名称"""
        return self.label_encoder.classes_
    
    def decode_label(self, label_id):
        """将标签ID解码为标签名称"""
        return self.label_encoder.inverse_transform([label_id])[0]

    def get_label_dict(self):
        """获取标签到ID的映射字典"""
        return {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
    
    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes
        
    def get_original_data(self):
        """获取原始数据和标签，用于特征提取"""
        return self.data['text'].values, self.data['label_encoded'].values


if __name__ == "__main__":
    # 测试代码
    processor = DataProcessor("sample_data.csv")
    
    # 打印数据集大小
    print(f"训练集大小: {len(processor.X_train)}")
    print(f"验证集大小: {len(processor.X_val)}")
    print(f"测试集大小: {len(processor.X_test)}")
    
    # 显示标签分布
    processor.visualize_label_distribution()
    plt.show()
    
    # 显示文本长度分布
    processor.visualize_text_length_distribution()
    plt.show()
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = processor.get_dataloaders()
    
    # 打印一个批次的数据
    batch = next(iter(train_loader))
    print("批次大小:", batch['input_ids'].shape)
    print("标签:", batch['labels']) 