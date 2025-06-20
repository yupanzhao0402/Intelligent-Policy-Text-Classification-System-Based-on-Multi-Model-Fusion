#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split

def custom_stratified_split(X, y, test_size=0.2, random_state=None):
    """
    自定义分层抽样分割，可以处理样本少的类别
    
    参数:
    X: 特征数据
    y: 标签数据
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    X_train, X_test, y_train, y_test: 分割后的数据
    """
    # 获取唯一标签及其计数
    unique_labels, counts = np.unique(y, return_counts=True)
    
    # 初始化训练集和测试集索引
    train_indices = []
    test_indices = []
    
    # 对每个类别单独处理
    for label in unique_labels:
        # 获取当前类别的所有样本索引
        label_indices = np.where(y == label)[0]
        num_samples = len(label_indices)
        
        # 如果样本数大于等于4，使用分层抽样
        if num_samples >= 4:
            num_test = max(1, int(num_samples * test_size))
            # 随机选择测试样本
            if random_state is not None:
                np.random.seed(random_state + int(label))  # 为每个类别使用不同的随机种子
            
            # 随机打乱索引
            perm = np.random.permutation(num_samples)
            test_idx = label_indices[perm[:num_test]]
            train_idx = label_indices[perm[num_test:]]
        else:
            # 对于样本少的类别，确保至少一个样本在测试集中
            if random_state is not None:
                np.random.seed(random_state + int(label))
            
            # 随机决定分配到测试集的样本数
            if num_samples <= 1:
                # 如果只有1个样本，放入训练集
                train_idx = label_indices
                test_idx = []
            else:
                # 如果有2-3个样本，随机选择1个样本作为测试集
                perm = np.random.permutation(num_samples)
                test_idx = label_indices[perm[0:1]]
                train_idx = label_indices[perm[1:]]
        
        # 将当前类别的训练和测试索引添加到总列表中
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # 转换为数组
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # 返回分割后的数据
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def stratified_train_val_test_split(X, y, val_size=0.1, test_size=0.1, random_state=None):
    """
    将数据集分为训练集、验证集和测试集，使用分层抽样，并处理样本少的类别
    
    参数:
    X: 特征数据
    y: 标签数据
    val_size: 验证集比例
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    X_train, X_val, X_test, y_train, y_val, y_test: 分割后的数据
    """
    # 计算测试集和验证集的总比例
    temp_size = val_size + test_size
    
    # 首先将数据划分为训练集和临时集(包含验证集和测试集)
    try:
        # 尝试使用sklearn的train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # 如果失败，使用自定义分割方法
        print("使用自定义分层抽样方法进行第一次划分")
        X_train, X_temp, y_train, y_temp = custom_stratified_split(
            X, y, test_size=temp_size, random_state=random_state
        )
    
    # 然后将临时集划分为验证集和测试集
    try:
        # 计算验证集在临时集中的比例
        val_ratio = val_size / temp_size
        # 尝试使用sklearn的train_test_split
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), random_state=random_state, stratify=y_temp
        )
    except ValueError:
        # 如果失败，使用自定义分割方法
        print("使用自定义分层抽样方法进行第二次划分")
        val_ratio = val_size / temp_size
        X_val, X_test, y_val, y_test = custom_stratified_split(
            X_temp, y_temp, test_size=(1-val_ratio), random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # 测试代码
    # 创建一个测试数据集，包含一些样本数很少的类别
    X = np.random.rand(100, 5)  # 100个样本，5个特征
    y = np.zeros(100)
    y[0:50] = 0  # 50个类别0的样本
    y[50:80] = 1  # 30个类别1的样本 
    y[80:90] = 2  # 10个类别2的样本
    y[90:95] = 3  # 5个类别3的样本
    y[95:98] = 4  # 3个类别4的样本
    y[98:99] = 5  # 1个类别5的样本
    y[99:100] = 6  # 1个类别6的样本
    
    # 尝试分割
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X, y, val_size=0.1, test_size=0.1, random_state=42
    )
    
    # 打印结果
    print("训练集大小:", X_train.shape[0])
    print("验证集大小:", X_val.shape[0])
    print("测试集大小:", X_test.shape[0])
    
    # 打印每个集合中各类别的样本数
    for i, name in enumerate(["训练集", "验证集", "测试集"]):
        y_set = [y_train, y_val, y_test][i]
        unique_labels, counts = np.unique(y_set, return_counts=True)
        print(f"\n{name}中各类别的样本数:")
        for label, count in zip(unique_labels, counts):
            print(f"  类别 {label}: {count}个样本") 