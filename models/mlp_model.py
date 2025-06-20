#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import os
import joblib
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class MLPClassifier(nn.Module):
    """多层感知机神经网络模型"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3, use_batch_norm=True):
        """
        初始化MLP模型
        
        参数:
        input_size (int): 输入特征维度
        hidden_sizes (list): 隐藏层维度列表
        output_size (int): 输出维度（类别数）
        dropout_rate (float): Dropout比率
        use_batch_norm (bool): 是否使用批归一化
        """
        super(MLPClassifier, self).__init__()
        
        # 构建多层感知机
        layers = []
        
        # 添加第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 添加中间隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 添加输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # 构建网络
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        x (torch.Tensor): 输入特征
        
        返回:
        output (torch.Tensor): 输出logits
        """
        return self.net(x)


class MLPModel:
    """MLP模型包装类"""
    def __init__(self, input_size=768, hidden_sizes=[512, 256, 128], output_size=None, 
                 dropout_rate=0.3, lr=0.001, weight_decay=1e-5, batch_size=64, 
                 epochs=100, device=None, use_batch_norm=True, early_stopping_patience=10,
                 scheduler_type='cosine', lr_patience=5, lr_factor=0.5, min_lr=1e-6, t_max=10):
        """
        初始化MLP模型
        
        参数:
        input_size (int): 输入特征维度
        hidden_sizes (list): 隐藏层维度列表
        output_size (int): 输出维度（类别数），如果为None，则在训练时根据标签确定
        dropout_rate (float): Dropout比率
        lr (float): 学习率
        weight_decay (float): 权重衰减系数
        batch_size (int): 批次大小
        epochs (int): 训练轮数
        device (str): 计算设备，可以是'cuda'或'cpu'
        use_batch_norm (bool): 是否使用批归一化
        early_stopping_patience (int): 早停的耐心值
        scheduler_type (str): 学习率调度器类型，'plateau'或'cosine'
        lr_patience (int): ReduceLROnPlateau的耐心值
        lr_factor (float): 学习率衰减因子
        min_lr (float): 最小学习率
        t_max (int): CosineAnnealingLR的周期
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_batch_norm = use_batch_norm
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_type = scheduler_type
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.t_max = t_max
        
        # 设置计算设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        if output_size is not None:
            self.model = MLPClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            ).to(self.device)
        
        self.class_names = None
        self.history = None
        self.best_model_state = None
        self.best_val_acc = 0.0
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练MLP模型
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        
        返回:
        history (dict): 训练历史
        """
        # 如果输出维度未指定，则根据标签推断
        if self.output_size is None:
            self.output_size = len(np.unique(y_train))
            self.model = MLPClassifier(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.output_size,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm
            ).to(self.device)
        
        # 准备数据
        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.LongTensor(y_train)
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # 准备验证数据
        val_loader = None
        if X_val is not None and y_val is not None:
            val_tensor_x = torch.FloatTensor(X_val)
            val_tensor_y = torch.LongTensor(y_val)
            val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        if self.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, 
                                          mode='max', 
                                          factor=self.lr_factor,
                                          patience=self.lr_patience,
                                          verbose=True,
                                          min_lr=self.min_lr)
        else:  # cosine
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=self.t_max,
                                          eta_min=self.min_lr)
        
        # 初始化历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 早停计数器和最佳模型
        early_stopping_counter = 0
        self.best_val_acc = 0.0
        
        # 开始训练
        print(f"开始训练MLP模型，最多{self.epochs}轮...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            history['train_loss'].append(epoch_loss)
            
            # 验证
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_on_validation(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # 更新学习率
                if self.scheduler_type == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
                
                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    early_stopping_counter = 0
                    print(f"新的最佳验证准确率: {val_acc:.4f}")
                else:
                    early_stopping_counter += 1
                    
                # 早停检查
                if early_stopping_counter >= self.early_stopping_patience:
                    print(f"早停激活! {self.early_stopping_patience} 轮验证准确率没有提高")
                    break
            else:
                # 无验证集时更新学习率
                if self.scheduler_type != 'plateau':
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            print("加载训练过程中的最佳模型...")
            self.model.load_state_dict(self.best_model_state)
        
        # 保存训练历史
        self.history = history
        return history
    
    def _evaluate_on_validation(self, val_loader, criterion):
        """
        在验证集上评估模型
        
        参数:
        val_loader (DataLoader): 验证数据加载器
        criterion: 损失函数
        
        返回:
        avg_loss (float): 平均损失
        accuracy (float): 准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return running_loss / len(val_loader), correct / total
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_pred (np.ndarray): 预测标签
        """
        self.model.eval()
        tensor_x = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_x)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_proba (np.ndarray): 预测概率
        """
        self.model.eval()
        tensor_x = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_x)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试标签
        
        返回:
        metrics (dict): 评估指标
        """
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 返回评估指标
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print("MLP模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        保存模型
        
        参数:
        filepath (str): 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型和配置
        state = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'class_names': self.class_names,
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(state, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        参数:
        filepath (str): 加载路径
        """
        # 加载模型
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复模型配置
        self.input_size = checkpoint['input_size']
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.output_size = checkpoint['output_size']
        self.dropout_rate = checkpoint['dropout_rate']
        self.use_batch_norm = checkpoint.get('use_batch_norm', True)
        self.class_names = checkpoint.get('class_names', None)
        self.history = checkpoint.get('history', None)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        # 创建模型
        self.model = MLPClassifier(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        # 加载参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"模型已从 {filepath} 加载")
    
    def set_class_names(self, class_names):
        """
        设置类别名称
        
        参数:
        class_names (list): 类别名称列表
        """
        self.class_names = class_names
    
    def plot_confusion_matrix(self, X_test, y_test):
        """
        绘制混淆矩阵
        
        参数:
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试标签
        """
        try:
            # 预测
            y_pred = self.predict(X_test)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            # 设置中文字体显示
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 
                                               'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 绘制混淆矩阵
            plt.figure(figsize=(12, 10))
            
            # 处理标签 - 解决标签可能是数组导致的模糊比较问题
            if self.class_names is not None:
                x_labels = self.class_names
                y_labels = self.class_names
            else:
                unique_labels = np.unique(y_test)
                x_labels = unique_labels
                y_labels = unique_labels
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=x_labels, 
                       yticklabels=y_labels)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('MLP模型混淆矩阵')
            plt.tight_layout()
            return plt.gcf()
        except Exception as e:
            print(f"绘制混淆矩阵失败: {e}")
            # 返回一个空白图形
            plt.figure()
            plt.text(0.5, 0.5, f"绘制混淆矩阵失败: {e}", 
                    horizontalalignment='center', verticalalignment='center')
            return plt.gcf()
    
    def print_classification_report(self, X_test, y_test):
        """
        打印分类报告
        
        参数:
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试标签
        """
        y_pred = self.predict(X_test)
        
        if self.class_names:
            report = classification_report(y_test, y_pred, target_names=self.class_names)
        else:
            report = classification_report(y_test, y_pred)
        
        print("分类报告:")
        print(report)
        
        return report
        
    def plot_training_history(self):
        """
        绘制训练历史
        """
        if self.history is None:
            print("没有训练历史记录")
            return None
        
        # 设置中文字体显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 
                                           'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('模型训练验证损失')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.axhline(y=self.best_val_acc, color='r', linestyle='--', label=f'最佳准确率: {self.best_val_acc:.4f}')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.title('模型验证准确率')
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
        
    def plot_training_process(self, X_train, y_train, X_val, y_val):
        """
        训练模型并可视化训练过程
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        """
        # 训练模型
        history = self.train(X_train, y_train, X_val, y_val)
        
        # 可视化训练历史
        self.plot_training_history()
        plt.show()


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    
    # 将父目录添加到路径中，以便导入其他模块
    sys.path.append(os.path.abspath('..'))
    
    from data_processor import DataProcessor
    from feature_extractor import FeatureExtractor
    
    # 加载数据
    processor = DataProcessor("../final_augmented_data20000样本0.3扰动.csv")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = processor.get_dataloaders()
    
    # 提取特征
    feature_extractor = FeatureExtractor()
    
    # 提取训练集和验证集特征
    X_train, y_train = feature_extractor.extract_features(train_loader)
    X_val, y_val = feature_extractor.extract_features(val_loader)
    
    # 创建MLP模型
    mlp_model = MLPModel(
        input_size=X_train.shape[1],
        hidden_sizes=[512, 256, 128],
        output_size=len(np.unique(y_train)),
        epochs=100,
        use_batch_norm=True,
        early_stopping_patience=10,
        scheduler_type='cosine'
    )
    
    # 训练模型并可视化
    mlp_model.plot_training_process(X_train, y_train, X_val, y_val) 