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
import joblib
import os
from collections import Counter
import pandas as pd


class SelfAttention(nn.Module):
    """自注意力机制模块"""
    def __init__(self, input_dim, hidden_size, num_attention_heads, dropout_prob=0.1):
        """
        初始化自注意力模块
        
        参数:
        input_dim (int): 输入维度(模型数量*类别数)
        hidden_size (int): 隐藏层大小
        num_attention_heads (int): 注意力头数
        dropout_prob (float): Dropout比率
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"隐藏层大小 ({hidden_size}) 不是注意力头数 ({num_attention_heads}) 的整数倍"
            )
        
        self.input_dim = input_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # 查询、键、值的线性层
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Dropout和输出层
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def transpose_for_scores(self, x):
        """重塑张量以准备多头注意力计算"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, model_outputs):
        """
        前向传播
        
        参数:
        model_outputs (torch.Tensor): 模型输出 [batch_size, n_models, n_classes]
        
        返回:
        context_layer (torch.Tensor): 输出特征
        attention_probs (torch.Tensor): 注意力权重
        """
        batch_size, n_models, n_classes = model_outputs.size()
        
        # 将模型输出展平为 [batch_size, n_models, n_classes]
        flattened_outputs = model_outputs.reshape(batch_size, n_models, n_classes)
        
        # 应用输入投影，将展平后的输出映射到隐藏空间
        hidden_states = self.input_projection(flattened_outputs)
        
        # 计算查询、键、值
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 重塑张量以便进行多头注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # 应用softmax获取注意力权重
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 应用输出变换
        output = self.output(context_layer)
        
        return output, attention_probs


class AttentionVotingModel:
    """基于注意力机制的投票集成模型"""
    def __init__(self, models, hidden_size=None, num_attention_heads=8, 
                 dropout_prob=0.1, lr=0.001, weight_decay=1e-5, 
                 batch_size=64, epochs=30, device=None):
        """
        初始化注意力投票模型
        
        参数:
        models (list): 模型列表，每个元素为(名称, 模型)元组
        hidden_size (int): 注意力机制隐藏层大小
        num_attention_heads (int): 注意力头数量
        dropout_prob (float): Dropout概率
        lr (float): 学习率
        weight_decay (float): 权重衰减
        batch_size (int): 批次大小
        epochs (int): 训练轮数
        device (str): 设备，'cuda'或'cpu'
        """
        self.models = models
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 初始化模型组件
        self.attention = None
        self.voting_layer = None
        self.is_initialized = False
        self.history = None
        self.class_names = None
        
        # 从模型列表获取类别数
        try:
            # 尝试从第一个模型获取类别数
            _, first_model = self.models[0]
            sample_X = np.zeros((1, first_model.predict_proba(np.zeros((1, first_model.model.coef_.shape[1]))).shape[1]))
            self.num_classes = first_model.predict_proba(sample_X).shape[1]
        except (AttributeError, IndexError, ValueError) as e:
            # 如果无法从模型获取，设置一个默认值，在训练时会更新
            print(f"无法从模型获取类别数，将在训练时设置: {e}")
            self.num_classes = None
        
        # 初始化输入维度
        self.input_dim = None
        
        self.num_models = len(models)
        self.model_names = [name for name, _ in models]
    
    def _get_model_predictions(self, X):
        """
        获取各个模型的预测结果和概率
        
        参数:
        X (np.ndarray): 特征数据
        
        返回:
        predictions (list): 各个模型的预测结果
        probas (list): 各个模型的预测概率
        """
        predictions = []
        probas = []
        
        # 定义批处理大小
        batch_size = 512  # 较小的批处理大小，减少内存使用
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整
        
        for name, model in self.models:
            # 初始化批处理结果存储
            model_predictions = np.zeros(n_samples, dtype=np.int64)
            model_probas = np.zeros((n_samples, self.num_classes))
            
            # 批处理预测
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_X = X[start_idx:end_idx]
                
                # 获取当前批次的预测
                try:
                    batch_pred = model.predict(batch_X)
                    batch_proba = model.predict_proba(batch_X)
                    
                    # 存储当前批次结果
                    model_predictions[start_idx:end_idx] = batch_pred
                    model_probas[start_idx:end_idx] = batch_proba
                except Exception as e:
                    print(f"模型 {name} 预测失败: {e}，使用随机预测代替")
                    # 出错时使用随机预测
                    batch_pred = np.random.randint(0, self.num_classes, size=(end_idx-start_idx))
                    batch_proba = np.zeros((end_idx-start_idx, self.num_classes))
                    for j, cls in enumerate(batch_pred):
                        batch_proba[j, cls] = 1.0
                    
                    model_predictions[start_idx:end_idx] = batch_pred
                    model_probas[start_idx:end_idx] = batch_proba
            
            predictions.append(model_predictions)
            probas.append(model_probas)
        
        return predictions, probas
    
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping=False, early_stopping_params=None):
        """
        训练注意力投票模型
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        early_stopping (bool): 是否启用早停
        early_stopping_params (dict): 早停参数，包含 'patience' 和 'min_delta'
        
        返回:
        history (dict): 训练历史
        """
        # 设置早停默认参数
        if early_stopping and early_stopping_params is None:
            early_stopping_params = {'patience': 5, 'min_delta': 0.001}
        
        # 获取类别数
        if self.num_classes is None:
            self.num_classes = len(np.unique(y_train))
            print(f"设置模型类别数为: {self.num_classes}")
        
        # 获取各个模型的预测
        print("开始获取基础模型的预测结果，这可能需要一些时间...")
        _, train_probas = self._get_model_predictions(X_train)
        
        # 判断是否需要初始化注意力机制
        if self.attention is None or not self.is_initialized:
            # 从数据中获取实际维度
            n_models = len(train_probas)
            n_classes = train_probas[0].shape[1]
            # 输入维度是每个模型的类别数之和
            self.input_dim = n_classes
            
            # 如果hidden_size未指定，自动设置为合适的值
            if self.hidden_size is None:
                # 设置hidden_size为输入维度的倍数，确保能被注意力头数整除
                self.hidden_size = self.input_dim * 4
                # 调整hidden_size以能被注意力头数整除
                if self.hidden_size % self.num_attention_heads != 0:
                    self.hidden_size = ((self.hidden_size // self.num_attention_heads) + 1) * self.num_attention_heads
            
            print(f"初始化注意力机制，input_dim={self.input_dim}, hidden_size={self.hidden_size}，attention_heads={self.num_attention_heads}")
            
            # 创建注意力机制
            self.attention = SelfAttention(
                input_dim=self.input_dim,
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                dropout_prob=self.dropout_prob
            ).to(self.device)
            
            # 创建投票机制
            self.voting_layer = nn.Linear(self.hidden_size, 1).to(self.device)
            
            # 标记已初始化
            self.is_initialized = True
        
        # 将概率转换为张量
        train_proba_tensor = torch.FloatTensor(np.array(train_probas)).permute(1, 0, 2).to(self.device)  # [n_samples, n_models, n_classes]
        train_label_tensor = torch.LongTensor(y_train).to(self.device)
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(train_proba_tensor, train_label_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # 准备验证数据
        val_loader = None
        if X_val is not None and y_val is not None:
            _, val_probas = self._get_model_predictions(X_val)
            val_proba_tensor = torch.FloatTensor(np.array(val_probas)).permute(1, 0, 2).to(self.device)
            val_label_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(val_proba_tensor, val_label_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(self.attention.parameters()) + list(self.voting_layer.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 初始化历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 初始化早停相关变量
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        # 训练模型
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            # 训练模式
            self.attention.train()
            
            for batch_proba, batch_labels in train_loader:
                # 计算注意力权重
                attention_output, attention_weights = self.attention(batch_proba)
                
                # 聚合模型预测
                batch_size, num_models, num_classes = batch_proba.size()
                
                # 使用注意力输出加权模型预测
                weighted_proba = torch.zeros(batch_size, num_classes).to(self.device)
                
                # 对每个样本应用权重
                for i in range(batch_size):
                    # 提取当前样本的注意力输出
                    sample_attention = attention_output[i]  # [n_models, hidden_size]
                    
                    # 对每个模型计算权重
                    model_weights = torch.sigmoid(self.voting_layer(sample_attention)).squeeze(-1)  # [n_models]
                    model_weights = torch.softmax(model_weights, dim=0)  # 归一化
                    
                    # 加权平均模型预测
                    for j in range(num_models):
                        weighted_proba[i] += model_weights[j] * batch_proba[i, j]
                
                # 计算损失
                loss = criterion(weighted_proba, batch_labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            history['train_loss'].append(epoch_loss)
            
            # 在验证集上评估
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_on_validation(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # 早停检查
                if early_stopping and val_loader is not None:
                    if val_loss < best_val_loss - early_stopping_params.get('min_delta', 0.001):
                        best_val_loss = val_loss
                        early_stopping_counter = 0
                        # 保存当前最佳模型
                        best_model_state = {
                            'attention': self.attention.state_dict(),
                            'voting_layer': self.voting_layer.state_dict()
                        }
                        print(f"验证损失改善，保存当前模型状态")
                    else:
                        early_stopping_counter += 1
                        print(f"验证损失未改善，早停计数: {early_stopping_counter}/{early_stopping_params.get('patience', 5)}")
                        
                    if early_stopping_counter >= early_stopping_params.get('patience', 5):
                        print(f"早停激活! {early_stopping_params.get('patience', 5)} 轮验证损失没有提高")
                        # 恢复最佳模型
                        if best_model_state is not None:
                            self.attention.load_state_dict(best_model_state['attention'])
                            self.voting_layer.load_state_dict(best_model_state['voting_layer'])
                            print("加载训练过程中的最佳模型...")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")
        
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
        self.attention.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_proba, batch_labels in val_loader:
                # 计算注意力权重
                attention_output, attention_weights = self.attention(batch_proba)
                
                # 聚合模型预测
                batch_size, num_models, num_classes = batch_proba.size()
                
                # 使用注意力输出加权模型预测
                weighted_proba = torch.zeros(batch_size, num_classes).to(self.device)
                
                # 对每个样本应用权重
                for i in range(batch_size):
                    # 提取当前样本的注意力输出
                    sample_attention = attention_output[i]  # [n_models, hidden_size]
                    
                    # 对每个模型计算权重
                    model_weights = torch.sigmoid(self.voting_layer(sample_attention)).squeeze(-1)  # [n_models]
                    model_weights = torch.softmax(model_weights, dim=0)  # 归一化
                    
                    # 加权平均模型预测
                    for j in range(num_models):
                        weighted_proba[i] += model_weights[j] * batch_proba[i, j]
                
                # 计算损失
                loss = criterion(weighted_proba, batch_labels)
                running_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(weighted_proba.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        self.attention.train()
        return running_loss / len(val_loader), correct / total
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_pred (np.ndarray): 预测标签
        """
        # 获取概率
        probas = self.predict_proba(X)
        
        # 取最大概率的类别作为预测结果
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_proba (np.ndarray): 预测概率
        """
        # 确保模型已初始化
        if not self.is_initialized:
            raise RuntimeError("模型尚未初始化，请先训练模型")
            
        # 获取各个模型的预测概率
        _, model_probas = self._get_model_predictions(X)
        
        # 将概率转换为张量
        proba_tensor = torch.FloatTensor(np.array(model_probas)).permute(1, 0, 2).to(self.device)  # [n_samples, n_models, n_classes]
        
        # 设置为评估模式
        self.attention.eval()
        
        with torch.no_grad():
            # 计算注意力权重
            attention_output, attention_weights = self.attention(proba_tensor)
            
            # 聚合模型预测
            batch_size, num_models, num_classes = proba_tensor.size()
            
            # 使用注意力输出加权模型预测
            weighted_proba = torch.zeros(batch_size, num_classes).to(self.device)
            
            # 对每个样本应用权重
            for i in range(batch_size):
                # 提取当前样本的注意力输出
                sample_attention = attention_output[i]  # [n_models, hidden_size]
                
                # 对每个模型计算权重
                model_weights = torch.sigmoid(self.voting_layer(sample_attention)).squeeze(-1)  # [n_models]
                model_weights = torch.softmax(model_weights, dim=0)  # 归一化
                
                # 加权平均模型预测
                for j in range(num_models):
                    weighted_proba[i] += model_weights[j] * proba_tensor[i, j]
        
        # 恢复训练模式
        self.attention.train()
        
        return weighted_proba.cpu().numpy()
    
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
        
        print("注意力投票模型评估结果:")
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
        
        # 保存模型
        state = {
            'attention_state_dict': self.attention.state_dict(),
            'voting_layer_state_dict': self.voting_layer.state_dict(),
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'dropout_prob': self.dropout_prob,
            'model_names': self.model_names,
            'class_names': self.class_names,
            'history': self.history
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
        self.hidden_size = checkpoint['hidden_size']
        self.num_attention_heads = checkpoint['num_attention_heads']
        self.dropout_prob = checkpoint['dropout_prob']
        self.model_names = checkpoint.get('model_names', [name for name, _ in self.models])
        self.class_names = checkpoint.get('class_names', None)
        self.history = checkpoint.get('history', None)
        
        # 创建注意力机制
        self.attention = SelfAttention(
            input_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dropout_prob=self.dropout_prob
        ).to(self.device)
        
        # 创建投票机制
        self.voting_layer = nn.Linear(self.hidden_size, 1).to(self.device)
        
        # 加载参数
        self.attention.load_state_dict(checkpoint['attention_state_dict'])
        self.voting_layer.load_state_dict(checkpoint['voting_layer_state_dict'])
        
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
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names if self.class_names else range(cm.shape[1]), 
                   yticklabels=self.class_names if self.class_names else range(cm.shape[0]))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('注意力投票模型混淆矩阵')
        
        return plt
    
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
    
    def get_model_weights(self, X):
        """
        获取各个模型的权重
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        model_weights (np.ndarray): 模型权重
        """
        # 获取各个模型的预测概率
        _, model_probas = self._get_model_predictions(X)
        
        # 将概率转换为张量
        proba_tensor = torch.FloatTensor(np.array(model_probas)).permute(1, 0, 2).to(self.device)  # [n_samples, n_models, n_classes]
        
        # 设置为评估模式
        self.attention.eval()
        
        with torch.no_grad():
            # 计算注意力权重
            attention_output, attention_weights = self.attention(proba_tensor)
            
            # 获取每个模型的权重
            batch_size, num_models, _ = proba_tensor.size()
            model_weights = np.zeros((batch_size, num_models))
            
            for i in range(num_models):
                model_weight = self.voting_layer(attention_output[:, i, :]).squeeze(-1)
                model_weights[:, i] = torch.softmax(model_weight, dim=0).cpu().numpy()
        
        # 恢复训练模式
        self.attention.train()
        
        return model_weights
    
    def plot_model_weights(self, X):
        """
        可视化模型权重
        
        参数:
        X (np.ndarray): 特征
        """
        # 获取模型权重
        model_weights = self.get_model_weights(X)
        
        # 计算平均权重
        avg_weights = np.mean(model_weights, axis=0)
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.model_names)), avg_weights)
        plt.xlabel('模型')
        plt.ylabel('平均权重')
        plt.title('注意力投票模型的模型权重')
        plt.xticks(range(len(self.model_names)), self.model_names, rotation=45, ha='right')
        plt.tight_layout()
        
        return plt
    
    def compare_base_models(self, X_test, y_test):
        """
        比较各个基础模型的性能
        
        参数:
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试标签
        
        返回:
        comparison_df (pd.DataFrame): 比较结果
        """
        # 创建结果表格
        results = []
        
        # 评估各个基础模型
        for name, model in self.models:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                '模型': name,
                '准确率': accuracy,
                '精确率': precision,
                '召回率': recall,
                'F1分数': f1
            })
        
        # 评估投票模型
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            '模型': '注意力投票模型',
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1分数': f1
        })
        
        # 创建DataFrame并排序
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('准确率', ascending=False).reset_index(drop=True)
        
        return comparison_df
    
    def plot_training_history(self):
        """
        可视化训练历史
        
        返回:
        plt: matplotlib图表对象
        """
        if self.history is None:
            print("没有可用的训练历史。请先使用验证集训练模型。")
            return None
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('注意力投票模型训练损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.title('注意力投票模型验证准确率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        return plt
    
    def plot_model_weights_by_class(self, X_test, y_test):
        """
        可视化不同类别下各模型的权重
        
        参数:
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试标签
        """
        if self.class_names is None:
            print("请先设置类别名称。")
            return None
        
        # 获取模型权重
        model_weights = self.get_model_weights(X_test)
        y_pred = self.predict(X_test)
        
        # 计算每个类别下各模型的平均权重
        class_weights = {}
        for class_idx in range(len(self.class_names)):
            class_mask = (y_pred == class_idx)
            if np.sum(class_mask) > 0:  # 确保有该类别的样本
                class_weights[class_idx] = np.mean(model_weights[class_mask], axis=0)
        
        # 可视化每个类别下的模型权重
        plt.figure(figsize=(15, 10))
        bar_width = 0.8 / len(class_weights)
        
        for i, (class_idx, weights) in enumerate(class_weights.items()):
            plt.bar(
                np.arange(len(self.model_names)) + i * bar_width - 0.4 + bar_width/2, 
                weights, 
                width=bar_width, 
                label=f'类别: {self.class_names[class_idx]}'
            )
        
        plt.xlabel('模型')
        plt.ylabel('权重')
        plt.title('不同类别下各模型的平均权重')
        plt.xticks(np.arange(len(self.model_names)), self.model_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        return plt
    
    def plot_training_process(self, X_train, y_train, X_val, y_val):
        """
        训练模型并可视化训练过程和模型权重
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        """
        # 训练模型
        self.train(X_train, y_train, X_val, y_val)
        
        # 可视化训练历史
        self.plot_training_history()
        plt.show()
        
        # 可视化模型权重
        self.plot_model_weights(X_val)
        plt.show()
        
        # 如果已设置类别名称，可视化不同类别下的模型权重
        if self.class_names is not None:
            self.plot_model_weights_by_class(X_val, y_val)
            plt.show()


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    
    # 将父目录添加到路径中，以便导入其他模块
    sys.path.append(os.path.abspath('..'))
    
    from data_processor import DataProcessor
    from feature_extractor import FeatureExtractor
    from models.logistic_regression_model import LogisticRegressionModel
    from models.random_forest_model import RandomForestModel
    from models.svm_model import SVMModel
    
    # 加载数据
    processor = DataProcessor("../final_augmented_data20000样本0.3扰动.csv")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = processor.get_dataloaders()
    
    # 提取特征
    feature_extractor = FeatureExtractor()
    
    # 提取数据集特征
    X_train, y_train = feature_extractor.extract_features(train_loader)
    X_val, y_val = feature_extractor.extract_features(val_loader)
    X_test, y_test = feature_extractor.extract_features(test_loader)
    
    # 训练基础模型
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    
    svm_model = SVMModel()
    svm_model.train(X_train, y_train)
    
    # 创建模型列表
    base_models = [
        ('逻辑回归', lr_model),
        ('随机森林', rf_model),
        ('SVM', svm_model)
    ]
    
    # 创建注意力投票模型
    attention_voting = AttentionVotingModel(
        models=base_models,
        hidden_size=None,
        num_attention_heads=4,
        epochs=10
    )
    
    # 设置类别名称
    class_names = processor.get_class_names()
    attention_voting.set_class_names(class_names)
    
    # 训练模型并可视化
    attention_voting.plot_training_process(X_train, y_train, X_val, y_val) 