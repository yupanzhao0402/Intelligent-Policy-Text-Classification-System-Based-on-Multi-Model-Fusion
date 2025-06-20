#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预提取特征的Transformer模型：
- 使用预先提取的BERT特征训练自定义Transformer模型
- 提供与其他模型兼容的接口
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# 自定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 多头自注意力
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 自定义Transformer分类器
class PretrainedFeatureTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=768, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super(PretrainedFeatureTransformer, self).__init__()
        
        # 特征降维层（如果输入维度不是d_model）
        self.feature_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 投影到d_model维度
        x = self.feature_projection(x)
        
        # 添加位置信息（这里简单处理为批次维度为1）
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 取序列的平均值作为特征表示
        x = x.mean(dim=1)
        
        # 分类
        x = self.classifier(x)
        
        return x

# 自定义Transformer模型包装器，用于兼容sklearn接口
class TransformerModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.is_trained = True
        self.class_names = None
        self.feature_importances_ = None  # 添加空的特征重要性属性
        
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy()
        
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    def set_class_names(self, class_names):
        """设置类别名称"""
        self.class_names = class_names
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, path):
        """保存模型"""
        joblib.dump(self, path)
        print(f"模型已保存到 {path}")

# 预训练特征Transformer模型类
class PretrainedTransformerModel:
    def __init__(self, input_dim=768, num_classes=None, d_model=768, nhead=8, 
                 num_layers=4, dim_feedforward=2048, dropout=0.1, 
                 lr=1e-4, batch_size=64, epochs=50, device=None):
        """初始化预训练特征Transformer模型"""
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_trained = False
        self.class_names = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        # 确定类别数量
        if self.num_classes is None:
            self.num_classes = len(np.unique(y_train))
        
        # 创建模型
        model = PretrainedFeatureTransformer(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        model.to(self.device)
        
        # 转换数据为PyTorch张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 验证集
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 设置优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 训练循环
        best_val_f1 = 0.0
        best_model = None
        no_improve_epochs = 0
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # 训练模式
            model.train()
            total_loss = 0
            
            # 训练步骤
            for batch in tqdm(train_dataloader, desc="训练中"):
                # 获取数据
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # 清除之前的梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新参数
                optimizer.step()
                
                total_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = total_loss / len(train_dataloader)
            self.training_history['train_loss'].append(avg_train_loss)
            print(f"平均训练损失: {avg_train_loss:.4f}")
            
            # 在验证集上评估
            if val_dataloader:
                model.eval()
                val_preds = []
                val_true = []
                val_loss = 0
                
                for batch in tqdm(val_dataloader, desc="验证中"):
                    # 获取数据
                    features, labels = batch
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 不计算梯度
                    with torch.no_grad():
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predictions = torch.max(outputs, 1)
                    
                    # 添加到列表
                    val_preds.extend(predictions.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                
                # 计算验证损失
                avg_val_loss = val_loss / len(val_dataloader)
                self.training_history['val_loss'].append(avg_val_loss)
                
                # 计算指标
                val_f1 = f1_score(val_true, val_preds, average='weighted')
                self.training_history['val_f1'].append(val_f1)
                
                print(f"验证损失: {avg_val_loss:.4f}, 验证F1: {val_f1:.4f}")
                
                # 更新学习率
                scheduler.step(val_f1)
                
                # 保存最佳模型
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model = model.state_dict().copy()
                    print(f"发现更好的模型，F1分数: {best_val_f1:.4f}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f"没有改进，当前最佳F1: {best_val_f1:.4f}, 无改进轮数: {no_improve_epochs}")
                
                # 早停
                if no_improve_epochs >= 10:
                    print(f"连续10轮没有改进，提前停止训练")
                    break
        
        # 恢复最佳模型
        if best_model is not None:
            model.load_state_dict(best_model)
        
        # 创建模型包装器
        self.model = TransformerModelWrapper(model, self.device)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """预测类别"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        return self.model.predict_proba(X)
    
    def set_class_names(self, class_names):
        """设置类别名称"""
        self.class_names = class_names
        if self.model is not None:
            self.model.class_names = class_names
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        return self.model.evaluate(X_test, y_test)
    
    def save_model(self, path):
        """保存模型"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, path)
        print(f"模型已保存到 {path}") 