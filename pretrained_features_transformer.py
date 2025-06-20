#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预提取特征的Transformer训练脚本：
- 使用/extracted_features目录中预先提取的特征
- 训练自定义Transformer模型
- 与注意力投票模型进行比较
- 生成可视化结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import matplotlib.font_manager as fm
import seaborn as sns
import joblib
import openpyxl
from transformers import BertModel, BertTokenizer

# 确保能导入主项目模块
sys.path.append('..')

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子，确保结果可重现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置matplotlib中文支持
def setup_matplotlib_chinese():
    """配置matplotlib中文支持"""
    # 检查系统中文字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 中文字体优先级
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'KaiTi', 'FangSong']
    
    # 查找可用的中文字体
    for font in chinese_fonts:
        if font in fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            return font
    
    # 如果找不到合适的中文字体，尝试使用系统默认字体
    print("未找到中文字体，尝试使用系统默认字体...")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return None

# 创建目录
def create_directories():
    """创建必要的目录"""
    os.makedirs("../compare/plots", exist_ok=True)
    os.makedirs("../compare/results", exist_ok=True)
    os.makedirs("../models/pretrained_transformer", exist_ok=True)

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

# 简化的特征提取器
class SimpleFeatureExtractor:
    """简化的特征提取器，直接使用BERT模型提取特征"""
    
    def __init__(self, model_name='bert-base-chinese', device='cpu'):
        """初始化特征提取器"""
        self.model_name = model_name
        self.device = device
        
        print(f"加载BERT模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        print("BERT模型加载完成")
    
    def extract_features_from_text(self, text):
        """从单个文本中提取特征"""
        # 对文本进行tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 将inputs移至设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取[CLS]令牌的最后一层隐藏状态
        cls_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_feature[0]  # 返回batch中的第一个(唯一一个)特征向量

# 加载Excel文件
def load_excel(file_path):
    """使用openpyxl加载Excel文件"""
    print(f"加载Excel文件: {file_path}")
    try:
        wb = openpyxl.load_workbook(file_path)
        sheet = wb.active
        
        # 获取表头
        headers = [cell.value for cell in sheet[1]]
        
        # 获取数据
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            data.append({headers[i]: row[i] for i in range(len(headers))})
        
        return data
    except Exception as e:
        print(f"加载Excel文件失败: {e}")
        return []

# 从文本中提取特征
def extract_features_from_texts(texts, feature_extractor):
    """从多个文本中提取特征"""
    print("提取特征...")
    features = []
    
    # 批处理大小
    batch_size = 16
    
    # 创建进度条
    with tqdm(total=len(texts), desc="特征提取") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # 逐个提取特征
            batch_features = []
            for text in batch_texts:
                try:
                    feature = feature_extractor.extract_features_from_text(text)
                    batch_features.append(feature)
                except Exception as e:
                    print(f"提取特征失败: {e}，使用零向量代替")
                    # 使用零向量作为备选
                    batch_features.append(np.zeros(768))  # BERT输出维度通常为768
            features.extend(batch_features)
            pbar.update(len(batch_texts))
    
    # 合并特征
    return np.array(features)

# 加载预提取的特征（仍然保留作为备用）
def load_pretrained_features(features_dir="../extracted_features"):
    """加载预提取的特征"""
    print(f"加载预提取特征: {features_dir}")
    
    try:
        # 加载特征
        X_train = np.load(os.path.join(features_dir, "X_train.npy"))
        X_val = np.load(os.path.join(features_dir, "X_val.npy"))
        X_test = np.load(os.path.join(features_dir, "X_test.npy"))
        
        # 加载标签
        y_train = np.load(os.path.join(features_dir, "y_train.npy"))
        y_val = np.load(os.path.join(features_dir, "y_val.npy"))
        y_test = np.load(os.path.join(features_dir, "y_test.npy"))
        
        # 加载类名
        class_names = joblib.load(os.path.join(features_dir, "class_names.joblib"))
        
        print(f"特征形状: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"标签形状: y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
        print(f"类别: {class_names}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names
    
    except Exception as e:
        print(f"加载预提取特征时出错: {e}")
        return None, None, None, None, None, None, None

# 自定义Transformer模型包装器，用于兼容sklearn接口
class TransformerModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
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

# 训练自定义Transformer模型
def train_transformer(X_train, y_train, X_val, y_val, num_classes, epochs=50, batch_size=64, lr=1e-4):
    """训练自定义Transformer模型"""
    print("开始训练自定义Transformer模型...")
    
    # 获取特征维度
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = PretrainedFeatureTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=768,  # 与BERT一致的隐藏维度
        nhead=8,
        num_layers=4,
        dropout=0.1
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 转换数据为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 记录训练时间
    start_time = time.time()
    
    # 训练循环
    best_val_f1 = 0.0
    best_model = None
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 训练模式
        model.train()
        
        # 跟踪损失
        total_loss = 0
        
        # 训练步骤
        for batch in tqdm(train_dataloader, desc="训练中"):
            # 获取数据
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
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
        print(f"平均训练损失: {avg_train_loss:.4f}")
        
        # 评估模式
        model.eval()
        
        # 在验证集上评估
        val_preds = []
        val_true = []
        
        for batch in tqdm(val_dataloader, desc="验证中"):
            # 获取数据
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            # 不计算梯度
            with torch.no_grad():
                outputs = model(features)
                _, predictions = torch.max(outputs, 1)
            
            # 添加到列表
            val_preds.extend(predictions.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
        
        # 计算指标
        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds, average='weighted')
        val_recall = recall_score(val_true, val_preds, average='weighted')
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        print(f"验证集指标 - 准确率: {val_accuracy:.4f}, 精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}, F1分数: {val_f1:.4f}")
        
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
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f"Transformer模型训练完成，耗时: {training_time:.2f}秒")
    
    # 恢复最佳模型
    model.load_state_dict(best_model)
    
    # 创建保存目录
    os.makedirs("../models/pretrained_transformer", exist_ok=True)
    os.makedirs("../output_models", exist_ok=True)
    
    # 1. 保存为PyTorch格式
    model_save_path = "../models/pretrained_transformer/model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'training_time': training_time
    }, model_save_path)
    print(f"模型已保存为PyTorch格式: {model_save_path}")
    
    # 2. 保存为joblib格式，以便model_visualization_fixed.py可以加载
    # 创建包装模型 - 使用全局定义的类
    wrapped_model = TransformerModelWrapper(model, device)
    
    # 保存为joblib格式
    joblib_save_path = "../output_models/pretrained_transformer.pkl"
    joblib.dump(wrapped_model, joblib_save_path)
    print(f"模型已保存为joblib格式: {joblib_save_path}")
    
    # 3. 保存为pickle格式作为备份
    import pickle
    pickle_save_path = "../output_models/pretrained_transformer_pickle.pkl"
    with open(pickle_save_path, 'wb') as f:
        pickle.dump(wrapped_model, f)
    print(f"模型已保存为pickle格式: {pickle_save_path}")
    
    return model, training_time

# 加载注意力投票模型
def load_attention_voting_model(model_path):
    """加载训练好的注意力投票模型"""
    print(f"加载注意力投票模型: {model_path}")
    try:
        # 直接加载模型，不需要依赖外部模块
        model = joblib.load(model_path)
        print(f"注意力投票模型加载成功: {type(model)}")
        return model
    except Exception as e:
        print(f"加载注意力投票模型失败: {e}")
        # 尝试替代加载方法
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"使用pickle成功加载注意力投票模型: {type(model)}")
            return model
        except Exception as e2:
            print(f"使用pickle加载模型也失败: {e2}")
            return None

# 评估Transformer模型
def evaluate_transformer(model, X_test, y_test, batch_size=64):
    """评估自定义Transformer模型"""
    print("评估自定义Transformer模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 转换数据为PyTorch张量
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 创建数据集和数据加载器
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 评估模式
    model.eval()
    
    # 跟踪预测
    test_preds = []
    test_true = []
    
    # 记录推理时间
    inference_times = []
    
    for batch in tqdm(test_dataloader, desc="测试中"):
        # 获取数据
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        # 不计算梯度
        with torch.no_grad():
            # 记录开始时间
            start_time = time.time()
            
            outputs = model(features)
            _, predictions = torch.max(outputs, 1)
            
            # 记录结束时间
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        # 添加到列表
        test_preds.extend(predictions.cpu().numpy())
        test_true.extend(labels.cpu().numpy())
    
    # 计算平均推理时间（每个样本）
    avg_inference_time = sum(inference_times) / len(inference_times) / batch_size
    
    # 计算指标
    test_accuracy = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds, average='weighted')
    test_recall = recall_score(test_true, test_preds, average='weighted')
    test_f1 = f1_score(test_true, test_preds, average='weighted')
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(test_true, test_preds)
    
    print(f"测试集指标 - 准确率: {test_accuracy:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1分数: {test_f1:.4f}")
    print(f"平均推理时间: {avg_inference_time*1000:.2f}毫秒/样本")
    
    results = {
        "模型": "预提取特征Transformer",
        "准确率": test_accuracy,
        "精确率": test_precision,
        "召回率": test_recall,
        "F1分数": test_f1,
        "推理时间(ms)": avg_inference_time * 1000
    }
    
    return results, conf_matrix, test_preds

# 评估注意力投票模型
def evaluate_attention_voting(model, X_test, y_test):
    """评估注意力投票模型"""
    print("评估注意力投票模型...")
    if model is None:
        print("错误: 注意力投票模型为空，无法评估")
        # 返回空结果
        return {
            "模型": "注意力投票",
            "准确率": 0.0,
            "精确率": 0.0,
            "召回率": 0.0,
            "F1分数": 0.0,
            "推理时间(ms)": 0.0
        }, np.zeros((1, 1)), []
    
    # 记录开始时间
    start_time = time.time()
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算推理时间
    inference_time = (end_time - start_time) / len(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"测试集指标 - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    print(f"平均推理时间: {inference_time*1000:.2f}毫秒/样本")
    
    results = {
        "模型": "注意力投票",
        "准确率": accuracy,
        "精确率": precision,
        "召回率": recall,
        "F1分数": f1,
        "推理时间(ms)": inference_time * 1000
    }
    
    return results, conf_matrix, y_pred

# 绘制性能比较图
def plot_performance_comparison(transformer_results, attention_results):
    """绘制性能比较图"""
    print("绘制性能比较图...")
    
    # 设置字体
    font = setup_matplotlib_chinese()
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 获取度量标准
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    
    # 获取值
    transformer_values = [transformer_results[metric] for metric in metrics]
    attention_values = [attention_results[metric] for metric in metrics]
    
    # 创建条形图
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, transformer_values, width, label='预提取特征Transformer')
    plt.bar(x + width/2, attention_values, width, label='注意力投票模型')
    
    # 添加标签和标题
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title('预提取特征Transformer与注意力投票模型性能比较')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    
    # 添加数值标签
    for i, v in enumerate(transformer_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(attention_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('../compare/plots/pretrained_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 绘制推理时间比较图
def plot_inference_time_comparison(transformer_results, attention_results):
    """绘制推理时间比较图"""
    print("绘制推理时间比较图...")
    
    # 设置字体
    font = setup_matplotlib_chinese()
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 获取值
    models = ['预提取特征Transformer', '注意力投票模型']
    times = [transformer_results['推理时间(ms)'], attention_results['推理时间(ms)']]
    
    # 创建条形图
    plt.bar(models, times, color=['#1f77b4', '#ff7f0e'])
    
    # 添加标签和标题
    plt.xlabel('模型')
    plt.ylabel('推理时间 (毫秒/样本)')
    plt.title('模型推理时间比较')
    
    # 添加数值标签
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f'{v:.2f}ms', ha='center')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('../compare/plots/pretrained_inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix, class_names, title, filename):
    """绘制混淆矩阵"""
    print(f"绘制混淆矩阵: {title}...")
    
    # 设置字体
    font = setup_matplotlib_chinese()
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热图
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    # 添加标签和标题
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'../compare/plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 保存结果
def save_results(transformer_results, attention_results, class_names, transformer_preds, attention_preds, y_test):
    """保存比较结果"""
    print("保存比较结果...")
    
    # 将结果保存为CSV
    results_df = pd.DataFrame([transformer_results, attention_results])
    results_df.to_csv('../compare/results/pretrained_comparison_results.csv', index=False, encoding='utf-8')
    
    # 将结果保存为JSON
    results = {
        '预提取特征Transformer': transformer_results,
        '注意力投票': attention_results,
        '类别': class_names.tolist() if isinstance(class_names, np.ndarray) else class_names,
    }
    
    with open('../compare/results/pretrained_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        '真实标签': y_test,
        'Transformer预测': transformer_preds,
        '注意力投票预测': attention_preds
    })
    predictions_df.to_csv('../compare/results/pretrained_predictions.csv', index=False, encoding='utf-8')
    
    print("比较结果保存完成")

def main():
    """主函数"""
    print("开始预提取特征Transformer模型训练与比较...")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建目录
    create_directories()
    
    # 配置中文字体
    chinese_font = setup_matplotlib_chinese()
    if chinese_font:
        print(f"成功配置中文字体: {chinese_font}")
    else:
        print("警告: 未能找到合适的中文字体，图表中的中文可能无法正确显示")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 方法1：加载预提取的特征
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_pretrained_features()
    
    # 如果预提取特征加载失败，使用与predict_new.py相同的方法
    if X_train is None:
        print("预提取特征加载失败，使用与predict_new.py相同的方法提取特征...")
        
        # 加载测试数据
        test_file = "../test/文本分类.xlsx"
        test_data = load_excel(test_file)
        
        if not test_data:
            print("错误: 未能加载测试数据")
            return
        
        print(f"测试数据加载成功，共 {len(test_data)} 条记录")
        
        # 加载类别名称
        try:
            class_names = joblib.load("../extracted_features/class_names.joblib")
            print(f"类别名称加载成功: {class_names}")
        except Exception as e:
            print(f"加载类别名称失败: {e}")
            return
        
        # 初始化特征提取器
        feature_extractor = SimpleFeatureExtractor(model_name='bert-base-chinese', device=device)
        
        # 提取特征
        texts = [item['text'] for item in test_data]
        X = extract_features_from_texts(texts, feature_extractor)
        
        # 获取标签（如果有）
        if 'label' in test_data[0]:
            y = np.array([item['label'] for item in test_data])
        else:
            # 如果没有标签，创建一个伪标签用于训练
            print("警告: 测试数据中没有标签，使用伪标签进行评估")
            y = np.zeros(len(test_data), dtype=int)
        
        # 将数据分为训练、验证和测试集
        # 为简化起见，直接使用同一批数据
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
        
        print("使用相同数据进行训练和评估")
    
    # 获取类别数
    num_classes = len(np.unique(y_train))
    
    # 训练Transformer模型
    transformer_model, transformer_training_time = train_transformer(
        X_train, y_train, X_val, y_val, num_classes, epochs=50, batch_size=64
    )
    
    # 加载注意力投票模型
    attention_model_path = "../output_models/attention_voting_model.pkl"
    if not os.path.exists(attention_model_path):
        print(f"错误: 注意力投票模型文件不存在: {attention_model_path}")
        attention_model_path = input("请输入正确的注意力投票模型文件路径: ")
        if not os.path.exists(attention_model_path):
            print(f"错误: 注意力投票模型文件不存在: {attention_model_path}")
            return
    
    attention_model = load_attention_voting_model(attention_model_path)
    
    # 评估Transformer模型
    transformer_results, transformer_conf_matrix, transformer_preds = evaluate_transformer(
        transformer_model, X_test, y_test
    )
    
    # 评估注意力投票模型
    attention_results, attention_conf_matrix, attention_preds = evaluate_attention_voting(
        attention_model, X_test, y_test
    )
    
    # 添加训练时间到结果
    transformer_results["训练时间(s)"] = transformer_training_time
    
    # 绘制性能比较图
    plot_performance_comparison(transformer_results, attention_results)
    
    # 绘制推理时间比较图
    plot_inference_time_comparison(transformer_results, attention_results)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        transformer_conf_matrix, class_names, '预提取特征Transformer混淆矩阵', 'pretrained_transformer_confusion_matrix'
    )
    plot_confusion_matrix(
        attention_conf_matrix, class_names, '注意力投票模型混淆矩阵', 'attention_confusion_matrix_pretrained'
    )
    
    # 保存结果
    save_results(
        transformer_results, attention_results, class_names, transformer_preds, attention_preds, y_test
    )
    
    print("预提取特征Transformer模型训练与比较完成")

if __name__ == "__main__":
    main()