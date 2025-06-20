#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import BertModel, AutoModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# 设置中文字体和负号显示
plt.rcParams["font.family"] = "SimHei"  # 设置默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import joblib


class FeatureExtractor:
    """文本特征提取器，基于预训练的BERT模型"""
    def __init__(self, model_name='bert-base-chinese', device=None):
        """
        初始化特征提取器
        
        参数:
        model_name (str): 预训练模型名称
        device (str): 计算设备，可以是'cuda'或'cpu'
        """
        self.model_name = model_name
        
        # 设置计算设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型
        try:
            # 尝试从本地加载
            print(f"尝试从本地加载预训练模型 {model_name}...")
            if 'bert' in model_name.lower():
                self.model = BertModel.from_pretrained(model_name, local_files_only=True)
            else:
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            print(f"成功从本地加载预训练模型 {model_name}")
        except Exception as e:
            print(f"无法从本地加载模型，错误: {e}")
            print("尝试从在线源下载模型...")
            # 尝试在线下载
            try:
                if 'bert' in model_name.lower():
                    self.model = BertModel.from_pretrained(model_name)
                else:
                    self.model = AutoModel.from_pretrained(model_name)
                print(f"成功从在线源下载预训练模型 {model_name}")
            except Exception as e2:
                print(f"在线下载失败，错误: {e2}")
                raise ValueError(f"无法加载模型 {model_name}，请检查网络连接或手动下载模型到本地。")
            
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
    def extract_features(self, dataloader, pooling_strategy='cls', layers=None):
        """
        从文本中提取特征
        
        参数:
        dataloader: 数据加载器，包含输入文本
        pooling_strategy (str): 池化策略，可以是'cls', 'mean', 'max'
        layers (list): 要使用的层的索引列表，如果为None则使用最后一层
        
        返回:
        features (np.ndarray): 特征数组
        labels (np.ndarray): 标签数组
        """
        features = []
        labels = []
        
        # 确保模型处于评估模式
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 获取模型的所有隐藏层
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # 选择层
                hidden_states = outputs.hidden_states
                if layers is None:
                    # 如果没有指定层，则使用最后一层
                    selected_layers = [hidden_states[-1]]
                else:
                    # 否则使用指定的层
                    selected_layers = [hidden_states[i] for i in layers]
                
                # 对选定的层取平均
                layer_avg = torch.stack(selected_layers).mean(dim=0)
                
                # 根据池化策略提取特征
                if pooling_strategy == 'cls':
                    # 使用[CLS]标记的表示作为整个序列的表示
                    batch_features = layer_avg[:, 0]
                elif pooling_strategy == 'mean':
                    # 对序列中的所有标记进行平均，考虑注意力掩码
                    batch_features = torch.sum(layer_avg * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
                elif pooling_strategy == 'max':
                    # 对序列中的所有标记进行最大池化，考虑注意力掩码
                    masked = layer_avg * attention_mask.unsqueeze(-1)
                    masked[~attention_mask.bool().unsqueeze(-1)] = float('-inf')
                    batch_features = torch.max(masked, dim=1)[0]
                else:
                    raise ValueError(f"不支持的池化策略: {pooling_strategy}")
                
                # 转移到CPU并添加到列表
                features.append(batch_features.cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())
        
        # 连接所有特征和标签
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return features, labels
    
    def save_features(self, features, labels, save_dir='features'):
        """
        保存提取的特征和标签
        
        参数:
        features (np.ndarray): 特征数组
        labels (np.ndarray): 标签数组
        save_dir (str): 保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        model_name_short = self.model_name.split('/')[-1]
        filename = f"{model_name_short}_features"
        
        # 保存特征和标签
        np.savez(
            os.path.join(save_dir, filename),
            features=features,
            labels=labels
        )
        
        print(f"特征已保存到 {os.path.join(save_dir, filename)}.npz")
    
    def load_features(self, filepath):
        """
        加载特征和标签
        
        参数:
        filepath (str): npz文件路径
        
        返回:
        features (np.ndarray): 特征数组
        labels (np.ndarray): 标签数组
        """
        data = np.load(filepath)
        return data['features'], data['labels']
    
    def visualize_features_pca(self, features, labels, class_names=None, n_components=2):
        """
        使用PCA可视化特征
        
        参数:
        features (np.ndarray): 特征数组
        labels (np.ndarray): 标签数组
        class_names (list): 类别名称列表
        n_components (int): PCA组件数量
        """
        # 使用PCA降维
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='tab20', alpha=0.7)
        
        # 添加图例
        if class_names is not None:
            handles, _ = scatter.legend_elements()
            plt.legend(handles, class_names, loc='best', title='类别')
        
        # 添加标题和标签
        plt.title('特征PCA可视化')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
        
        plt.tight_layout()
        return plt
    
    def visualize_features_tsne(self, features, labels, class_names=None, perplexity=30):
        """
        使用t-SNE可视化特征
        
        参数:
        features (np.ndarray): 特征数组
        labels (np.ndarray): 标签数组
        class_names (list): 类别名称列表
        perplexity (int): t-SNE的困惑度参数
        """
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab20', alpha=0.7)
        
        # 添加图例
        if class_names is not None:
            handles, _ = scatter.legend_elements()
            plt.legend(handles, class_names, loc='best', title='类别')
        
        # 添加标题和标签
        plt.title('特征t-SNE可视化 (perplexity=%d)' % perplexity)
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        
        plt.tight_layout()
        return plt
    
    def visualize_features_umap(self, features, labels, class_names=None, n_neighbors=15):
        """
        使用UMAP可视化特征
        
        参数:
        features (np.ndarray): 特征数组
        labels (np.ndarray): 标签数组
        class_names (list): 类别名称列表
        n_neighbors (int): UMAP的邻居数量参数
        """
        # 使用UMAP降维
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        features_umap = reducer.fit_transform(features)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=labels, cmap='tab20', alpha=0.7)
        
        # 添加图例
        if class_names is not None:
            handles, _ = scatter.legend_elements()
            plt.legend(handles, class_names, loc='best', title='类别')
        
        # 添加标题和标签
        plt.title('特征UMAP可视化 (n_neighbors=%d)' % n_neighbors)
        plt.xlabel('UMAP 维度 1')
        plt.ylabel('UMAP 维度 2')
        
        plt.tight_layout()
        return plt
    
    def get_pca_explained_variance(self, features, n_components=10):
        """
        计算PCA的解释方差比
        
        参数:
        features (np.ndarray): 特征数组
        n_components (int): PCA组件数量
        
        返回:
        explained_variance_ratio (np.ndarray): 解释方差比数组
        """
        pca = PCA(n_components=n_components)
        pca.fit(features)
        
        # 可视化解释方差比
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'r-')
        plt.xlabel('主成分')
        plt.ylabel('解释方差比')
        plt.title('PCA解释方差比')
        plt.xticks(range(1, n_components + 1))
        plt.grid(True)
        
        return plt


if __name__ == "__main__":
    from data_processor import DataProcessor
    
    # 测试代码
    processor = DataProcessor("sample_data.csv", sample_size=None)
    train_loader, val_loader, test_loader = processor.get_dataloaders(batch_size=32)
    
    # 初始化特征提取器
    feature_extractor = FeatureExtractor()
    
    # 从训练集提取特征
    print("从训练集提取特征...")
    train_features, train_labels = feature_extractor.extract_features(train_loader)
    
    # 从验证集提取特征
    print("从验证集提取特征...")
    val_features, val_labels = feature_extractor.extract_features(val_loader)
    
    # 从测试集提取特征
    print("从测试集提取特征...")
    test_features, test_labels = feature_extractor.extract_features(test_loader)
    
    # 保存特征
    os.makedirs("./extracted_features", exist_ok=True)
    np.save("./extracted_features/X_train.npy", train_features)
    np.save("./extracted_features/y_train.npy", train_labels)
    np.save("./extracted_features/X_val.npy", val_features)
    np.save("./extracted_features/y_val.npy", val_labels)
    np.save("./extracted_features/X_test.npy", test_features)
    np.save("./extracted_features/y_test.npy", test_labels)
    
    # 保存类别名称
    class_names = processor.get_class_names()
    joblib.dump(class_names, "./extracted_features/class_names.joblib")
    
    print("特征保存成功。")
    
    # 可视化特征可选
    os.makedirs("visualizations", exist_ok=True)
    feature_extractor.visualize_features_pca(train_features, train_labels, class_names).savefig('visualizations/pca.png')
    feature_extractor.visualize_features_tsne(train_features, train_labels, class_names).savefig('visualizations/tsne.png')
    feature_extractor.visualize_features_umap(train_features, train_labels, class_names).savefig('visualizations/umap.png')
    feature_extractor.get_pca_explained_variance(train_features).savefig('visualizations/pca_variance.png') 