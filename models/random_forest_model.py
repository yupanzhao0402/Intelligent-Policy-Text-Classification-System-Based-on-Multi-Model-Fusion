#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import time
from tqdm import tqdm


class RandomForestModel:
    """随机森林模型"""
    def __init__(self, n_estimators=300, max_depth=20, min_samples_split=4, 
                 min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                 class_weight='balanced', random_state=42, n_jobs=-1,
                 max_samples=0.8, criterion='gini', 
                 min_impurity_decrease=1e-5, ccp_alpha=0.0):
        """
        初始化随机森林模型
        
        参数:
        n_estimators (int): 树的数量，默认300（增加以提高稳定性）
        max_depth (int): 树的最大深度，默认20（调整以适应高维特征）
        min_samples_split (int): 分裂内部节点所需的最小样本数，默认4
        min_samples_leaf (int): 叶节点所需的最小样本数，默认2
        max_features (str, int, float): 寻找最佳分裂时考虑的特征数量，'sqrt'考虑特征总数的平方根
        bootstrap (bool): 是否使用bootstrap样本，默认True
        class_weight (str, dict): 类别权重，用于处理不平衡数据集，默认'balanced'
        random_state (int): 随机种子
        n_jobs (int): 并行运行的作业数，默认-1表示使用所有处理器
        max_samples (float): bootstrap样本占总样本的比例，默认0.8
        criterion (str): 分裂质量测量标准，默认'gini'
        min_impurity_decrease (float): 节点分裂所需的最小不纯度减少，默认1e-5
        ccp_alpha (float): 控制树的复杂度的参数，默认0.0
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            max_samples=max_samples if bootstrap else None,
            criterion=criterion,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            oob_score=True if bootstrap else False,  # 使用袋外样本估计精度
            verbose=0,
            warm_start=False
        )
        self.class_names = None
        self.feature_importances_ = None
        self.train_time = None
    
    def train(self, X_train, y_train):
        """
        训练随机森林模型
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        """
        print(f"特征维度: {X_train.shape[1]}")
        start_time = time.time()
        
        # 使用进度条跟踪训练过程
        with tqdm(total=100, desc="训练随机森林") as pbar:
            self.model.fit(X_train, y_train)
            pbar.update(100)
        
        self.train_time = time.time() - start_time
        print(f"随机森林训练完成，耗时 {self.train_time:.2f} 秒")
        
        # 如果启用了OOB评分，则打印OOB分数
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
            print(f"袋外(OOB)分数: {self.model.oob_score_:.4f}")
        
        # 保存特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            # 打印前10个重要特征
            indices = np.argsort(self.feature_importances_)[-10:]
            print("模型前10个重要特征权重:")
            for i in indices[::-1]:
                print(f"特征 {i}: {self.feature_importances_[i]:.4f}")
    
    def tune_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, n_iter=20, quick_mode=False):
        """
        超参数调优
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        n_iter (int): 随机搜索的迭代次数
        quick_mode (bool): 是否使用简化的参数空间进行快速调优
        
        返回:
        dict: 最佳参数
        """
        print("正在进行随机森林超参数优化...")
        
        # 定义参数空间 - 根据quick_mode选择不同复杂度的参数空间
        if quick_mode:
            print("使用快速模式进行调优，参数空间已简化")
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [15, 20],
                'min_samples_split': [4],
                'min_samples_leaf': [2],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'class_weight': ['balanced'],
                'criterion': ['gini'],
                'max_samples': [0.8]
            }
            # 减少交叉验证折数
            cv = 2
            # 减少迭代次数
            if n_iter > 5:
                n_iter = 5
                print(f"已将迭代次数降低为{n_iter}以加快优化速度")
        else:
            param_grid = {
                'n_estimators': [200, 300, 400, 500],
                'max_depth': [10, 15, 20, 25, 30, None],
                'min_samples_split': [2, 4, 6, 8],
                'min_samples_leaf': [1, 2, 3, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7],
                'bootstrap': [True],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'criterion': ['gini', 'entropy'],
                'max_samples': [0.7, 0.8, 0.9, 1.0]
            }
            cv = 5
        
        # 创建模型和随机搜索对象
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            rf, param_grid, n_iter=n_iter, cv=cv,
            scoring='accuracy', random_state=42, n_jobs=-1,
            verbose=1
        )
        
        # 执行搜索
        search.fit(X_train, y_train)
        
        # 输出最佳参数和分数
        print(f"最佳参数: {search.best_params_}")
        print(f"最佳交叉验证分数: {search.best_score_:.4f}")
        
        # 使用最佳参数更新模型
        self.model = search.best_estimator_
        
        # 如果提供了验证集，则评估最佳模型
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_pred)
            print(f"验证集准确率: {val_accuracy:.4f}")
        
        return search.best_params_
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_pred (np.ndarray): 预测标签
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_proba (np.ndarray): 预测概率
        """
        return self.model.predict_proba(X)
    
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
        
        print("随机森林模型评估结果:")
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
        joblib.dump(self.model, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        参数:
        filepath (str): 加载路径
        """
        self.model = joblib.load(filepath)
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
        plt.title('随机森林模型混淆矩阵')
        
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
    
    def feature_importance(self):
        """
        获取特征重要性
        
        返回:
        feature_importance (np.ndarray): 特征重要性
        """
        if self.feature_importances_ is not None:
            return self.feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            return self.feature_importances_
        else:
            raise AttributeError("模型没有feature_importances_属性")
    
    def plot_feature_importance(self, top_n=20):
        """
        绘制特征重要性
        
        参数:
        top_n (int): 显示的特征数量
        """
        # 获取特征重要性
        importance = self.feature_importance()
        
        # 仅显示前top_n个特征
        if top_n > 0 and top_n < len(importance):
            indices = np.argsort(importance)[-top_n:]
            importance = importance[indices]
            feature_idx = indices
        else:
            feature_idx = np.arange(len(importance))
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance)), importance[::-1])
        plt.yticks(range(len(importance)), [f"特征 {i}" for i in feature_idx[::-1]])
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title('随机森林模型特征重要性 (Top {})'.format(top_n))
        plt.tight_layout()
        
        return plt


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
    
    # 提取特征
    feature_extractor = FeatureExtractor()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = processor.get_dataloaders()
    
    # 提取特征
    print("提取训练集特征...")
    X_train, y_train = feature_extractor.extract_features(train_loader)
    
    print("提取测试集特征...")
    X_test, y_test = feature_extractor.extract_features(test_loader)
    
    # 创建并训练随机森林模型
    rf_model = RandomForestModel()
    print("训练随机森林模型...")
    rf_model.train(X_train, y_train)
    
    # 设置类别名称
    rf_model.set_class_names(processor.get_class_names())
    
    # 评估模型
    metrics = rf_model.evaluate(X_test, y_test)
    
    # 绘制混淆矩阵
    rf_model.plot_confusion_matrix(X_test, y_test)
    plt.savefig('rf_confusion_matrix.png')
    
    # 打印分类报告
    rf_model.print_classification_report(X_test, y_test)
    
    # 绘制特征重要性
    rf_model.plot_feature_importance()
    plt.savefig('rf_feature_importance.png')
    
    # 保存模型
    rf_model.save_model('models/rf_model.joblib') 