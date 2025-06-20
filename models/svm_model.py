#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import joblib
import os
from tqdm import tqdm

try:
    from sklearn.utils import parallel_backend
except ImportError:
    print("警告: sklearn.utils.parallel_backend未能导入，将使用默认后端")

class SVMModel:
    """SVM分类器模型"""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', class_weight=None, 
                 probability=True, random_state=42, max_iter=1000, cache_size=2000,
                 shrinking=True, tol=1e-3, decision_function_shape='ovr',
                 degree=3, coef0=0.0, n_jobs=None):
        """
        初始化SVM模型
        
        参数:
        C (float): 正则化参数
        kernel (str): 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
        gamma (str/float): 'scale'、'auto'或浮点数
        class_weight (None/dict/str): 类别权重
        probability (bool): 是否启用概率估计
        random_state (int): 随机种子
        max_iter (int): 最大迭代次数
        cache_size (float): 缓存大小(MB)
        shrinking (bool): 是否使用收缩启发式
        tol (float): 停止条件的容差
        decision_function_shape (str): 决策函数形状
        degree (int): 多项式核的度
        coef0 (float): 核函数的独立项
        """
        # 保存模型参数
        self.params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'class_weight': class_weight,
            'probability': probability,
            'random_state': random_state,
            'max_iter': max_iter,
            'cache_size': cache_size,
            'shrinking': shrinking,
            'tol': tol,
            'decision_function_shape': decision_function_shape,
            'degree': degree,
            'coef0': coef0
        }
        
        # 将n_jobs传递给模型
        self.n_jobs = n_jobs if n_jobs is not None else -1
        
        # 创建SVM模型
        self.model = SVC(**self.params)
        
        # 初始化其他属性
        self.class_names = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.use_feature_selection = False
        self.pca = None
        self.use_pca = False
        self.is_trained = False
        
    def train(self, X_train, y_train, use_scaling=True, use_feature_selection=False, 
              feature_selection_method='l1', n_features_to_select=None,
              use_pca=False, pca_components=None):
        """
        训练SVM模型
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        use_scaling (bool): 是否使用特征缩放
        use_feature_selection (bool): 是否使用特征选择
        feature_selection_method (str): 特征选择方法 ('l1' 或 'rfe')
        n_features_to_select (int): 要选择的特征数量
        use_pca (bool): 是否使用PCA降维
        pca_components (int): PCA组件数量
        
        返回:
        self: 训练好的模型
        """
        start_time = time.time()
        print(f"开始训练SVM模型，数据形状: {X_train.shape}")
        
        # 记录原始特征维度
        original_dim = X_train.shape[1]
        print(f"原始特征维度: {original_dim}")
        
        # 应用特征缩放
        self.use_scaling = use_scaling
        if use_scaling:
            print("应用特征缩放...")
            X_train = self.scaler.fit_transform(X_train)
        
        # 特征处理后的数据
        processed_X_train = X_train.copy()
        
        # 应用PCA降维 - 优先于特征选择
        self.use_pca = use_pca
        if use_pca:
            print("应用PCA降维...")
            
            # 确定PCA组件数量
            if pca_components is None:
                # 默认取原始维度的一半
                pca_components = min(int(original_dim * 0.5), X_train.shape[0])
                print(f"自动选择PCA组件数: {pca_components}")
            
            # 创建并应用PCA
            self.pca = PCA(n_components=pca_components, random_state=42)
            processed_X_train = self.pca.fit_transform(X_train)
            
            explained_var = sum(self.pca.explained_variance_ratio_)
            print(f"PCA降维后维度: {processed_X_train.shape[1]}")
            print(f"保留的方差比例: {explained_var:.4f}")
        
        # 应用特征选择（如果没有使用PCA）
        self.use_feature_selection = use_feature_selection and not use_pca
        if self.use_feature_selection:
            print(f"应用{feature_selection_method}特征选择...")
            
            # 设置要选择的特征数量
            if n_features_to_select is None:
                # 默认为20%的特征
                n_features_to_select = max(int(original_dim * 0.2), 10)
                print(f"自动设置特征选择数量: {n_features_to_select}")
            
            if feature_selection_method == 'l1':
                from sklearn.linear_model import LogisticRegression
                
                # 使用L1正则化进行特征选择
                l1_model = LogisticRegression(penalty='l1', solver='saga', C=0.1, 
                                           max_iter=1000, random_state=42, n_jobs=self.n_jobs)
                
                # 创建选择器
                selector = SelectFromModel(
                    l1_model,
                    max_features=n_features_to_select,
                    threshold=-np.inf  # 仅根据max_features选择
                )
                
                # 拟合选择器
                with tqdm(total=100, desc="L1特征选择") as pbar:
                    selector.fit(X_train, y_train)
                    pbar.update(100)
                
                # 获取选择的特征
                selected_features = selector.get_support()
                processed_X_train = selector.transform(X_train)
                
                # 保存特征选择器
                self.feature_selector = selector
                self.selected_features = selected_features
                
                print(f"特征数量从 {original_dim} 减少到 {processed_X_train.shape[1]}")
                
            elif feature_selection_method == 'rfe':
                # 使用递归特征消除
                print("使用递归特征消除法...")
                
                # 创建基模型
                base_model = SVC(kernel='linear', random_state=42)
                
                # 创建RFE选择器
                selector = RFE(
                    estimator=base_model,
                    n_features_to_select=n_features_to_select,
                    step=0.1,  # 每次移除10%的特征
                )
                
                # 拟合选择器
                with tqdm(total=100, desc="RFE特征选择") as pbar:
                    selector.fit(X_train, y_train)
                    pbar.update(100)
                
                # 获取选择的特征
                selected_features = selector.support_
                processed_X_train = selector.transform(X_train)
                
                # 保存特征选择器
                self.feature_selector = selector
                self.selected_features = selected_features
                
                print(f"特征数量从 {original_dim} 减少到 {processed_X_train.shape[1]}")
            else:
                print(f"未知的特征选择方法: {feature_selection_method}，使用所有特征")
        
        # 训练模型
        print(f"使用 {processed_X_train.shape[1]} 个特征训练SVM...")
        
        # 尝试使用并行后端加速
        if hasattr(self, 'n_jobs') and self.n_jobs != 1:
            try:
                with parallel_backend("loky", n_jobs=self.n_jobs):
                    with tqdm(total=100, desc="训练SVM") as pbar:
                        self.model.fit(processed_X_train, y_train)
                        pbar.update(100)
            except ImportError:
                # 旧版sklearn，直接训练
                with tqdm(total=100, desc="训练SVM") as pbar:
                    self.model.fit(processed_X_train, y_train)
                    pbar.update(100)
        else:
            # 普通训练
            with tqdm(total=100, desc="训练SVM") as pbar:
                self.model.fit(processed_X_train, y_train)
                pbar.update(100)
        
        training_time = time.time() - start_time
        print(f"SVM模型训练完成，耗时 {training_time:.2f} 秒")
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """预测类别"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
        
        # 应用特征处理
        X_processed = self._preprocess_features(X)
            
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """预测类别概率"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
        
        # 应用特征处理
        X_processed = self._preprocess_features(X)
        
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X):
        """预处理特征"""
        X_processed = X.copy()
        
        # 应用特征缩放
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        # 应用PCA降维
        if self.use_pca and self.pca is not None:
            X_processed = self.pca.transform(X_processed)
        
        # 应用特征选择
        elif self.use_feature_selection and self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
            
        return X_processed
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
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
        
        print("SVM模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return metrics
    
    def set_class_names(self, class_names):
        """设置类别名称"""
        self.class_names = class_names
    
    def save_model(self, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 创建包含所有必要对象的字典
        model_data = {
            'model': self.model,
            'class_names': self.class_names,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'use_feature_selection': self.use_feature_selection,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"SVM模型及相关组件已保存到 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        # 加载所有组件
        self.model = model_data['model']
        self.class_names = model_data['class_names']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data['selected_features']
        self.use_feature_selection = model_data['use_feature_selection']
        self.pca = model_data['pca']
        self.use_pca = model_data['use_pca']
        self.is_trained = model_data['is_trained']
        
        print(f"SVM模型及相关组件已从 {filepath} 加载")
    
    def tune_hyperparameters(self, X_train, y_train, X_val=None, y_val=None):
        """
        超参数调优
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        
        返回:
        best_params (dict): 最佳参数
        """
        print("开始SVM超参数优化...")
        
        # 应用特征缩放
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly'],
            'class_weight': ['balanced', None]
        }
        
        # 创建GridSearchCV
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42, max_iter=1000, cache_size=2000),
            param_grid,
            cv=3,  # 减少交叉验证折数以加快速度
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        # 执行网格搜索
        grid_search.fit(X_train_scaled, y_train)
        
        # 获取最佳参数
        best_params = grid_search.best_params_
        
        print(f"最佳参数: {best_params}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 使用最佳参数更新模型
        self.model = SVC(**best_params, probability=True, random_state=42)
        
        return best_params
    
    def plot_confusion_matrix(self, X_test, y_test):
        """绘制混淆矩阵"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names if self.class_names else None,
                   yticklabels=self.class_names if self.class_names else None)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('SVM混淆矩阵')
        plt.tight_layout()
    
    def print_classification_report(self, X_test, y_test):
        """打印分类报告"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 打印分类报告
        report = classification_report(y_test, y_pred, target_names=self.class_names if self.class_names else None)
        print(report)
    
    def create_pipeline(self, use_scaling=True, use_pca=False, pca_components=None):
        """创建模型管道"""
        from sklearn.pipeline import Pipeline
        steps = []
        
        # 添加特征缩放
        if use_scaling:
            steps.append(('scaler', StandardScaler()))
        
        # 添加PCA
        if use_pca:
            if pca_components is None:
                steps.append(('pca', PCA(random_state=42)))
            else:
                steps.append(('pca', PCA(n_components=pca_components, random_state=42)))
        
        # 添加SVM模型
        steps.append(('svm', SVC(**self.params)))
        
        # 创建管道
        pipeline = Pipeline(steps)
        
        return pipeline


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
    
    # 创建并训练SVM模型
    svm_model = SVMModel()
    print("训练SVM模型...")
    svm_model.train(X_train, y_train)
    
    # 设置类别名称
    svm_model.set_class_names(processor.get_class_names())
    
    # 评估模型
    metrics = svm_model.evaluate(X_test, y_test)
    
    # 绘制混淆矩阵
    svm_model.plot_confusion_matrix(X_test, y_test)
    plt.savefig('svm_confusion_matrix.png')
    
    # 打印分类报告
    svm_model.print_classification_report(X_test, y_test)
    
    # 保存模型
    svm_model.save_model('models/svm_model.joblib') 