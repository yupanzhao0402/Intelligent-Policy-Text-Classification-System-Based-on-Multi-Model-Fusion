#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import time
import joblib
import os
from tqdm import tqdm

class LogisticRegressionModel:
    """逻辑回归分类器"""
    
    def __init__(self, C=2.0, max_iter=3000, solver='saga', penalty='l1', 
                 class_weight='balanced', random_state=42, n_jobs=-1, 
                 warm_start=True, tol=1e-3, l1_ratio=None, dual=False,
                 fit_intercept=True, intercept_scaling=1.0, multi_class='auto'):
        """
        初始化逻辑回归模型
        
        参数:
        C (float): 正则化强度的倒数，值越小正则化越强
        max_iter (int): 最大迭代次数
        solver (str): 求解器类型 ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
        penalty (str): 正则化类型 ('l1', 'l2', 'elasticnet', 'none')
        class_weight (str/dict): 类别权重
        random_state (int): 随机种子
        n_jobs (int): 并行任务数，-1代表使用所有处理器
        warm_start (bool): 是否使用上次训练的结果作为初始值
        tol (float): 停止条件的容差
        l1_ratio (float): elasticnet混合参数，0<=l1_ratio<=1
        dual (bool): 是否使用对偶形式
        fit_intercept (bool): 是否计算截距
        intercept_scaling (float): 截距缩放系数
        multi_class (str): 多分类策略 ('auto', 'ovr', 'multinomial')
        """
        params = {
            'C': C,
            'max_iter': max_iter,
            'solver': solver,
            'penalty': penalty,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'warm_start': warm_start,
            'tol': tol,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'multi_class': multi_class,
            'dual': dual
        }
        
        # 处理elasticnet特定参数
        if penalty == 'elasticnet' and l1_ratio is not None:
            if solver != 'saga':
                print("警告: elasticnet惩罚只能与saga求解器一起使用，已自动将求解器更改为saga")
                params['solver'] = 'saga'
            params['l1_ratio'] = l1_ratio
            
        # 检查liblinear求解器的约束条件
        if solver == 'liblinear':
            if penalty not in ['l1', 'l2']:
                print("警告: liblinear求解器只支持l1或l2惩罚，已自动将惩罚更改为l2")
                params['penalty'] = 'l2'
        
        # 检查penalty和solver的兼容性
        if penalty == 'none' and solver not in ['newton-cg', 'sag', 'lbfgs', 'saga']:
            print("警告: 'none'惩罚不能与liblinear求解器一起使用，已自动将求解器更改为saga")
            params['solver'] = 'saga'
        
        # 创建模型
        self.model = LogisticRegression(**params)
        self.class_names = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None
        self.use_feature_selection = False
        self.pca = None
        self.use_pca = False
        self.training_time = None
        self.convergence_status = None
        self.params = params
        self.is_trained = False
        
        print(f"逻辑回归模型初始化完成，求解器: {solver}, 惩罚: {penalty}")
        
    def train(self, X_train, y_train, use_scaling=True, use_feature_selection=False, 
              feature_selection_method='l1', n_features_to_select=None, 
              use_pca=False, pca_components=None, verbose=1):
        """
        训练模型
        
        参数:
        X_train: 训练特征
        y_train: 训练标签
        use_scaling: 是否使用特征缩放
        use_feature_selection: 是否使用特征选择
        feature_selection_method: 特征选择方法，可以是'l1'或'rfe'
        n_features_to_select: 要选择的特征数量，None表示自动选择
        use_pca: 是否使用PCA降维
        pca_components: PCA组件数量，None表示自动选择
        verbose: 详细程度，0表示静默，1表示显示进度
        
        返回:
        self: 训练好的模型实例
        """
        start_time = time.time()
        
        if verbose >= 1:
            print("开始训练逻辑回归模型...")
            print(f"数据形状: {X_train.shape}, 类别数: {len(np.unique(y_train))}")
        
        # 记录原始特征维度
        original_dim = X_train.shape[1]
        if verbose >= 1:
            print(f"原始特征维度: {original_dim}")
        
        # 应用特征缩放
        self.use_scaling = use_scaling
        if use_scaling:
            if verbose >= 1:
                print("应用特征缩放...")
            X_train = self.scaler.fit_transform(X_train)
        
        # 特征处理后的数据
        processed_X_train = X_train.copy()
            
        # 应用PCA降维 - 优先于特征选择
        self.use_pca = use_pca
        if use_pca:
            if verbose >= 1:
                print("应用PCA降维...")
                
            # 确定PCA组件数量
            if pca_components is None:
                # 默认取原始维度的一半，但不超过样本数
                pca_components = min(int(original_dim * 0.5), X_train.shape[0])
                if verbose >= 1:
                    print(f"自动选择PCA组件数: {pca_components}")
            
            # 创建并应用PCA
            self.pca = PCA(n_components=pca_components, random_state=42)
            processed_X_train = self.pca.fit_transform(X_train)
            
            if verbose >= 1:
                explained_var = sum(self.pca.explained_variance_ratio_)
                print(f"PCA降维后维度: {processed_X_train.shape[1]}")
                print(f"保留的方差比例: {explained_var:.4f}")

        # 应用特征选择 (如果没有使用PCA)
        self.use_feature_selection = use_feature_selection and not use_pca
        if self.use_feature_selection:
            if verbose >= 1:
                print(f"应用{feature_selection_method}特征选择...")
            
            # 设置要选择的特征数量
            if n_features_to_select is None:
                # 默认选择30%的特征，但至少10个
                n_features_to_select = max(int(original_dim * 0.3), 10)
                if verbose >= 1:
                    print(f"自动设置特征选择数量: {n_features_to_select}")
            
            if feature_selection_method == 'l1':
                # 使用L1正则化进行特征选择
                l1_model = LogisticRegression(
                    penalty='l1', C=0.1, solver='saga', 
                    max_iter=1000, random_state=42, tol=1e-3, n_jobs=-1
                )
                
                # 创建选择器
                selector = SelectFromModel(
                    l1_model,
                    max_features=n_features_to_select,
                    threshold=-np.inf  # 仅根据max_features选择
                )
                
                # 拟合选择器
                with tqdm(total=100, desc="L1特征选择", disable=verbose<1) as pbar:
                    selector.fit(X_train, y_train)
                    pbar.update(100)
                
                # 获取选择的特征
                selected_features = selector.get_support()
                processed_X_train = selector.transform(X_train)
                
                # 保存特征选择器
                self.feature_selector = selector
                self.selected_features = selected_features
                
                if verbose >= 1:
                    print(f"特征数量从 {original_dim} 减少到 {processed_X_train.shape[1]}")
                
            elif feature_selection_method == 'rfe':
                # 使用递归特征消除
                if verbose >= 1:
                    print("使用递归特征消除法...")
                
                # 对于高维数据，调整步长以加快计算
                step = max(10, X_train.shape[1] // 100)
                
                # 创建基模型
                base_model = LogisticRegression(
                    max_iter=500, random_state=42, n_jobs=-1
                )
                
                # 创建RFE选择器
                selector = RFE(
                    estimator=base_model,
                    n_features_to_select=n_features_to_select,
                    step=step,
                    verbose=max(0, verbose-1)
                )
                
                # 拟合选择器
                with tqdm(total=100, desc="RFE特征选择", disable=verbose<1) as pbar:
                    selector.fit(X_train, y_train)
                    pbar.update(100)
                
                # 获取选择的特征
                selected_features = selector.support_
                processed_X_train = selector.transform(X_train)
                
                # 保存特征选择器
                self.feature_selector = selector
                self.selected_features = selected_features
                
                if verbose >= 1:
                    print(f"特征数量从 {original_dim} 减少到 {processed_X_train.shape[1]}")
            else:
                if verbose >= 1:
                    print(f"未知的特征选择方法: {feature_selection_method}，使用所有特征训练模型")
        
        # 使用处理后的特征训练模型
        if verbose >= 1:
            print(f"使用 {processed_X_train.shape[1]} 个特征训练逻辑回归模型...")
        
        with tqdm(total=100, desc="训练逻辑回归", disable=verbose<1) as pbar:
            self.model.fit(processed_X_train, y_train)
            pbar.update(100)
        
        # 检查收敛状态
        if hasattr(self.model, 'n_iter_'):
            self.convergence_status = {
                'n_iter': self.model.n_iter_,
                'converged': False  # 默认假设未收敛
            }
            
            # 如果n_iter_是数组（多分类情况），检查是否所有类别都收敛
            if isinstance(self.model.n_iter_, np.ndarray):
                self.convergence_status['converged'] = np.all(self.model.n_iter_ < self.model.max_iter)
            else:
                self.convergence_status['converged'] = self.model.n_iter_ < self.model.max_iter
            
            if verbose >= 1:
                if self.convergence_status['converged']:
                    print(f"模型已收敛，迭代次数: {self.model.n_iter_}")
        else:
                    print(f"警告：模型未收敛，达到最大迭代次数 {self.model.max_iter}，考虑增加max_iter参数")
        
        # 记录结束时间
        self.training_time = time.time() - start_time
        
        if verbose >= 1:
            minutes = int(self.training_time / 60)
            seconds = int(self.training_time % 60)
            print(f"逻辑回归模型训练完成，耗时: {minutes}分{seconds}秒")
        
        # 标记模型已训练
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        X: 特征
        
        返回:
        y_pred: 预测标签
        """
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        # 应用相同的预处理
        X_processed = self._preprocess_features(X)
        
        # 返回预测结果
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        X: 特征
        
        返回:
        y_proba: 预测概率
        """
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        # 应用相同的预处理
        X_processed = self._preprocess_features(X)
        
        # 返回概率
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X):
        """
        预处理特征
        
        参数:
        X: 原始特征
        
        返回:
        X_processed: 预处理后的特征
        """
        X_processed = X.copy()
        
        # 应用特征缩放
        if hasattr(self, 'use_scaling') and self.use_scaling and self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        # 应用PCA
        if self.use_pca and self.pca is not None:
            X_processed = self.pca.transform(X_processed)
        
        # 应用特征选择
        elif self.use_feature_selection and self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
            
        return X_processed
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        
        返回:
        metrics: 评估指标字典
        """
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        # 预测
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print("逻辑回归模型评估结果:")
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
        if not self.is_trained:
            print("警告：保存未训练的模型")
            
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 创建保存字典
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'use_feature_selection': self.use_feature_selection,
            'use_scaling': getattr(self, 'use_scaling', True),
            'class_names': self.class_names,
            'params': self.params,
            'training_time': self.training_time,
            'convergence_status': self.convergence_status,
            'is_trained': self.is_trained
        }
        
        # 保存模型
        joblib.dump(save_dict, filepath)
        print(f"模型已保存到 {filepath}")
        
    def load_model(self, filepath):
        """加载模型"""
        # 加载模型
        save_dict = joblib.load(filepath)
        
        # 恢复模型状态
        self.model = save_dict['model']
        self.scaler = save_dict.get('scaler')
        self.feature_selector = save_dict.get('feature_selector')
        self.selected_features = save_dict.get('selected_features')
        self.pca = save_dict.get('pca')
        self.use_pca = save_dict.get('use_pca', False)
        self.use_feature_selection = save_dict.get('use_feature_selection', False)
        self.use_scaling = save_dict.get('use_scaling', True)
        self.class_names = save_dict.get('class_names')
        self.params = save_dict.get('params', {})
        self.training_time = save_dict.get('training_time')
        self.convergence_status = save_dict.get('convergence_status')
        self.is_trained = save_dict.get('is_trained', True)
        
        print(f"模型已从 {filepath} 加载")
        
    def plot_confusion_matrix(self, X_test, y_test):
        """绘制混淆矩阵"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        # 预测
        y_pred = self.predict(X_test)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 配置混淆矩阵可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else None,
            yticklabels=self.class_names if self.class_names else None
        )
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('逻辑回归混淆矩阵')
        plt.tight_layout()
    
    def print_classification_report(self, X_test, y_test):
        """打印分类报告"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        # 预测
        y_pred = self.predict(X_test)
        
        # 打印分类报告
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names if self.class_names else None
        )
        print("逻辑回归分类报告:")
        print(report)
        
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=3, 
                            method='grid', n_iter=10, scoring='accuracy', 
                            verbose=1):
        """
        超参数调优
        
        参数:
        X_train: 训练特征
        y_train: 训练标签
        param_grid: 参数网格，如果为None则使用默认网格
        cv: 交叉验证折数
        method: 调优方法，'grid'或'random'
        n_iter: 随机搜索迭代次数
        scoring: 评分标准
        verbose: 详细程度
        
        返回:
        best_params: 最佳参数
        """
        # 应用特征预处理
        X_processed = X_train.copy()
        
        # 特征缩放
        X_processed = self.scaler.fit_transform(X_processed)
        
        if param_grid is None:
            # 默认参数网格
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'] if self.model.solver in ['liblinear', 'saga'] else ['l2'],
                'solver': ['liblinear', 'saga'] if 'l1' in param_grid.get('penalty', ['l2']) else ['lbfgs', 'sag', 'newton-cg', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [1000, 2000, 5000]
            }
            
            # 如果求解器是saga，添加elasticnet选项
            if 'saga' in param_grid.get('solver', []):
                param_grid['penalty'].append('elasticnet')
                param_grid['l1_ratio'] = [0.2, 0.5, 0.8]
        
        print(f"开始{method}超参数调优，交叉验证折数：{cv}")
        
        if method == 'grid':
            search = GridSearchCV(
                LogisticRegression(random_state=42),
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=verbose
            )
        else:
            search = RandomizedSearchCV(
                LogisticRegression(random_state=42),
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=verbose
            )
        
        # 执行参数搜索
        search.fit(X_processed, y_train)
        
        # 获取最佳参数
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"最佳{scoring}分数: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        # 更新模型参数
        self.model = search.best_estimator_
        self.params = best_params
        
        return best_params
    
    def create_pipeline(self, use_scaling=True, use_pca=False, pca_components=None):
        """创建模型流水线"""
        steps = []
        
        if use_scaling:
            steps.append(('scaler', StandardScaler()))
        
        if use_pca:
            steps.append(('pca', PCA(n_components=pca_components, random_state=42)))
        
        steps.append(('logreg', self.model))
        
        return Pipeline(steps)
        
if __name__ == "__main__":
    # 测试代码
    from sklearn.datasets import make_classification
    
    # 生成模拟数据
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=5, 
        random_state=42
    )
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建和训练模型
    model = LogisticRegressionModel()
    model.train(X_train, y_train, use_scaling=True, use_feature_selection=True)
    
    # 评估模型
    metrics = model.evaluate(X_test, y_test)
    print("评估指标:", metrics)
    
    # 绘制混淆矩阵
    model.plot_confusion_matrix(X_test, y_test)
    plt.show()
    
    # 打印分类报告
    model.print_classification_report(X_test, y_test)
    
    # 绘制特征重要性
    model.plot_feature_importance()
    plt.show() 