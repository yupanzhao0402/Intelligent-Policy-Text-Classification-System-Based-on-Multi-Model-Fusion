#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import time
import traceback
import xgboost as xgb
from tqdm import tqdm

# 导入matplotlib配置
from models.plt_config import set_plt_configs

class XGBoostModel:
    """XGBoost模型"""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, colsample_bylevel=None, colsample_bynode=None,
                 reg_alpha=0, reg_lambda=1, gamma=0, min_child_weight=1, random_state=42,
                 scale_pos_weight=1, tree_method='auto'):
        """
        初始化XGBoost模型
        
        参数:
        n_estimators (int): 决策树的数量
        max_depth (int): 树的最大深度
        learning_rate (float): 学习率
        subsample (float): 样本采样比例
        colsample_bytree (float): 每棵树的特征采样比例
        colsample_bylevel (float): 每层的特征采样比例
        colsample_bynode (float): 每个节点的特征采样比例
        reg_alpha (float): L1正则化项
        reg_lambda (float): L2正则化项
        gamma (float): 分裂所需的最小损失减少量
        min_child_weight (float): 子节点所需的最小样本权重和
        random_state (int): 随机种子
        scale_pos_weight (float): 正样本的权重倍数
        tree_method (str): 树构建算法
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'objective': 'multi:softprob',
            'random_state': random_state,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': tree_method,
            'verbosity': 0  # 0表示静默模式，不打印训练信息
        }
        
        # 添加可选参数
        if colsample_bylevel is not None:
            self.params['colsample_bylevel'] = colsample_bylevel
        if colsample_bynode is not None:
            self.params['colsample_bynode'] = colsample_bynode
        
        try:
            self.model = XGBClassifier(**self.params)
            self.early_stopping_rounds = None
            self.class_names = None
            self.is_trained = False
            print("成功初始化XGBoostModel")
        except Exception as e:
            print(f"初始化XGBoostModel时出错: {e}")
            raise
    
    def train(self, X_train, y_train):
        """
        训练XGBoost模型
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        
        返回:
        self: 训练好的模型实例
        """
        print("开始训练XGBoost模型...")
        start_time = time.time()
        
        try:
            with tqdm(total=100, desc="训练XGBoost") as pbar:
                # 训练模型
                self.model.fit(X_train, y_train)
                pbar.update(100)
            
            training_time = time.time() - start_time
            print(f"XGBoost模型训练完成，耗时 {training_time:.2f} 秒")
            
            # 记录特征重要性
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
                
                # 打印最重要的10个特征
                if X_train.shape[1] > 10:
                    print("\n最重要的10个特征:")
                    indices = np.argsort(self.feature_importances_)[::-1]
                    for i in range(min(10, len(indices))):
                        print(f"特征 {indices[i]}: {self.feature_importances_[indices[i]]:.6f}")
            
            # 标记模型已训练
            self.is_trained = True
            return self
            
        except Exception as e:
            print(f"训练XGBoost模型时出错: {e}")
            self.is_trained = False
            raise

    # 其他基本方法...
    
    def predict(self, X):
        """预测类别"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测类别概率"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            print(f"预测概率时出错: {e}")
            # 如果出现问题，尝试使用原始模型的predict_proba方法
            if hasattr(self.model, '_Booster'):
                # 使用原始XGBoost模型预测
                dtest = xgb.DMatrix(X)
                raw_preds = self.model._Booster.predict(dtest)
                
                # 处理输出形状
                if len(raw_preds.shape) == 1:
                    # 二分类问题
                    probs = np.zeros((len(X), 2))
                    probs[:, 1] = raw_preds
                    probs[:, 0] = 1.0 - raw_preds
                    return probs
                else:
                    # 多分类问题
                    return raw_preds
            
            # 如果都失败了，返回均匀分布
            print("无法预测概率，返回均匀分布")
            n_classes = self.n_classes_ if self.n_classes_ is not None else 2
            return np.ones((X.shape[0], n_classes)) / n_classes
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
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
        
        print("XGBoost模型评估结果:")
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
        joblib.dump(self.model, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        self.model = joblib.load(filepath)
        print(f"模型已从 {filepath} 加载")
        self.is_trained = True
    
    def plot_confusion_matrix(self, X_test, y_test):
        """绘制混淆矩阵"""
        set_plt_configs()  # 应用中文字体配置
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names if self.class_names else range(cm.shape[1]), 
                   yticklabels=self.class_names if self.class_names else range(cm.shape[0]))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('XGBoost模型混淆矩阵')
        
        return plt
    
    def feature_importance(self):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        return importance, indices
    
    def plot_feature_importance(self, top_n=20):
        """可视化特征重要性"""
        if not hasattr(self.model, 'feature_importances_'):
            print("模型没有feature_importances_属性")
            return
        
        set_plt_configs()  # 设置matplotlib全局配置
        
        # 获取特征重要性
        importance = self.model.feature_importances_
        
        # 显示前top_n个重要特征
        indices = np.argsort(importance)[-top_n:]
        plt.figure(figsize=(10, 6))
        plt.title('XGBoost模型特征重要性')
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [f'特征 {i}' for i in indices])
        plt.xlabel('重要性')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        print("注意: 当前XGBoost版本不支持训练历史记录")
        return None
    
    def plot_training_process(self, X_train, y_train, X_val, y_val, top_n=20):
        """
        完整的训练和可视化过程
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        X_val (np.ndarray): 验证特征
        y_val (np.ndarray): 验证标签
        top_n (int): 显示的特征数量
        """
        set_plt_configs()  # 应用中文字体配置
        
        # 创建评估集
        eval_set = [(X_val, y_val)]
        
        # 训练模型
        self.train(X_train, y_train, eval_set=eval_set)
        
        # 可视化训练历史
        training_history = self.plot_training_history()
        if training_history:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.evals_result_['validation_0']['mlogloss'], label='验证集')
            plt.xlabel('迭代次数')
            plt.ylabel('对数损失')
            plt.title('XGBoost训练历史')
            plt.legend()
            plt.grid(True)
            
            # 可视化特征重要性
            plt.subplot(1, 2, 2)
            importance, indices = self.feature_importance()
            n_features = min(top_n, len(importance))
            top_indices = indices[:n_features]
            plt.bar(range(n_features), importance[top_indices], align='center')
            plt.title('XGBoost特征重要性')
            plt.xticks(range(n_features), [f"特征 {i}" for i in top_indices], rotation=90)
            plt.tight_layout()
        
        return plt 