#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import time
import lightgbm as lgb
from tqdm import tqdm

# 导入matplotlib配置
from models.plt_config import set_plt_configs

class LightGBMModel:
    """LightGBM模型"""
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31,
                 min_child_samples=30, min_split_gain=0.05, reg_alpha=0.3, reg_lambda=0.5,
                 feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=5, cat_smooth=10,
                 path_smooth=None, verbose=-1, random_state=42, deterministic=False,
                 device='cpu', gpu_platform_id=0, gpu_device_id=0, gpu_use_dp=True):
        """
        初始化LightGBM模型
        
        参数:
        n_estimators (int): 决策树的数量
        max_depth (int): 树的最大深度
        learning_rate (float): 学习率
        num_leaves (int): 叶子节点数量
        min_child_samples (int): 一个叶子节点上最少的样本数
        min_split_gain (float): 分裂所需的最小增益
        reg_alpha (float): L1正则化系数
        reg_lambda (float): L2正则化系数
        feature_fraction (float): 特征采样率
        bagging_fraction (float): 样本采样率
        bagging_freq (int): bagging频率
        cat_smooth (float): 类别特征平滑系数
        path_smooth (float): 树路径平滑系数
        verbose (int): 详细程度，负值表示静默
        random_state (int): 随机种子
        deterministic (bool): 是否确保结果可复现
        device (str): 使用的设备，'cpu'或'gpu'
        gpu_platform_id (int): GPU平台ID
        gpu_device_id (int): GPU设备ID
        gpu_use_dp (bool): 是否使用双精度浮点数
        """
        # 创建参数字典
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'min_split_gain': min_split_gain,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'random_state': random_state,
            'verbose': verbose,
            'boosting_type': 'gbdt',  # 默认使用标准GBDT
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'deterministic': deterministic
        }
        
        # 添加可选参数
        if cat_smooth is not None:
            self.params['cat_smooth'] = cat_smooth
        if path_smooth is not None:
            self.params['path_smooth'] = path_smooth
        
        # 配置GPU加速
        self.device = device.lower()
        if self.device == 'gpu':
            try:
                # 尝试导入GPU版本的LightGBM
                import lightgbm as lgb
                # 检查是否支持GPU
                if lgb.gpb_devices_info() != "No GPU available.":
                    print("LightGBM检测到可用GPU，启用GPU加速...")
                    
                    # 设置GPU参数
                    self.params['device'] = 'gpu'
                    self.params['gpu_platform_id'] = gpu_platform_id
                    self.params['gpu_device_id'] = gpu_device_id
                    self.params['gpu_use_dp'] = gpu_use_dp
                    self.params['tree_learner'] = 'gpu'  # 使用GPU学习器
                    self.params['force_col_wise'] = True  # 强制列优先，对GPU更友好
                    self.params['max_bin'] = 63   # GPU模式下降低max_bin提升速度
                    self.params['verbosity'] = -1 # 减少输出
                else:
                    print("指定使用GPU但未检测到可用GPU设备，将回退到CPU模式")
                    self.device = 'cpu'
            except Exception as e:
                print(f"启用GPU加速时出错: {e}")
                print("回退到CPU模式")
                self.device = 'cpu'
            
        try:
            self.model = LGBMClassifier(**self.params)
            self.class_names = None
            self.early_stopping_rounds = None
            self.is_trained = False
            print(f"成功初始化LightGBM模型，运行设备: {self.device}")
        except Exception as e:
            print(f"初始化LightGBM模型时出错: {e}")
            raise

    def train(self, X_train, y_train, eval_set=None, early_stopping_rounds=None):
        """
        训练LightGBM模型

        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        eval_set (list): 可选的评估集，格式为 [(X_val, y_val)]
        early_stopping_rounds (int): 早停轮数，如果为None则不使用早停

        返回:
        self: 训练好的模型实例
        """
        print("开始训练LightGBM模型...")
        start_time = time.time()

        try:
            with tqdm(total=100, desc=f"训练LightGBM({self.device})") as pbar:
                # 训练模型，支持eval_set和早停
                fit_params = {}
                if eval_set:
                    fit_params['eval_set'] = eval_set
                    if early_stopping_rounds:
                        fit_params['early_stopping_rounds'] = early_stopping_rounds
                        fit_params['callbacks'] = [lgb.callback.early_stopping(early_stopping_rounds)]
                
                # 训练模型
                self.model.fit(X_train, y_train, **fit_params)
                pbar.update(100)

            training_time = time.time() - start_time
            print(f"LightGBM模型训练完成，耗时 {training_time:.2f} 秒")

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
            print(f"训练LightGBM模型时出错: {e}")
            self.is_trained = False
            raise
    
    def predict(self, X):
        """预测类别"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练或训练失败，请先调用train方法")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测类别概率"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练或训练失败，请先调用train方法")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        # 添加模型状态检查
        if not self.is_trained:
            raise RuntimeError("模型未经训练或训练失败，请先调用train方法")
                
        # 预测和评估
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
        
        print("LightGBM模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        if not hasattr(self.model, 'feature_importances_'):
            print("模型没有feature_importances_属性，无法绘制特征重要性")
            return
        
        # 设置中文字体和负号显示
        set_plt_configs()
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [f"特征 {i}" for i in indices])
        plt.xlabel('特征重要性')
        plt.title('LightGBM特征重要性 (Top {})'.format(top_n))
        plt.tight_layout()
    
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
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")
    
    def plot_confusion_matrix(self, X_test, y_test):
        """绘制混淆矩阵"""
        if not self.is_trained:
            raise RuntimeError("模型未经训练，请先调用train方法")
            
        # 设置中文字体和负号显示
        set_plt_configs()
        
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
        plt.title('LightGBM混淆矩阵')
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
        print("LightGBM分类报告:")
        print(report)


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
    
    print("提取验证集特征...")
    X_val, y_val = feature_extractor.extract_features(val_loader)
    
    print("提取测试集特征...")
    X_test, y_test = feature_extractor.extract_features(test_loader)
    
    # 创建并训练LightGBM模型
    lgbm_model = LightGBMModel()
    print("训练LightGBM模型...")
    lgbm_model.train(X_train, y_train)
    
    # 设置类别名称
    lgbm_model.set_class_names(processor.get_class_names())
    
    # 评估模型
    metrics = lgbm_model.evaluate(X_test, y_test)
    
    # 绘制混淆矩阵
    lgbm_model.plot_confusion_matrix(X_test, y_test)
    plt.savefig('lgbm_confusion_matrix.png')
    
    # 打印分类报告
    lgbm_model.print_classification_report(X_test, y_test)
    
    # 绘制特征重要性
    lgbm_model.plot_feature_importance()
    plt.savefig('lgbm_feature_importance.png')
    
    # 保存模型
    lgbm_model.save_model('models/lgbm_model.joblib') 