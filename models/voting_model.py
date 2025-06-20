#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
from collections import Counter


class VotingModel:
    """使用多个模型投票的模型集成类"""
    def __init__(self, models=None, voting='soft'):
        """初始化"""
        self.models = models if models else []
        self.voting = voting
        self.class_names = None  # 类别名称
    
    def train(self, X, y):
        """训练模型"""
        # 投票模型不需要额外训练
        pass
    
    def predict(self, X):
        """预测类别"""
        if self.voting == 'soft':
            # 软投票，使用概率平均
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # 硬投票，使用多数表决
            predictions = []
            for name, model in self.models:
                try:
                    y_pred = model.predict(X)
                    predictions.append(y_pred)
                except Exception as e:
                    print(f"模型 {name} 预测失败: {e}")
            
            # 转换为numpy数组并进行多数表决
            predictions = np.array(predictions)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, minlength=len(self.class_names))), 
                axis=0, 
                arr=predictions
            )
            return maj
    
    def predict_proba(self, X):
        """预测概率"""
        # 收集每个模型的概率预测
        probas = []
        for name, model in self.models:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    probas.append(y_proba)
            except Exception as e:
                print(f"模型 {name} 预测概率失败: {e}")
        
        # 如果没有概率预测，返回None
        if not probas:
            return None
        
        # 计算平均概率
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
    
    def get_params(self, deep=True):
        """获取参数（兼容sklean API）"""
        return {"models": self.models, "voting": self.voting}
    
    def set_params(self, **parameters):
        """设置参数（兼容sklean API）"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def set_class_names(self, class_names):
        """设置类别名称"""
        self.class_names = class_names
    
    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        
        # 计算评估指标
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        print(f"{self.__class__.__name__}评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


class MlpEnhancedVotingModel(VotingModel):
    """结合MLP和其他模型的增强型投票模型，可完全序列化"""
    
    def __init__(self, base_voting_model, mlp_model, voting='soft'):
        """
        初始化MLP增强型投票模型
        
        参数:
        base_voting_model: 基础投票模型
        mlp_model: MLP模型
        voting: 投票类型，'soft'或'hard'
        """
        super().__init__(base_voting_model.models, voting)
        self.base_voting_model = base_voting_model
        self.mlp_model = mlp_model
        self.class_names = base_voting_model.class_names
    
    def predict(self, X):
        """结合MLP和投票模型进行预测"""
        if self.voting == 'hard':
            # 硬投票：多数表决
            vote_pred = self.base_voting_model.predict(X)
            mlp_pred = self.mlp_model.predict(X)
            
            from collections import Counter
            final_pred = []
            for i in range(len(X)):
                # 每个样本的所有模型预测
                all_preds = [vote_pred[i], mlp_pred[i]]
                # 取最常见的预测结果
                counter = Counter(all_preds)
                final_pred.append(counter.most_common(1)[0][0])
            return np.array(final_pred)
        else:
            # 软投票：通过predict_proba进行
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """结合MLP和投票模型预测概率"""
        # 获取投票模型的概率预测
        vote_proba = self.base_voting_model.predict_proba(X)
        # 获取MLP的概率预测
        mlp_proba = self.mlp_model.predict_proba(X)
        
        # 合并概率（简单平均）
        combined_proba = (vote_proba + mlp_proba) / 2
        return combined_proba


class DESASModel:
    """动态集成选择模型（DES-AS算法）"""
    def __init__(self, base_models, k=7, n_perm=400, random_state=42):
        """
        初始化动态集成选择模型
        
        参数:
        base_models (list): 基础模型列表，每个元素是一个元组 (name, model)
        k (int): 邻居数量
        n_perm (int): 排列数量
        random_state (int): 随机种子
        """
        self.base_models = base_models
        self.k = k
        self.n_perm = n_perm
        self.random_state = random_state
        self.class_names = None
    
    def train(self, X_train, y_train):
        """
        训练所有基础模型
        
        参数:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练标签
        """
        # 存储训练数据
        self.X_dsel = X_train
        self.y_dsel = y_train
        
        # 训练每个基础模型
        for name, model in self.base_models:
            print(f"训练 {name} 模型...")
            model.fit(X_train, y_train)
    
    def majority_vote(self, pred_matrix):
        """
        多数表决
        
        参数:
        pred_matrix (np.ndarray): 预测矩阵，形状为 (n_models, n_samples)
        
        返回:
        votes (np.ndarray): 多数表决结果
        """
        # 转置以便按样本处理
        votes = []
        for col in pred_matrix.T:
            counter = Counter(col)
            most_common = counter.most_common(1)[0][0]
            votes.append(most_common)
        return np.array(votes)
    
    def utility(self, sub_idx, preds, y_true):
        """
        计算子集的准确率
        
        参数:
        sub_idx (list): 子集索引
        preds (np.ndarray): 预测矩阵
        y_true (np.ndarray): 真实标签
        
        返回:
        accuracy (float): 准确率
        """
        if len(sub_idx) == 0:
            return 0.0
        votes = self.majority_vote(preds[sub_idx])
        return (votes == y_true).mean()
    
    def shapley_mc(self, preds, y_true, n_perm=None, r=100, d=0.02):
        """
        蒙特卡洛方法计算Shapley值
        
        参数:
        preds (np.ndarray): 预测矩阵
        y_true (np.ndarray): 真实标签
        n_perm (int): 排列数量
        r (int): 检查收敛的频率
        d (float): 收敛阈值
        
        返回:
        phi (np.ndarray): Shapley值
        """
        if n_perm is None:
            n_perm = self.n_perm
            
        rng = np.random.RandomState(self.random_state)
        m = preds.shape[0]
        phi = np.zeros(m)
        cache = []
        
        for t in range(1, n_perm + 1):
            order = rng.permutation(m)
            cur_set = []
            v_prev = 0.0
            for clf_idx in order:
                v_new = self.utility(cur_set + [clf_idx], preds, y_true)
                phi[clf_idx] += (v_new - v_prev)
                cur_set.append(clf_idx)
                v_prev = v_new
            
            # 收敛检测
            if t % r == 0:
                cache.append(phi / t)
                if len(cache) >= 2:
                    diff = np.abs(cache[-1] - cache[-2]).mean() / (np.abs(cache[-2]).mean() + 1e-12)
                    if diff < d:
                        break
        
        return phi / t
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        X (np.ndarray): 特征
        
        返回:
        y_pred (np.ndarray): 预测标签
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 预测结果
        y_pred = np.zeros(len(X), dtype=int)
        
        # 对每个样本单独预测
        for i, x_query in enumerate(X):
            # 搜索最近邻
            nn = NearestNeighbors(n_neighbors=self.k).fit(self.X_dsel)
            roc_idx = nn.kneighbors(x_query.reshape(1, -1), return_distance=False).ravel()
            Xr, yr = self.X_dsel[roc_idx], self.y_dsel[roc_idx]
            
            # 获取每个模型在RoC上的预测
            m = len(self.base_models)
            preds_roc = np.empty((m, self.k), dtype=int)
            preds_q = np.empty(m, dtype=int)
            
            for j, (name, model) in enumerate(self.base_models):
                preds_roc[j] = model.predict(Xr)
                preds_q[j] = model.predict(x_query.reshape(1, -1))[0]
            
            # 计算Shapley值
            phi = self.shapley_mc(preds_roc, yr)
            
            # 正值归一化
            pos = phi > 0
            if not pos.any():  # 如果全 <= 0，则退化为简单多数
                y_pred[i] = Counter(preds_q).most_common(1)[0][0]
                continue
            
            weights = phi[pos] / phi[pos].sum()
            preds_pos = preds_q[pos]
            
            # 加权投票
            score = {}
            for w, cls in zip(weights, preds_pos):
                score[cls] = score.get(cls, 0) + w
            
            y_pred[i] = max(score.items(), key=lambda kv: kv[1])[0]
        
        return y_pred
    
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
        
        print("DES-AS模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return metrics
    
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
        plt.title('DES-AS模型混淆矩阵')
        
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
    
    def compare_base_models(self, X_test, y_test):
        """
        比较基础模型和DES-AS模型的性能
        
        参数:
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试标签
        """
        # 收集每个基础模型的性能
        model_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # 评估每个基础模型
        for name, model in self.base_models:
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            model_names.append(name)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            print(f"{name} 模型评估结果:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            print()
        
        # 评估DES-AS模型
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_names.append('DES-AS')
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print("DES-AS模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 绘制比较图
        metrics = {
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric_name, metric_values) in enumerate(metrics.items()):
            ax = axes[i]
            ax.bar(model_names, metric_values)
            ax.set_title(metric_name)
            ax.set_ylim(0, 1)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            # 在每个柱状图上添加数值
            for j, v in enumerate(metric_values):
                ax.text(j, v + 0.01, f'{v:.4f}', ha='center')
        
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
    from models.logistic_regression_model import LogisticRegressionModel
    from models.svm_model import SVMModel
    from models.random_forest_model import RandomForestModel
    
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
    
    # 创建并训练基础模型
    # 逻辑回归
    lr_model = LogisticRegressionModel()
    print("训练逻辑回归模型...")
    lr_model.train(X_train, y_train)
    
    # SVM
    svm_model = SVMModel()
    print("训练SVM模型...")
    svm_model.train(X_train, y_train)
    
    # 随机森林
    rf_model = RandomForestModel()
    print("训练随机森林模型...")
    rf_model.train(X_train, y_train)
    
    # 创建基础模型列表
    base_models = [
        ('LR', lr_model.model),
        ('SVM', svm_model.model),
        ('RF', rf_model.model)
    ]
    
    # 创建并训练硬投票模型
    hard_voting = VotingModel(base_models, voting='hard')
    print("训练硬投票模型...")
    hard_voting.train(X_train, y_train)
    
    # 设置类别名称
    hard_voting.set_class_names(processor.get_class_names())
    
    # 评估硬投票模型
    metrics_hard = hard_voting.evaluate(X_test, y_test)
    
    # 创建并训练软投票模型
    soft_voting = VotingModel(base_models, voting='soft')
    print("训练软投票模型...")
    soft_voting.train(X_train, y_train)
    
    # 设置类别名称
    soft_voting.set_class_names(processor.get_class_names())
    
    # 评估软投票模型
    metrics_soft = soft_voting.evaluate(X_test, y_test)
    
    # 比较基础模型和投票模型
    comparison_plot = soft_voting.compare_base_models(X_test, y_test)
    comparison_plot.savefig('voting_model_comparison.png')
    
    # 绘制混淆矩阵
    hard_voting.plot_confusion_matrix(X_test, y_test)
    plt.savefig('hard_voting_confusion_matrix.png')
    
    soft_voting.plot_confusion_matrix(X_test, y_test)
    plt.savefig('soft_voting_confusion_matrix.png')
    
    # 创建并训练DES-AS模型
    des_as = DESASModel(base_models)
    print("训练DES-AS模型...")
    des_as.train(X_train, y_train)
    
    # 设置类别名称
    des_as.set_class_names(processor.get_class_names())
    
    # 评估DES-AS模型
    metrics_des_as = des_as.evaluate(X_test, y_test)
    
    # 比较基础模型和DES-AS模型
    comparison_plot_des_as = des_as.compare_base_models(X_test, y_test)
    comparison_plot_des_as.savefig('des_as_model_comparison.png')
    
    # 绘制混淆矩阵
    des_as.plot_confusion_matrix(X_test, y_test)
    plt.savefig('des_as_confusion_matrix.png') 