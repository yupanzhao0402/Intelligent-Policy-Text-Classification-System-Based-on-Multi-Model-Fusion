#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
# 设置环境变量以抑制joblib的物理核心警告
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
import time
from tqdm import tqdm
import traceback
from torch.utils.data import Dataset, DataLoader

# 模块导入
from models.voting_model import VotingModel
from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
from models.logistic_regression_model import LogisticRegressionModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.mlp_model import MLPModel

# 添加预训练Transformer模型
try:
    from models.pretrained_transformer import PretrainedTransformerModel
    TRANSFORMER_MODEL_AVAILABLE = True
except ImportError:
    print("注意: 预训练Transformer模型不可用，请安装相关依赖")
    TRANSFORMER_MODEL_AVAILABLE = False

# 检查是否有可用的注意力模型
try:
    from models.attention_voting_model import AttentionVotingModel
    ATTENTION_MODEL_AVAILABLE = True
except ImportError:
    print("注意: 注意力投票模型不可用，请安装相关依赖")
    ATTENTION_MODEL_AVAILABLE = False

# 定义计算设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    # 显示CUDA内存信息
    print(f"CUDA可用内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"当前已用内存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
else:
    print("使用CPU")

print(f"PyTorch版本: {torch.__version__}")

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
    print("PyTorch随机种子设置成功")

# 检查并配置matplotlib中文支持
def setup_matplotlib_chinese():
    """配置matplotlib中文支持"""
    try:
        # 尝试使用中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        # 验证字体
        for font in plt.rcParams['font.sans-serif']:
            try:
                from matplotlib.font_manager import FontProperties
                FontProperties(font=font)
                print(f"成功设置Windows中文字体: {font}")
                break
            except:
                continue
                
        # 设置其他样式
        plt.style.use('seaborn-v0_8-whitegrid')
        print("matplotlib中文字体和风格配置已加载")
    except Exception as e:
        print(f"matplotlib中文配置加载失败: {e}")

# 通用模型检查和加载函数
def check_and_load_model(model_name, model_path=None):
    """
    检查指定路径是否存在已训练的模型，如果存在则加载
    
    参数:
    model_name (str): 模型名称
    model_path (str): 模型文件路径（如果为None，则使用默认路径格式）
    
    返回:
    model: 加载的模型（如果存在）
    metrics: 模型指标（None）
    train_time: 训练时间（0，因为跳过了训练）
    exists (bool): 是否成功加载已存在的模型
    """
    if model_path is None:
        model_path = f"./output_models/{model_name}.pkl"
    
    if os.path.exists(model_path):
        print(f"发现已训练的{model_name}模型，正在加载...")
        try:
            model = joblib.load(model_path)
            print(f"模型{model_name}加载成功，跳过训练")
            # 返回模型、None指标和0训练时间
            return model, None, 0
        except Exception as e:
            print(f"模型{model_name}加载失败: {e}，将继续训练新模型")
            return None, None, 0
    return None, None, 0

def extract_and_save_features():
    """提取特征并保存到文件"""
    # 创建数据处理器 - 使用示例数据
    data_filepath = "test/sample_data.csv"
    processor = DataProcessor(data_filepath, sample_size=None)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = processor.get_dataloaders(batch_size=32)
    
    # 获取类别名称
    class_names = processor.get_class_names()
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model_name='bert-base-chinese', device=device)
    
    # 定义特征提取参数
    pooling_strategy = 'cls'  # 可选: 'cls', 'mean', 'max'
    layers = None  # 使用最后一层，或者指定多层如 [-1, -2, -3, -4]
    
    # 创建保存目录
    features_dir = "./extracted_features"
    os.makedirs(features_dir, exist_ok=True)
    
    # 提取训练集特征
    print("从训练集提取特征...")
    X_train, y_train = feature_extractor.extract_features(train_loader, pooling_strategy=pooling_strategy, layers=layers)
    
    # 提取验证集特征
    print("从验证集提取特征...")
    X_val, y_val = feature_extractor.extract_features(val_loader, pooling_strategy=pooling_strategy, layers=layers)
    
    # 提取测试集特征
    print("从测试集提取特征...")
    X_test, y_test = feature_extractor.extract_features(test_loader, pooling_strategy=pooling_strategy, layers=layers)
    
    # 打印特征形状
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"验证集特征形状: {X_val.shape}, 标签形状: {y_val.shape}")
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
    # 保存特征和类别名称
    np.save(os.path.join(features_dir, "X_train.npy"), X_train)
    np.save(os.path.join(features_dir, "y_train.npy"), y_train)
    np.save(os.path.join(features_dir, "X_val.npy"), X_val)
    np.save(os.path.join(features_dir, "y_val.npy"), y_val)
    np.save(os.path.join(features_dir, "X_test.npy"), X_test)
    np.save(os.path.join(features_dir, "y_test.npy"), y_test)
    
    # 保存类别名称
    joblib.dump(class_names, os.path.join(features_dir, "class_names.joblib"))
    
    print(f"特征已保存到 {features_dir} 目录")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names

def load_features():
    """加载已保存的特征"""
    features_dir = "./extracted_features"
    
    # 检查目录是否存在
    if not os.path.exists(features_dir):
        print(f"特征目录 {features_dir} 不存在，请先提取特征")
        return None, None, None, None, None, None, None
    
    # 加载特征
    X_train = np.load(os.path.join(features_dir, "X_train.npy"))
    y_train = np.load(os.path.join(features_dir, "y_train.npy"))
    X_val = np.load(os.path.join(features_dir, "X_val.npy"))
    y_val = np.load(os.path.join(features_dir, "y_val.npy"))
    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))
    
    # 加载类别名称
    class_names = joblib.load(os.path.join(features_dir, "class_names.joblib"))
    
    print("特征加载成功")
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"验证集特征形状: {X_val.shape}, 标签形状: {y_val.shape}")
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names

# 拆分模型评估函数
def evaluate_model_with_cv(model_class, X, y, class_names, cv=5, params=None, model_name="模型"):
    """使用交叉验证评估模型"""
    print(f"\n使用{cv}折交叉验证评估{model_name}...")
    
    # 初始化结果存储
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    # 准备交叉验证
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 开始交叉验证
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"处理第{fold+1}/{cv}折...")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 创建并训练模型
        if model_name in ["MLP"]:  # 神经网络模型需要验证集
            model = model_class(**params)
            model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        else:  # 传统机器学习模型
            model = model_class(**params)
            model.train(X_train_fold, y_train_fold)
        
        model.set_class_names(class_names)
        metrics = model.evaluate(X_val_fold, y_val_fold)
        
        # 记录结果
        all_accuracies.append(metrics['accuracy'])
        all_precisions.append(metrics['precision'])
        all_recalls.append(metrics['recall'])
        all_f1s.append(metrics['f1'])
    
    # 计算平均指标
    mean_metrics = {
        'accuracy': np.mean(all_accuracies),
        'precision': np.mean(all_precisions),
        'recall': np.mean(all_recalls),
        'f1': np.mean(all_f1s),
        'accuracy_std': np.std(all_accuracies),
        'precision_std': np.std(all_precisions),
        'recall_std': np.std(all_recalls),
        'f1_std': np.std(all_f1s),
    }
    
    print(f"{model_name} 交叉验证结果:")
    print(f"  准确率: {mean_metrics['accuracy']:.4f} ± {mean_metrics['accuracy_std']:.4f}")
    print(f"  精确率: {mean_metrics['precision']:.4f} ± {mean_metrics['precision_std']:.4f}")
    print(f"  召回率: {mean_metrics['recall']:.4f} ± {mean_metrics['recall_std']:.4f}")
    print(f"  F1分数: {mean_metrics['f1']:.4f} ± {mean_metrics['f1_std']:.4f}")
    
    return mean_metrics

# 分别训练各个模型的函数
def train_logistic_regression(X_train, y_train, X_val, y_val, class_names, use_cv=True, optimize=False):
    """训练逻辑回归模型"""
    print("\n训练逻辑回归模型...")
    
    # 检查是否存在已训练的模型
    model_name = "logistic_regression"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        # 在验证集上评估已加载的模型
        metrics = model.evaluate(X_val, y_val)
        print(f"验证集指标: {metrics}")
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 默认参数 - 已知在该任务中表现良好的参数
    lr_params = {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'liblinear',  # 或者选择 'lbfgs', 'newton-cg', 'sag', 'saga'
        'penalty': 'l2',
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    if optimize:
        print("正在对逻辑回归进行高级优化...")
        
        # 使用优化前的有效参数，恢复到更好的性能配置
        best_params = {
            'C': 1.0,                # 原始C值
            'max_iter': 1000,        # 足够的迭代次数
            'solver': 'liblinear',   # 对高维数据有效的求解器
            'penalty': 'l2',         # 使用L2正则化提高泛化能力
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1             # 使用所有CPU核心加速训练
        }
        print(f"使用高维数据优化参数: {best_params}")
        
        # 创建模型对象，使用L1和L2混合正则化避免过拟合
        model = LogisticRegressionModel(**best_params)
        
        # 无需特征选择，而是利用ElasticNet的内置特征选择
        model.train(X_train, y_train, use_scaling=True, use_feature_selection=False)
        model.set_class_names(class_names)
        
        # 评估模型性能并记录指标
        metrics = model.evaluate(X_val, y_val)
        
        # 保存优化结果
        os.makedirs('./models/optimized', exist_ok=True)
        model.save_model('./models/optimized/logistic_regression_optimized.pkl')
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
    else:
        if use_cv:
            # 使用交叉验证
            metrics = evaluate_model_with_cv(
                LogisticRegressionModel, X_train, y_train, class_names, 
                cv=5, params=lr_params, model_name="逻辑回归"
            )
            # 创建完整模型
            model = LogisticRegressionModel(**lr_params)
            model.train(X_train, y_train, use_scaling=True)
            model.set_class_names(class_names)
        else:
            # 单次训练
            model = LogisticRegressionModel(**lr_params)
            model.train(X_train, y_train, use_scaling=True)
            model.set_class_names(class_names)
            metrics = None
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
    
    # 记录训练时间
    train_time = time.time() - start_time
    minutes = int(train_time / 60)
    seconds = int(train_time % 60)
    print(f"逻辑回归模型训练完成，耗时: {minutes}分{seconds}秒")
    
    return model, metrics, train_time

def train_svm(X_train, y_train, X_val, y_val, class_names, use_cv=True, optimize=False):
    """训练SVM模型"""
    print("\n训练SVM模型...")
    
    # 检查是否存在已训练的模型
    model_name = "svm"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        # 在验证集上评估已加载的模型
        metrics = model.evaluate(X_val, y_val)
        print(f"验证集指标: {metrics}")
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 默认参数 - 使用已知效果好的参数
    svm_params = {
        'C': 1,                 # 增加模型适应性
        'kernel': 'linear',
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42,
        'cache_size': 2000,
        'max_iter': 15000
    }
    
    if optimize:
        print("正在对SVM进行高级优化...")
        # 优化后的SVM参数，专门针对高维数据调整
        best_params = {
            'C': 1.5,                 # 略微增加正则化强度，但不要过大
            'kernel': 'linear',       # 使用与原始版本相同的线性核
            'gamma': 'scale',         # 使用与原始版本相同的gamma计算方式
            'probability': True,      # 需要概率输出
            'class_weight': 'balanced', # 处理类别不平衡
            'cache_size': 2000,       # 保持较大缓存提高速度
            'max_iter': 20000,        # 增加迭代次数确保收敛
            'random_state': 42,
            'tol': 1e-5               # 提高收敛精度
        }
        print(f"使用针对高维数据优化的SVM参数: {best_params}")
        
        # 创建模型并训练，完全不使用PCA或特征选择
        model = SVMModel(**best_params)
        model.train(X_train, y_train, use_scaling=True, use_feature_selection=False, use_pca=False)
        model.set_class_names(class_names)
        
        # 评估模型性能并记录指标
        metrics = model.evaluate(X_val, y_val)
        
        # 保存优化结果
        os.makedirs('./models/optimized', exist_ok=True)
        model.save_model('./models/optimized/svm_optimized.pkl')
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
    else:
        if use_cv:
            # 使用交叉验证
            metrics = evaluate_model_with_cv(
                SVMModel, X_train, y_train, class_names,
                cv=5, params=svm_params, model_name="SVM"
            )
            # 创建完整模型
            model = SVMModel(**svm_params)
            # 标准版本也使用特征缩放，这对SVM至关重要，但不做特征选择
            model.train(X_train, y_train, use_scaling=True, use_feature_selection=False)
            model.set_class_names(class_names)
        else:
            # 单次训练
            model = SVMModel(**svm_params)
            # 标准版本也使用特征缩放，这对SVM至关重要，但不做特征选择
            model.train(X_train, y_train, use_scaling=True, use_feature_selection=False)
            model.set_class_names(class_names)
            metrics = None
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
    
    # 记录训练时间
    train_time = time.time() - start_time
    minutes = int(train_time / 60)
    seconds = int(train_time % 60)
    print(f"SVM模型训练完成，耗时: {minutes}分{seconds}秒")
    
    return model, metrics, train_time

def train_random_forest(X_train, y_train, class_names, use_cv=True, optimize=False):
    """训练随机森林模型"""
    print("\n训练随机森林模型...")
    
    # 检查是否存在已训练的模型
    model_name = "random_forest"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 基础参数 - 重新调整以提高性能
    rf_params = {
        'n_estimators': 2000,        # 大幅增加树的数量，提高模型复杂度和泛化能力
        'max_depth': 20,             # 优化树深度，避免过拟合
        'min_samples_split': 5,      # 增加分裂阈值，减少过拟合风险
        'min_samples_leaf': 2,       # 增加叶节点样本数，提高鲁棒性
        'max_features': 'sqrt',      # 使用特征平方根数量，增加随机性
        'bootstrap': True,
        'class_weight': 'balanced',  # 平衡类别权重
        'random_state': 42,
        'n_jobs': -1,
        'max_samples': 0.8,          # 提高样本多样性
        'criterion': 'gini'          # 使用基尼系数，对噪声更为稳健
        # 注意: RandomForestModel类的__init__方法内部已经设置了oob_score
    }
    
    if optimize:
        print("正在对随机森林进行高级优化...")
        # 高维数据的优化参数 - 强调更好的泛化性能
        optimized_rf_params = {
            'n_estimators': 1000,        # 大幅增加树的数量，提高集成效果
            'max_depth': 18,             # 适度树深度，平衡拟合能力和泛化性
            'min_samples_split': 4,      # 适度的分裂阈值
            'min_samples_leaf': 2,       # 确保叶节点有足够样本
            'max_features': 'sqrt',      # 使用特征平方根，增加随机性
            'bootstrap': True,
            'class_weight': 'balanced',  # 平衡类别权重
            'random_state': 42,
            'n_jobs': -1,
            'max_samples': 0.85,         # 较高的样本采样比例
            'criterion': 'gini'          # 使用基尼系数增加稳定性
            # RandomForestModel类已内置oob_score设置
        }
        
        print(f"使用高维数据优化参数: {optimized_rf_params}")
        
        # 创建模型并训练
        model = RandomForestModel(**optimized_rf_params)
        model.train(X_train, y_train)
        model.set_class_names(class_names)
        
        # 评估模型
        metrics = model.evaluate(X_train, y_train)
        print(f"随机森林训练集评估指标: {metrics}")
        
        # 保存优化结果
        os.makedirs('./models/optimized', exist_ok=True)
        model.save_model('./models/optimized/random_forest_optimized.pkl')
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
        
        # 确认模型已训练完成
        print("随机森林优化模型训练完成")
    else:
        if use_cv:
            # 使用交叉验证
            metrics = evaluate_model_with_cv(
                RandomForestModel, X_train, y_train, class_names,
                cv=5, params=rf_params, model_name="随机森林"
            )
            # 创建完整模型
            model = RandomForestModel(**rf_params)
            model.train(X_train, y_train)
            model.set_class_names(class_names)
        else:
            # 单次训练
            model = RandomForestModel(**rf_params)
            model.train(X_train, y_train)
            model.set_class_names(class_names)
            metrics = None
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
    
    # 记录训练时间
    train_time = time.time() - start_time
    minutes = int(train_time / 60)
    seconds = int(train_time % 60)
    print(f"随机森林模型训练完成，耗时: {minutes}分{seconds}秒")
    
    return model, metrics, train_time

def train_xgboost(X_train, y_train, X_val, y_val, class_names, use_cv=True, optimize=False):
    """训练XGBoost模型"""
    print("\n训练XGBoost模型...")
    model_name = "xgboost"
    if optimize:
        model_name += "_optimized"
    
    model_path = f"./output_models/{model_name}.pkl"
    
    # 检查是否存在已训练的模型
    if os.path.exists(model_path):
        print(f"发现已训练的{model_name}模型，正在加载...")
        try:
            model = joblib.load(model_path)
            print("模型加载成功，跳过训练")
            
            # 在验证集上评估
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            
            print(f"{model_name}验证集性能: F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")
            
            return model, None, 0  # 返回0作为训练时间，因为跳过了训练
        except Exception as e:
            print(f"模型加载失败: {e}，继续训练新模型")
    
    # 开始训练计时
    start_time = time.time()
    
    # 基础参数 - 针对高维数据的稳健配置
    xgb_params = {
        'n_estimators': 1200,         # 增加估计器数量
        'max_depth': 4,              # 增加树深度以提高拟合能力
        'learning_rate': 0.04,       # 微调学习率以获得更好的泛化能力
        'subsample': 0.85,           # 增加样本采样率
        'colsample_bytree': 0.85,    # 增加特征采样率
        'colsample_bylevel': 0.75,   # 每层级的特征采样率
        'colsample_bynode': 0.75,    # 每个节点的特征采样率
        'random_state': 42,
        'reg_alpha': 0.01,           # 轻微L1正则化
        'reg_lambda': 0.5,           # L2正则化
        'gamma': 0.04,               # 减少分裂所需的最小增益
        'min_child_weight': 6,       # 调整以减少过拟合
        # GPU加速
        'tree_method': 'gpu_hist' if device == 'cuda' else 'hist'  # 使用GPU加速
    }
    
    # XGBoost模型类中处理GPU参数，这里不直接添加到参数中
    # 避免初始化错误
    if device == 'cuda':
        print("检测到GPU，将在XGBoostModel类内部启用GPU加速")
        print("注意：使用的tree_method为'gpu_hist'，其他GPU相关参数将在模型初始化时设置")
    
    # GPU加速检测
    if device == 'cuda':
        print("XGBoost将使用GPU加速训练")
    
    try:
        if optimize:
            print("正在对XGBoost进行高级优化...")
            # 新的优化参数 - 更接近原始参数但进行微调
            optimized_params = {
                'n_estimators': 1600,        # 增加树的数量以提高模型性能
                'max_depth': 4,              # 增加树深度提高拟合能力
                'learning_rate': 0.035,      # 微调学习率
                'subsample': 0.9,            # 略微增加样本采样率
                'colsample_bytree': 0.9,     # 增加特征采样率
                'colsample_bylevel': 0.8,    # 略微增加
                'colsample_bynode': 0.8,     # 略微增加
                'random_state': 42,
                'reg_alpha': 0.03,           # 轻微L1正则化
                'reg_lambda': 0.45,          # 调整L2正则化
                'gamma': 0.02,               # 降低分裂阈值，允许更多的树生长
                'min_child_weight': 4,       # 减少最小子节点权重
                'tree_method': 'gpu_hist' if device == 'cuda' else 'hist'  # 保持GPU加速
            }
            
            print(f"使用针对高维数据专门优化的XGBoost参数: {optimized_params}")
            
            try:
                # 创建优化模型
                model = XGBoostModel(**optimized_params)
                
                # 训练模型 - 不使用早停参数
                model.train(X_train, y_train)
                model.set_class_names(class_names)
                
                # 验证模型是否成功训练
                if not model.is_trained:
                    raise RuntimeError("模型训练失败，is_trained标志为False")
                
                try:
                    # 尝试做一次预测，验证模型可用性
                    test_pred = model.predict(X_val[:1])
                    print("模型训练成功并能正常预测")
                except Exception as e:
                    print(f"模型预测测试失败: {e}")
                    raise RuntimeError("模型训练后无法正常预测")
                
                # 评估模型性能
                metrics = model.evaluate(X_val, y_val)
                
                # 保存优化结果
                os.makedirs('./models/optimized', exist_ok=True)
                model.save_model('./models/optimized/xgboost_optimized.pkl')
                
                print("XGBoost优化模型训练和评估成功完成")
                # 记录训练时间
                train_time = time.time() - start_time
                minutes = int(train_time / 60)
                seconds = int(train_time % 60)
                print(f"XGBoost模型优化训练完成，耗时: {minutes}分{seconds}秒")
                
                # 保存到output_models目录
                joblib.dump(model, model_path)
                print(f"优化模型已保存到 {model_path}")
                
                return model, metrics, train_time
            except Exception as e:
                print(f"XGBoost优化模型训练失败: {e}")
                # 回退到基础参数但使用优化标记
                print("尝试使用基础参数创建优化标记的模型...")
                model = XGBoostModel(**xgb_params)
                model.train(X_train, y_train)
                model.set_class_names(class_names)
                
                # 记录训练时间
                train_time = time.time() - start_time
                minutes = int(train_time / 60)
                seconds = int(train_time % 60)
                print(f"XGBoost备选模型训练完成，耗时: {minutes}分{seconds}秒")
                
                # 保存到output_models目录
                joblib.dump(model, model_path)
                print(f"备选模型已保存到 {model_path}")
                
                metrics = model.evaluate(X_val, y_val) if X_val is not None else None
                return model, metrics, train_time
        else:
            # 标准模式训练
            if use_cv:
                # 使用交叉验证
                metrics = evaluate_model_with_cv(
                    XGBoostModel, X_train, y_train, class_names,
                    cv=5, params=xgb_params, model_name="XGBoost"
                )
                # 创建完整模型
                model = XGBoostModel(**xgb_params)
                # 简单训练，不使用验证集和早停
                model.train(X_train, y_train)
                model.set_class_names(class_names)
            else:
                # 单次训练
                model = XGBoostModel(**xgb_params)
                model.train(X_train, y_train)
                model.set_class_names(class_names)
                metrics = None
            
            # 记录训练时间
            train_time = time.time() - start_time
            minutes = int(train_time / 60)
            seconds = int(train_time % 60)
            print(f"XGBoost模型训练完成，耗时: {minutes}分{seconds}秒")
            
            # 保存到output_models目录
            joblib.dump(model, model_path)
            print(f"模型已保存到 {model_path}")
            
            return model, metrics, train_time
    
    except Exception as e:
        print(f"XGBoost模型训练过程中发生未处理的异常: {e}")
        print("返回None模型和None指标")
        # 记录训练失败时间
        train_time = time.time() - start_time
        minutes = int(train_time / 60)
        seconds = int(train_time % 60)
        print(f"XGBoost训练失败，耗时: {minutes}分{seconds}秒")
        return None, None, train_time

def train_lightgbm(X_train, y_train, X_val, y_val, class_names, use_cv=True, optimize=False):
    """训练LightGBM模型"""
    print("\n训练LightGBM模型...")
    
    # 检查是否存在已训练的模型
    model_name = "lightgbm"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        # 在验证集上评估已加载的模型
        metrics = model.evaluate(X_val, y_val)
        print(f"验证集指标: {metrics}")
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 基础参数 - 针对高维数据的优化配置
    lgb_params = {
        'n_estimators': 1150,         # 保持估计器数量
        'max_depth': 5,               # 增加树深度以提高模型容量
        'learning_rate': 0.04,        # 调小学习率提高稳定性
        'num_leaves': 15,             # 增加叶子节点数以提高表达能力
        'min_child_samples': 20,      # 降低叶节点最小样本数
        'min_split_gain': 0.02,       # 降低分裂门槛允许更多分裂
        'reg_alpha': 0.02,            # 轻微L1正则化防止过拟合
        'reg_lambda': 0.45,           # 调整L2正则化
        'feature_fraction': 0.25,     # 增加特征采样率
        'bagging_fraction': 0.75,     # 增加样本采样率
        'bagging_freq': 4,            # 更频繁的bagging
        'verbose': -1,                # 减少输出
        'random_state': 42            # 固定随机种子
    }
    
    try:
        if optimize:
            print("正在对LightGBM进行高级优化...")
            # 恢复为与原始参数相似的配置，但略微调整以提高性能
            optimized_params = {
                'n_estimators': 1200,         # 增加树的数量，提高模型稳定性
                'max_depth': 5,               # 与原始参数相同
                'learning_rate': 0.04,        # 与原始参数相同
                'num_leaves': 16,             # 略微增加叶子节点数
                'min_child_samples': 20,      # 与原始参数相同
                'min_split_gain': 0.02,       # 与原始参数相同
                'reg_alpha': 0.02,            # 与原始参数相同
                'reg_lambda': 0.45,           # 与原始参数相同
                'feature_fraction': 0.3,      # 略微增加特征采样比例
                'bagging_fraction': 0.8,      # 略微增加样本采样率
                'bagging_freq': 4,            # 与原始参数相同
                'random_state': 42,
                'verbose': -1,                # 保持安静
                'deterministic': True         # 确保结果可复现
            }
            
            print(f"使用与原始参数相似的LightGBM参数: {optimized_params}")
            
            try:
                # 创建优化模型
                model = LightGBMModel(**optimized_params)
                
                # 训练模型 - 不使用早停参数
                model.train(X_train, y_train)
                
                # 设置类名
                model.set_class_names(class_names)
                
                # 验证模型是否成功训练
                if not model.is_trained:
                    raise RuntimeError("LightGBM优化模型训练失败，is_trained标志为False")
                
                # 做一次预测测试，确保模型可用
                try:
                    _ = model.predict(X_val[:1])
                    print("LightGBM优化模型可以正常预测")
                except Exception as pred_err:
                    print(f"LightGBM优化模型预测测试失败: {pred_err}")
                    raise RuntimeError("模型训练后无法正常预测")
                
                # 评估模型性能
                metrics = model.evaluate(X_val, y_val)
                
                # 保存优化结果
                os.makedirs('./models/optimized', exist_ok=True)
                model.save_model('./models/optimized/lightgbm_optimized.pkl')
                
                print("LightGBM优化模型训练和评估成功完成")
                
                # 保存到output_models目录
                model_path = f"./output_models/{model_name}.pkl"
                joblib.dump(model, model_path)
                print(f"模型已保存到 {model_path}")
                
                # 记录训练时间
                train_time = time.time() - start_time
                minutes = int(train_time / 60)
                seconds = int(train_time % 60)
                print(f"LightGBM优化模型训练完成，耗时: {minutes}分{seconds}秒")
                
                return model, metrics, train_time
            except Exception as e:
                print(f"LightGBM优化模型训练失败: {e}")
                # 回退到稳健简化模型
                print("尝试使用备选稳健参数创建优化标记的模型...")
                model = LightGBMModel(
                    n_estimators=800,         # 增加树的数量提高稳定性
                    max_depth=4,              # 适中的树深度
                    learning_rate=0.03,       # 较小的学习率增强稳定性
                    num_leaves=16,            # 适当的叶节点数量
                    min_child_samples=25,     # 较大的最小样本数
                    feature_fraction=0.4,     # 适中特征采样率
                    bagging_fraction=0.7,     # 适中样本采样率
                    random_state=42,          # 固定随机种子
                    verbose=-1                # 减少输出
                )
                model.train(X_train, y_train)
                model.set_class_names(class_names)
                # 此时不再使用验证集评估
                metrics = None
                
                # 保存到output_models目录
                model_path = f"./output_models/{model_name}.pkl"
                joblib.dump(model, model_path)
                print(f"备选模型已保存到 {model_path}")
                
                # 记录训练时间
                train_time = time.time() - start_time
                minutes = int(train_time / 60)
                seconds = int(train_time % 60)
                print(f"LightGBM备选模型训练完成，耗时: {minutes}分{seconds}秒")
                
                return model, metrics, train_time
        else:
            # 标准模式训练
            if use_cv:
                # 使用交叉验证
                metrics = evaluate_model_with_cv(
                    LightGBMModel, X_train, y_train, class_names,
                    cv=5, params=lgb_params, model_name="LightGBM"
                )
                # 创建完整模型
                model = LightGBMModel(**lgb_params)
                model.train(X_train, y_train)
                model.set_class_names(class_names)
            else:
                # 单次训练
                model = LightGBMModel(**lgb_params)
                model.train(X_train, y_train)
                model.set_class_names(class_names)
                metrics = None
            
            # 保存到output_models目录
            model_path = f"./output_models/{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"模型已保存到 {model_path}")
            
            # 记录训练时间
            train_time = time.time() - start_time
            minutes = int(train_time / 60)
            seconds = int(train_time % 60)
            print(f"LightGBM模型训练完成，耗时: {minutes}分{seconds}秒")
            
            return model, metrics, train_time
    
    except Exception as e:
        print(f"LightGBM模型训练过程中发生未处理的异常: {e}")
        print("返回None模型和None指标")
        # 记录训练失败时间
        train_time = time.time() - start_time
        minutes = int(train_time / 60)
        seconds = int(train_time % 60)
        print(f"LightGBM训练失败，耗时: {minutes}分{seconds}秒")
        return None, None, train_time

def train_mlp(X_train, y_train, X_val, y_val, class_names, use_cv=True, optimize=False):
    """训练MLP模型"""
    print("\n训练MLP模型...")
    
    # 检查是否存在已训练的模型
    model_name = "mlp"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        # 在验证集上评估已加载的模型
        metrics = model.evaluate(X_val, y_val)
        print(f"验证集指标: {metrics}")
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 检查是否使用CUDA
    if device == 'cuda':
        print("MLP模型将使用GPU加速训练")
        # 打印初始GPU内存状态
        print(f"训练前GPU已用内存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    mlp_params = {
        'input_size': X_train.shape[1],
        'hidden_sizes': [768, 512, 256],  # 更深的网络结构
        'output_size': len(class_names),
        'dropout_rate': 0.25,
        'lr': 0.0012,
        'weight_decay': 8e-6,
        'batch_size': 128 if device == 'cuda' else 64,  # GPU上使用更大的批次
        'epochs': 120,  # 增加训练轮次
        'device': device,
        'use_batch_norm': True,  # 使用批归一化
        'early_stopping_patience': 8,  # 早停机制
        'scheduler_type': 'cosine',  # 余弦退火学习率
        'lr_patience': 5,
        'lr_factor': 0.5,
        'min_lr': 1e-6,
        't_max': 10
    }
    
    try:
        if optimize:
            print("正在对MLP进行高级优化...")
            # 优化参数 - 基于实验结果进行强化优化
            optimized_params = {
                'input_size': X_train.shape[1],
                'hidden_sizes': [1024, 768, 512, 256, 128],  # 更深更宽的网络结构
                'output_size': len(class_names),
                'dropout_rate': 0.18,             # 降低dropout以减少欠拟合
                'lr': 0.0018,                    # 使用更合适的学习率
                'weight_decay': 4e-6,            # 降低权重衰减增强拟合能力
                'batch_size': 256 if device == 'cuda' else 64,  # GPU上使用更大批次加速训练
                'epochs': 180,                   # 增加训练轮次以确保收敛
                'device': device,
                'use_batch_norm': True,          # 保持批归一化
                'early_stopping_patience': 15,   # 增加早停耐心确保充分训练
                'scheduler_type': 'one_cycle',   # 使用one-cycle策略，通常效果更好
                'lr_patience': 10,                # 增加学习率调整耐心
                'lr_factor': 0.6,                # 使用更温和的学习率衰减
                'min_lr': 1e-7,                  # 降低最小学习率
                't_max': 20                      # 增加周期
            }
            
            # 如果使用CUDA，开启cudnn benchmark可以加速训练
            if device == 'cuda':
                print("为MLP优化模型启用cudnn benchmark以加速训练")
                torch.backends.cudnn.benchmark = True
            
            print(f"使用增强的优化参数: {optimized_params}")
            
            # 创建模型
            model = MLPModel(**optimized_params)
            # 使用验证集进行训练
            model.train(X_train, y_train, X_val, y_val)
            model.set_class_names(class_names)
            
            # 评估模型性能
            metrics = model.evaluate(X_val, y_val)
            
            # 保存混淆矩阵和训练历史
            os.makedirs("./plots", exist_ok=True)
            try:
                model.plot_confusion_matrix(X_val, y_val)
                plt.savefig("./plots/mlp_optimized_confusion_matrix.png")
                plt.close()
                
                # 保存训练历史图
                model.plot_training_history()
                plt.savefig("./plots/mlp_optimized_training_history.png")
                plt.close()
            except Exception as e:
                print(f"MLP优化模型图表生成失败: {e}")
            
            # 保存优化结果
            os.makedirs('./models/optimized', exist_ok=True)
            model.save_model('./models/optimized/mlp_optimized.pkl')
            
            # 保存到output_models目录
            model_path = f"./output_models/{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"模型已保存到 {model_path}")
            
            # 记录训练时间
            train_time = time.time() - start_time
            minutes = int(train_time / 60)
            seconds = int(train_time % 60)
            print(f"MLP优化模型训练完成，耗时: {minutes}分{seconds}秒")
            
            return model, metrics, train_time
            
        elif use_cv:
            # 使用交叉验证
            metrics = evaluate_model_with_cv(
                MLPModel, X_train, y_train, class_names,
                cv=5, params=mlp_params, model_name="MLP"
            )
            # 创建完整模型
            model = MLPModel(**mlp_params)
            model.train(X_train, y_train, X_val, y_val)
            model.set_class_names(class_names)
        else:
            # 单次训练
            model = MLPModel(**mlp_params)
            model.train(X_train, y_train, X_val, y_val)
            model.set_class_names(class_names)
            metrics = None
            
            # 保存MLP模型混淆矩阵
            os.makedirs("./plots", exist_ok=True)
            try:
                model.plot_confusion_matrix(X_val, y_val)
                plt.savefig("./plots/mlp_confusion_matrix.png")
                plt.close()
                
                # 保存训练历史图
                model.plot_training_history()
                plt.savefig("./plots/mlp_training_history.png")
                plt.close()
            except Exception as e:
                print(f"MLP模型图表生成失败: {e}")
        
        # 保存到output_models目录
        model_path = f"./output_models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
        
        # 记录训练时间
        train_time = time.time() - start_time
        minutes = int(train_time / 60)
        seconds = int(train_time % 60)
        print(f"MLP模型训练完成，耗时: {minutes}分{seconds}秒")
        
        return model, metrics, train_time
    
    except Exception as e:
        print(f"MLP模型训练失败: {e}")
        # 记录训练失败时间
        train_time = time.time() - start_time
        print(f"MLP训练失败，耗时: {int(train_time / 60)}分{int(train_time % 60)}秒")
        return None, None, train_time

def train_voting_model(base_models, X_train, y_train, X_val, y_val, class_names, voting='soft', f1_threshold=0.80):
    """
    训练投票模型
    
    参数:
    base_models (list): 基础模型列表，每个元素为(名称, 模型)元组
    X_train (np.ndarray): 训练特征
    y_train (np.ndarray): 训练标签
    X_val (np.ndarray): 验证特征（用于评估和筛选模型）
    y_val (np.ndarray): 验证标签
    class_names (list): 类别名称
    voting (str): 投票类型，'soft'或'hard'
    f1_threshold (float): F1分数阈值，只有高于此阈值的模型才会参与投票
    """
    print(f"\n训练{voting}投票模型...")
    
    # 检查是否存在已训练的模型
    model_name = f"voting_{voting}"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 创建模型列表，过滤掉不兼容的模型并评估模型性能
    compatible_models = []
    filtered_models = []
    mlp_model = None
    model_f1_scores = {}
    
    # 检查输入的模型列表是否包含MLP
    has_mlp = False
    for name, _ in base_models:
        if name == 'MLP':
            has_mlp = True
            break
    
    if not has_mlp:
        print("警告：输入的模型列表中不包含MLP模型")
    else:
        print("检测到MLP模型在输入列表中")
    
    print(f"开始在验证集上评估和筛选模型(F1阈值: {f1_threshold})...")
    
    # 分离MLP模型和其他模型
    for name, model in base_models:
        if model is not None:
            # 评估模型F1分数
            try:
                metrics = model.evaluate(X_val, y_val)
                model_f1 = metrics['f1']
                model_f1_scores[name] = model_f1
                print(f"模型{name} F1分数: {model_f1:.4f}")
                
                # 仅保留高于阈值的模型
                if model_f1 >= f1_threshold:
                    if name == 'MLP':
                        mlp_model = model
                        print(f"MLP模型F1分数({model_f1:.4f})高于阈值，将用于特殊处理")
                    elif hasattr(model, 'model') and hasattr(model.model, 'predict') and hasattr(model.model, 'fit'):
                        compatible_models.append((name, model.model))
                        filtered_models.append((name, model))
                        print(f"模型{name}符合标准格式且性能良好，添加到投票模型")
                    else:
                        print(f"模型{name}不是标准格式，尝试直接添加")
                        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                            # 尝试直接使用该模型（可能是自定义的模型类）
                            compatible_models.append((name, model))
                            filtered_models.append((name, model))
                            print(f"成功添加模型{name}到投票模型")
                        else:
                            print(f"跳过模型{name}，因为找不到兼容的预测方法")
                else:
                    print(f"模型{name} F1分数({model_f1:.4f})低于阈值{f1_threshold}，不参与投票")
            except Exception as e:
                print(f"评估模型{name}时出错: {e}")
                print(f"跳过模型{name}，不参与投票")
    
    # 检查筛选后的模型数量
    if not compatible_models:
        print("没有找到符合条件的模型，无法创建投票模型")
        return None, None, train_time
    
    print(f"筛选后使用 {len(compatible_models)} 个模型创建投票模型: {[name for name, _ in compatible_models]}")
    print(f"被筛选掉的模型: {[name for name, score in model_f1_scores.items() if score < f1_threshold]}")
    
    # 创建并训练投票模型
    voting_model = VotingModel(compatible_models, voting=voting)
    voting_model.train(X_train, y_train)
    voting_model.set_class_names(class_names)
    
    # 如果存在MLP模型并且性能良好，集成它
    if mlp_model is not None and model_f1_scores.get('MLP', 0) >= f1_threshold:
        print(f"将高性能的MLP模型集成到投票模型中")
        
        # 创建增强型投票模型
        from models.voting_model import MlpEnhancedVotingModel
        enhanced_voting_model = MlpEnhancedVotingModel(voting_model, mlp_model, voting)
        
        print(f"成功将MLP模型集成到{voting}投票模型中")
        
        # 使用增强型投票模型替代原始投票模型
        voting_model = enhanced_voting_model
    else:
        if mlp_model is not None:
            print(f"MLP模型F1分数({model_f1_scores.get('MLP', 0):.4f})低于阈值{f1_threshold}，不参与投票")
        else:
            print(f"警告：没有找到MLP模型进行集成，{voting}投票模型将不包含MLP")
    
    # 在验证集上评估投票模型
    metrics = voting_model.evaluate(X_val, y_val)
    print(f"投票模型在验证集上的F1分数: {metrics['f1']:.4f}")
    
    # 保存到output_models目录
    model_path = f"./output_models/{model_name}.pkl"
    joblib.dump(voting_model, model_path)
    print(f"投票模型({voting})已保存到 {model_path}")
    
    # 记录训练时间
    train_time = time.time() - start_time
    minutes = int(train_time / 60)
    seconds = int(train_time % 60)
    print(f"投票模型({voting})训练完成，耗时: {minutes}分{seconds}秒")
    
    return voting_model, metrics, train_time

def train_attention_voting_model(base_models, X_train, y_train, X_val, y_val, class_names, f1_threshold=0.85, optimize=False):
    """
    训练基于注意力机制的投票模型
    
    参数:
    base_models (list): 基础模型列表，每个元素为(名称, 模型)元组
    X_train (np.ndarray): 训练特征
    y_train (np.ndarray): 训练标签
    X_val (np.ndarray): 验证特征
    y_val (np.ndarray): 验证标签
    class_names (list): 类别名称
    f1_threshold (float): F1分数阈值，只有高于此阈值的模型才会参与投票
    optimize (bool): 是否为优化版本
    """
    if not ATTENTION_MODEL_AVAILABLE:
        print("注意力投票模型不可用，请确保已安装所需依赖")
        return None, None, 0
    
    print("\n训练注意力投票模型...")
    
    # 检查是否存在已训练的模型
    model_name = "attention_voting_model"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 创建模型列表，过滤掉性能不佳的模型
    filtered_models = []
    model_f1_scores = {}
    mlp_included = False
    
    print(f"开始在验证集上评估和筛选模型(F1阈值: {f1_threshold})...")
    
    # 过滤出只包含基础模型的列表，排除集成模型
    base_models_filtered = []
    for name, model in base_models:
        # 排除投票模型和注意力投票模型
        if not any(excluded in name for excluded in ['投票模型', '注意力投票模型']):
            base_models_filtered.append((name, model))
        else:
            print(f"排除集成模型 {name}，不参与注意力投票")
    
    print(f"过滤后的基础模型数量: {len(base_models_filtered)}")
    
    # 如果是优化版本，首先收集所有模型版本的F1分数
    if optimize:
        print("优化模式：将为每个模型类型选择F1分数更高的版本")
        
        # 存储每个模型类型的最佳版本
        best_models = {}
        model_versions = {}
        
        # 评估每个模型并记录F1分数
        for name, model in base_models_filtered:
            if model is not None:
                try:
                    # 评估模型性能
                    metrics = model.evaluate(X_val, y_val)
                    model_f1 = metrics['f1']
                    
                    # 提取基础模型名称（不带"优化"标记）
                    base_name = name.replace('(优化)', '').strip()
                    
                    # 记录这个模型版本
                    if base_name not in model_versions:
                        model_versions[base_name] = []
                    model_versions[base_name].append((name, model, model_f1))
                    
                    print(f"模型{name} F1分数: {model_f1:.4f}")
                except Exception as e:
                    print(f"评估模型{name}时出错: {e}")
        
        # 为每个模型类型选择F1分数最高的版本
        for base_name, versions in model_versions.items():
            if versions:
                # 按F1分数排序并选择最佳版本
                best_version = sorted(versions, key=lambda x: x[2], reverse=True)[0]
                best_name, best_model, best_f1 = best_version
                
                print(f"模型类型 {base_name} 的最佳版本是 {best_name}，F1分数: {best_f1:.4f}")
                
                # 只有F1分数高于阈值的模型才会被选择
                if best_f1 >= f1_threshold:
                    filtered_models.append((best_name, best_model))
                    model_f1_scores[best_name] = best_f1
                    print(f"添加模型 {best_name} 到优化版注意力投票模型")
                    
                    if 'MLP' in best_name:
                        mlp_included = True
                else:
                    print(f"模型 {best_name} F1分数({best_f1:.4f})低于阈值{f1_threshold}，不参与投票")
    else:
        # 标准模式：直接评估每个模型
        for name, model in base_models_filtered:
            if model is not None:
                try:
                    # 评估模型性能
                    metrics = model.evaluate(X_val, y_val)
                    model_f1 = metrics['f1']
                    model_f1_scores[name] = model_f1
                    print(f"模型{name} F1分数: {model_f1:.4f}")
                    
                    # 仅保留高于阈值的模型
                    if model_f1 >= f1_threshold:
                        filtered_models.append((name, model))
                        print(f"模型{name}性能良好，添加到注意力投票模型")
                        if name == 'MLP':
                            mlp_included = True
                    else:
                        print(f"模型{name} F1分数({model_f1:.4f})低于阈值{f1_threshold}，不参与投票")
                except Exception as e:
                    print(f"评估模型{name}时出错: {e}")
                    print(f"跳过模型{name}，不参与投票")
    
    # 检查是否有足够的模型
    if len(filtered_models) < 2:
        print(f"筛选后只有 {len(filtered_models)} 个模型，无法创建有效的注意力投票模型")
        return None, None, train_time
    
    # 检查是否包含MLP
    if not mlp_included:
        print("注意：MLP模型未能添加到注意力投票模型中")
    else:
        print("MLP模型已成功包含在注意力投票模型中")
    
    print(f"使用 {len(filtered_models)} 个模型创建注意力投票模型: {[name for name, _ in filtered_models]}")
    if model_f1_scores:
        print(f"被筛选掉的模型: {[name for name, score in model_f1_scores.items() if score < f1_threshold]}")
    
    # 计算注意力头数和隐藏层大小
    # 调整隐藏层大小，确保能被注意力头数整除
    attention_heads = 8  # 使用8个注意力头
    hidden_size = 768    # 默认隐藏层大小
    
    # 确保hidden_size能被注意力头数整除
    if hidden_size % attention_heads != 0:
        # 向上取整到能被注意力头数整除的最小值
        hidden_size = ((hidden_size // attention_heads) + 1) * attention_heads
        print(f"已调整隐藏层大小为 {hidden_size}，以确保能被注意力头数({attention_heads})整除")
    
    # 使用更保守的批次大小，减少内存占用
    batch_size = 64 if device == 'cuda' else 32
    
    try:
        # 创建注意力投票模型(启用早停和优化)，增加隐藏层数
        attention_model = AttentionVotingModel(
            models=filtered_models,
            hidden_size=hidden_size,  # 确保为注意力头数的整数倍
            num_attention_heads=attention_heads,  
            dropout_prob=0.2,  # 防止过拟合
            lr=0.001,  # 使用较小的学习率，提高稳定性
            weight_decay=1e-4,  # 权重衰减
            batch_size=batch_size,  # 使用更保守的批次大小
            epochs=60,  # 增加训练轮数
            device=device
        )
        
        # 准备早停参数
        early_stopping_params = {
            'patience': 10,  # 验证损失连续10轮不改善则早停
            'min_delta': 0.0005  # 降低最小改善阈值，更敏感地捕捉进步
        }
        
        # 训练模型(启用早停)
        print("注意力模型训练启用早停机制，patience=10")
        try:
            attention_model.train(
                X_train, y_train, X_val, y_val, 
                early_stopping=True,
                early_stopping_params=early_stopping_params
            )
            attention_model.set_class_names(class_names)
            
            # 在验证集上评估模型
            metrics = attention_model.evaluate(X_val, y_val)
            print(f"注意力投票模型在验证集上的F1分数: {metrics['f1']:.4f}")
            
            # 保存训练好的模型
            os.makedirs("./output_models", exist_ok=True)
            model_path = f"./output_models/{model_name}.pkl"
            joblib.dump(attention_model, model_path)
            print(f"注意力投票模型已保存到 {model_path}")
            
            # 记录训练时间
            train_time = time.time() - start_time
            minutes = int(train_time / 60)
            seconds = int(train_time % 60)
            print(f"注意力投票模型训练完成，耗时: {minutes}分{seconds}秒")
            
            return attention_model, metrics, train_time
        except Exception as train_error:
            print(f"注意力模型训练过程中出错: {train_error}")
            print("尝试创建简化版注意力模型...")
            
            # 使用更少的模型和更简单的配置重新尝试
            if len(filtered_models) > 3:
                # 只使用前3个性能最好的模型
                sorted_models = sorted(
                    [(name, model) for name, model in filtered_models],
                    key=lambda x: model_f1_scores.get(x[0], 0),
                    reverse=True
                )[:3]
                
                print(f"使用简化模型集: {[name for name, _ in sorted_models]}")
                
                # 创建简化版注意力模型
                simple_attention_model = AttentionVotingModel(
                    models=sorted_models,
                    hidden_size=256,  # 减小隐藏层大小
                    num_attention_heads=4,  # 减少注意力头数量
                    dropout_prob=0.1,
                    lr=0.001,
                    batch_size=32,  # 使用更小的批次大小
                    epochs=20,  # 减少训练轮数
                    device=device
                )
                
                # 训练简化模型
                simple_attention_model.train(
                    X_train, y_train, X_val, y_val,
                    early_stopping=True,
                    early_stopping_params={'patience': 8, 'min_delta': 0.0005}
                )
                simple_attention_model.set_class_names(class_names)
                
                # 评估简化模型
                metrics = simple_attention_model.evaluate(X_val, y_val)
                
                # 保存简化模型
                model_path = f"./output_models/{model_name}.pkl"
                joblib.dump(simple_attention_model, model_path)
                
                # 记录训练时间
                train_time = time.time() - start_time
                minutes = int(train_time / 60)
                seconds = int(train_time % 60)
                print(f"简化版注意力投票模型训练完成，耗时: {minutes}分{seconds}秒")
                
                return simple_attention_model, metrics, train_time
            else:
                # 模型太少，无法创建简化版
                print("基础模型数量不足，无法创建简化版注意力模型")
                return None, None, train_time
    except Exception as e:
        print(f"创建注意力投票模型时出错: {e}")
        print("无法创建注意力投票模型")
        return None, None, train_time

def evaluate_all_models(models_dict, X_test, y_test, force_evaluate=False):
    """
    评估所有模型
    
    参数:
    models_dict (dict): 模型字典，键为模型名称，值为模型对象
    X_test (np.ndarray): 测试特征
    y_test (np.ndarray): 测试标签
    force_evaluate (bool): 是否强制评估所有模型，不检查训练状态
    
    返回:
    pd.DataFrame: 模型评估结果
    """
    results = []
    skipped_models = []
    
    for model_name, model in models_dict.items():
        if model is None:
            print(f"\n跳过{model_name}模型(为None)")
            skipped_models.append(model_name)
            continue
            
        print(f"\n评估{model_name}模型...")
        
        # 如果设置了强制评估，则跳过检查直接评估
        if force_evaluate:
            try:
                metrics = model.evaluate(X_test, y_test)
                results.append({
                    '模型': model_name,
                    '准确率': metrics['accuracy'],
                    '精确率': metrics['precision'],
                    '召回率': metrics['recall'],
                    'F1分数': metrics['f1']
                })
                print(f"{model_name}模型评估成功")
            except Exception as e:
                print(f"强制评估{model_name}模型时出错: {e}")
                print(f"跳过{model_name}模型的评估")
                skipped_models.append(model_name)
            continue
            
        # 常规评估流程
        try:
            # 先检查模型是否明确标记为已训练
            is_trained = False
            
            # 1. 使用内置的训练标志(优先检查)
            if hasattr(model, 'is_trained'):
                is_trained = model.is_trained
                if is_trained:
                    print(f"{model_name}模型已明确标记为已训练")
                
            # 如果没有明确的训练标志或标志为False，继续检查
            if not is_trained:
                # 尝试验证模型可用性的方式：先做一次预测测试
                try:
                    print(f"检查{model_name}模型是否可用...")
                    test_X = X_test[:1]  # 使用一个样本进行测试
                    _ = model.predict(test_X)
                    is_trained = True
                    print(f"{model_name}模型预测测试成功，可以进行评估")
                except Exception as pred_err:
                    print(f"{model_name}模型预测测试失败: {pred_err}")
                    print(f"模型可能未正确训练或不兼容当前数据")
                    skipped_models.append(model_name)
                    continue

            # 模型通过了检查，进行评估
            try:
                metrics = model.evaluate(X_test, y_test)
                results.append({
                    '模型': model_name,
                    '准确率': metrics['accuracy'],
                    '精确率': metrics['precision'],
                    '召回率': metrics['recall'],
                    'F1分数': metrics['f1']
                })
                print(f"{model_name}模型评估完成")
            except Exception as eval_err:
                print(f"评估{model_name}模型时出错: {eval_err}")
                print(f"跳过{model_name}模型的评估")
                skipped_models.append(model_name)
                
        except Exception as outer_e:
            print(f"处理{model_name}模型时发生错误: {outer_e}")
            print(f"跳过{model_name}模型的评估")
            skipped_models.append(model_name)
    
    # 创建DataFrame并排序
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('F1分数', ascending=False).reset_index(drop=True)
        # 打印跳过的模型
        if skipped_models:
            print(f"\n以下模型被跳过评估: {', '.join(skipped_models)}")
        return results_df
    else:
        print("没有可用于评估的模型")
        return pd.DataFrame()

def plot_models_comparison(results_df):
    """可视化模型比较"""
    # 检查DataFrame是否为空
    if results_df.empty:
        print("没有模型结果可显示")
        return
        
    # 导入matplotlib配置
    from models.plt_config import set_plt_configs
    set_plt_configs()  # 确保中文显示正确
    
    plt.figure(figsize=(14, 10))
    
    # 按F1分数排序
    results_df = results_df.sort_values('F1分数')
    
    # 创建颜色映射，区分标准模型和优化模型
    colors = []
    for model_name in results_df['模型']:
        if '优化' in model_name:
            colors.append('orange')  # 优化模型使用橙色
        else:
            colors.append('skyblue')  # 标准模型使用蓝色
    
    # 绘制条形图
    bars = plt.barh(results_df['模型'], results_df['F1分数'], color=colors)
    
    # 添加数值标签
    for i, v in enumerate(results_df['F1分数']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=12)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='标准模型'),
        Patch(facecolor='orange', label='优化模型')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.xlabel('F1分数', fontsize=14, fontweight='bold')
    plt.ylabel('模型', fontsize=14, fontweight='bold')
    plt.title('模型F1分数比较', fontsize=16, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs('./plots', exist_ok=True)
    
    # 保存图片
    plt.savefig('./plots/models_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("模型F1分数比较图已保存到 ./plots/models_f1_comparison.png")

def plot_training_time_comparison(training_times):
    """可视化各模型训练时间比较"""
    if not training_times:
        print("没有训练时间数据可供可视化")
        return
    
    # 导入matplotlib配置
    from models.plt_config import set_plt_configs
    set_plt_configs()  # 确保中文显示正确
        
    # 限制模型数量，只显示前15个模型
    if len(training_times) > 15:
        print(f"训练时间数据过多({len(training_times)}个)，只显示训练时间最长的15个模型")
        sorted_times = sorted(training_times.items(), key=lambda x: x[1], reverse=True)
        training_times = {k: v for k, v in sorted_times[:15]}
        
    plt.figure(figsize=(12, 8))
    
    # 按训练时间排序
    sorted_items = sorted(training_times.items(), key=lambda x: x[1])
    model_names = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]
    
    # 设置不同的颜色
    colors = []
    for name in model_names:
        if 'MLP' in name:
            colors.append('cornflowerblue')
        elif '注意力' in name:
            colors.append('darkorange')
        elif '投票' in name:
            colors.append('lightgreen')
        elif '优化' in name:
            colors.append('salmon')
        else:
            colors.append('lightblue')
    
    # 绘制条形图
    bars = plt.barh(model_names, times, color=colors)
    
    # 为每个条形添加标签
    for i, v in enumerate(times):
        # 显示时间（分:秒）
        minutes = int(v / 60)
        seconds = int(v % 60)
        time_str = f"{minutes}:{seconds:02d}"
        plt.text(v + 5, i, time_str, va='center', fontsize=12)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='cornflowerblue', label='MLP模型'),
        Patch(facecolor='darkorange', label='注意力模型'),
        Patch(facecolor='lightgreen', label='投票模型'),
        Patch(facecolor='salmon', label='优化模型'),
        Patch(facecolor='lightblue', label='传统模型')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.xlabel('训练时间 (秒)', fontsize=14, fontweight='bold')
    plt.ylabel('模型', fontsize=14, fontweight='bold')
    plt.title('各模型训练时间对比', fontsize=16, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/model_training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("模型训练时间比较图已保存到 ./plots/model_training_time_comparison.png")

def train_pretrained_transformer(X_train, y_train, X_val, y_val, class_names, optimize=False):
    """训练预训练特征Transformer模型"""
    if not TRANSFORMER_MODEL_AVAILABLE:
        print("预训练Transformer模型不可用，请确保已安装所需依赖")
        return None, None, 0
    
    print("\n训练预训练特征Transformer模型...")
    
    # 检查是否存在已训练的模型
    model_name = "pretrained_transformer"
    if optimize:
        model_name += "_optimized"
    
    # 尝试加载已有模型
    model, metrics, train_time = check_and_load_model(model_name)
    if model is not None:
        # 在验证集上评估已加载的模型
        metrics = model.evaluate(X_val, y_val)
        print(f"验证集指标: {metrics}")
        return model, metrics, train_time
    
    # 如果没有找到已训练的模型，则进行训练
    start_time = time.time()
    
    # 设置参数
    transformer_params = {
        'input_dim': X_train.shape[1],
        'num_classes': len(class_names),
        'd_model': 768,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'lr': 1e-4,
        'batch_size': 64,
        'epochs': 50,
        'device': device
    }
    
    if optimize:
        print("正在对预训练Transformer模型进行优化...")
        # 优化参数
        optimized_params = {
            'input_dim': X_train.shape[1],
            'num_classes': len(class_names),
            'd_model': 768,
            'nhead': 8,
            'num_layers': 6,  # 增加层数
            'dim_feedforward': 2048,
            'dropout': 0.15,  # 调整dropout
            'lr': 8e-5,  # 降低学习率
            'batch_size': 128 if device == 'cuda' else 64,  # 增大批次大小
            'epochs': 80,  # 增加训练轮数
            'device': device
        }
        
        # 创建并训练模型
        model = PretrainedTransformerModel(**optimized_params)
        model.train(X_train, y_train, X_val, y_val)
        model.set_class_names(class_names)
        
        # 评估模型
        metrics = model.evaluate(X_val, y_val)
        
        # 保存模型
        os.makedirs('./models/optimized', exist_ok=True)
        model.save_model('./models/optimized/pretrained_transformer_optimized.pkl')
    else:
        # 创建并训练模型
        model = PretrainedTransformerModel(**transformer_params)
        model.train(X_train, y_train, X_val, y_val)
        model.set_class_names(class_names)
        
        # 评估模型
        metrics = model.evaluate(X_val, y_val)
    
    # 保存到output_models目录
    model_path = f"./output_models/{model_name}.pkl"
    model.save_model(model_path)
    print(f"模型已保存到 {model_path}")
    
    # 记录训练时间
    train_time = time.time() - start_time
    minutes = int(train_time / 60)
    seconds = int(train_time % 60)
    print(f"预训练Transformer模型训练完成，耗时: {minutes}分{seconds}秒")
    
    return model, metrics, train_time

def main():
    """主函数"""
    # 创建目录
    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./output_models", exist_ok=True)  # 创建输出模型目录
    
    # 记录模型训练时间
    training_times = {}
    
    # 检查是否已存在提取的特征
    if os.path.exists("./extracted_features/X_train.npy"):
        print("加载已提取的特征...")
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features()
    else:
        print("提取特征...")
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = extract_and_save_features()
    
    if X_train is None:
        print("特征加载失败，程序退出")
        return
    
    # 选择是否使用交叉验证进行更科学的模型评估 
    use_cv = False
    
    # 启用高级优化，利用GPU性能
    print("\n检测到CUDA可用，启用高级优化以充分利用GPU...")
    # 设置为True启用模型优化，允许运行优化模式
    optimize = True
    if device == 'cuda':
        print(f"GPU加速已启用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 在非优化模式下运行
    print("\n================ 标准模式 (optimize=False) ================")
    # 训练各个基础模型
    print("\n开始训练各个基础模型...")
    
    # 1. 逻辑回归模型
    lr_model, lr_metrics, lr_time = train_logistic_regression(X_train, y_train, X_val, y_val, class_names, use_cv, False)
    training_times["逻辑回归"] = lr_time
    
    # 2. SVM模型
    svm_model, svm_metrics, svm_time = train_svm(X_train, y_train, X_val, y_val, class_names, use_cv, False)
    training_times["SVM"] = svm_time
    
    # 3. 随机森林模型
    rf_model, rf_metrics, rf_time = train_random_forest(X_train, y_train, class_names, use_cv, False)
    training_times["随机森林"] = rf_time
    
    # 4. XGBoost模型
    xgb_model, xgb_metrics, xgb_time = train_xgboost(X_train, y_train, X_val, y_val, class_names, use_cv, False)
    training_times["XGBoost"] = xgb_time
    
    # 5. LightGBM模型
    lgb_model, lgb_metrics, lgb_time = train_lightgbm(X_train, y_train, X_val, y_val, class_names, use_cv, False)
    training_times["LightGBM"] = lgb_time
    
    # 6. MLP模型
    mlp_model, mlp_metrics, mlp_time = train_mlp(X_train, y_train, X_val, y_val, class_names, use_cv, False)
    training_times["MLP"] = mlp_time
    
    # 7. 预训练Transformer模型
    if TRANSFORMER_MODEL_AVAILABLE:
        transformer_model, transformer_metrics, transformer_time = train_pretrained_transformer(
            X_train, y_train, X_val, y_val, class_names, False
        )
        training_times["预训练Transformer"] = transformer_time
    
    # 收集所有模型
    models_dict_standard = {
        '逻辑回归': lr_model,
        'SVM': svm_model,
        '随机森林': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'MLP': mlp_model
    }
    
    # 添加Transformer模型
    if TRANSFORMER_MODEL_AVAILABLE and 'transformer_model' in locals() and transformer_model is not None:
        models_dict_standard['预训练Transformer'] = transformer_model

    # 添加投票模型 (不包括MLP，因为它使用不同的接口)
    print("\n训练投票模型...")
    base_models = [
        ('逻辑回归', lr_model),
        ('SVM', svm_model),
        ('随机森林', rf_model),
        ('XGBoost', xgb_model),
        ('LightGBM', lgb_model),
        ('MLP', mlp_model)  # 添加MLP到投票模型中
    ]
    voting_model, voting_metrics, voting_time = train_voting_model(base_models, X_train, y_train, X_val, y_val, class_names, voting='soft')
    models_dict_standard['投票模型(soft)'] = voting_model
    training_times["投票模型(soft)"] = voting_time
    
    # 硬投票模型
    voting_hard_model, voting_hard_metrics, voting_hard_time = train_voting_model(base_models, X_train, y_train, X_val, y_val, class_names, voting='hard')
    models_dict_standard['投票模型(hard)'] = voting_hard_model
    training_times["投票模型(hard)"] = voting_hard_time
    
    # 尝试添加注意力投票模型 (如果可用)
    if ATTENTION_MODEL_AVAILABLE:
        print("\n训练注意力投票模型...")
        # 开始训练计时
        attention_start_time = time.time()
        
        attention_voting_model, attention_metrics, attention_time = train_attention_voting_model(base_models, X_train, y_train, X_val, y_val, class_names)
        models_dict_standard['注意力投票模型'] = attention_voting_model
        training_times["注意力投票模型"] = attention_time
        minutes = int(attention_time / 60)
        seconds = int(attention_time % 60)
        print(f"注意力投票模型训练完成，耗时: {minutes}分{seconds}秒")
        
        # 保存训练好的模型
        if attention_voting_model is not None:
            os.makedirs("./output_models", exist_ok=True)
            model_path = "./output_models/attention_voting_model.pkl"
            joblib.dump(attention_voting_model, model_path)
            print(f"注意力投票模型已保存到 {model_path}")

    # 评估所有标准模型
    print("\n评估标准模型在测试集上的表现...")
    results_df_standard = evaluate_all_models(models_dict_standard, X_test, y_test)

    # 显示结果
    print("\n标准模式模型性能比较:")
    print(results_df_standard)
    
    # 在优化模式下运行
    print("\n\n================ 优化模式 (optimize=True) ================")
    # 切换为启用优化模式
    optimize = True
    
    # 优化CUDA性能
    if device == 'cuda':
        # 清理GPU缓存
        torch.cuda.empty_cache()
        print(f"已清理GPU缓存，当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # 设置更积极的CUDA优化
        torch.backends.cudnn.benchmark = True
        print("已启用CUDA性能优化")
    
    # 优化所有模型
    print("\n开始优化所有模型...")
    
    # 1. 优化逻辑回归模型
    print("\n优化逻辑回归模型...")
    lr_model_opt, lr_metrics_opt, lr_time_opt = train_logistic_regression(X_train, y_train, X_val, y_val, class_names, False, optimize)
    training_times["逻辑回归优化"] = lr_time_opt
    
    # 2. 优化SVM模型
    print("\n优化SVM模型...")
    svm_model_opt, svm_metrics_opt, svm_time_opt = train_svm(X_train, y_train, X_val, y_val, class_names, False, optimize)
    training_times["SVM优化"] = svm_time_opt
    
    # 3. 优化随机森林模型
    print("\n优化随机森林模型...")
    rf_model_opt, rf_metrics_opt, rf_time_opt = train_random_forest(X_train, y_train, class_names, False, optimize)
    training_times["随机森林优化"] = rf_time_opt
    
    # 4. 优化XGBoost模型
    print("\n优化XGBoost模型...")
    xgb_model_opt, xgb_metrics_opt, xgb_time_opt = train_xgboost(X_train, y_train, X_val, y_val, class_names, False, optimize)
    training_times["XGBoost优化"] = xgb_time_opt
    
    # 5. 优化LightGBM模型
    print("\n优化LightGBM模型...")
    lgb_model_opt, lgb_metrics_opt, lgb_time_opt = train_lightgbm(X_train, y_train, X_val, y_val, class_names, False, optimize)
    training_times["LightGBM优化"] = lgb_time_opt
    
    # 6. 优化MLP模型
    print("\n优化MLP模型...")
    mlp_model_opt, mlp_metrics_opt, mlp_time_opt = train_mlp(X_train, y_train, X_val, y_val, class_names, False, optimize)
    training_times["MLP优化"] = mlp_time_opt
    
    # 7. 优化预训练Transformer模型
    if TRANSFORMER_MODEL_AVAILABLE:
        print("\n优化预训练Transformer模型...")
        transformer_model_opt, transformer_metrics_opt, transformer_time_opt = train_pretrained_transformer(
            X_train, y_train, X_val, y_val, class_names, True
        )
        training_times["预训练Transformer优化"] = transformer_time_opt
    
    # 收集优化后的模型
    models_dict_optimized = {}
    
    # 逐个添加优化模型，确保只收集成功训练的模型
    if lr_model_opt is not None:
        models_dict_optimized['逻辑回归(优化)'] = lr_model_opt
    if svm_model_opt is not None:
        models_dict_optimized['SVM(优化)'] = svm_model_opt
    if rf_model_opt is not None:
        models_dict_optimized['随机森林(优化)'] = rf_model_opt
    if xgb_model_opt is not None:
        models_dict_optimized['XGBoost(优化)'] = xgb_model_opt
    if lgb_model_opt is not None:
        models_dict_optimized['LightGBM(优化)'] = lgb_model_opt
    if mlp_model_opt is not None:
        models_dict_optimized['MLP(优化)'] = mlp_model_opt
    
    # 检查并打印优化模型的状态
    for name, model in models_dict_optimized.items():
        if model is None:
            print(f"警告: {name} 模型为None，训练可能失败")
        else:
            print(f"成功添加 {name} 模型到评估列表")
    
    # 添加投票模型 (优化版) - 仅使用成功训练的模型
    print("\n训练优化版投票模型...")
    base_models_opt = []
    for name, model in [
        ('逻辑回归', lr_model_opt),
        ('SVM', svm_model_opt),
        ('随机森林', rf_model_opt),
        ('XGBoost', xgb_model_opt),
        ('LightGBM', lgb_model_opt),
        ('MLP', mlp_model_opt)  # 添加MLP到优化版投票模型
    ]:
        if model is not None:
            base_models_opt.append((name, model))
            print(f"添加 {name} 到优化投票模型")
        else:
            print(f"跳过 {name}，因为模型为None")
    
    # 只有当有足够的基本模型时才创建投票模型
    if len(base_models_opt) >= 2:
        voting_model_opt, voting_opt_metrics, voting_opt_time = train_voting_model(base_models_opt, X_train, y_train, X_val, y_val, class_names, voting='soft')
        if voting_model_opt is not None:
            models_dict_optimized['投票模型(软投票优化)'] = voting_model_opt
            training_times["投票模型(软投票优化)"] = voting_opt_time
        
        # 硬投票模型(优化版)
        voting_hard_model_opt, voting_hard_opt_metrics, voting_hard_opt_time = train_voting_model(base_models_opt, X_train, y_train, X_val, y_val, class_names, voting='hard')
        if voting_hard_model_opt is not None:
            models_dict_optimized['投票模型(硬投票优化)'] = voting_hard_model_opt
            training_times["投票模型(硬投票优化)"] = voting_hard_opt_time
        
        # 尝试添加注意力投票模型(优化版)
        if ATTENTION_MODEL_AVAILABLE and len(base_models_opt) >= 2:
            print("\n训练优化版注意力投票模型...")
            
            # 为优化版注意力模型收集所有的模型版本（包括标准版和优化版）
            all_models_for_attention = []
            
            # 首先添加标准版模型
            if 'models_dict_standard' in locals():
                for name, model in models_dict_standard.items():
                    if model is not None and not any(excluded in name for excluded in ['投票模型', '注意力投票模型']):
                        all_models_for_attention.append((name, model))
            else:
                # 添加单独训练的标准模型
                standard_models = {
                    '逻辑回归': lr_model,
                    'SVM': svm_model,
                    '随机森林': rf_model,
                    'XGBoost': xgb_model,
                    'LightGBM': lgb_model,
                    'MLP': mlp_model
                }
                for name, model in standard_models.items():
                    if model is not None:
                        all_models_for_attention.append((name, model))
            
            # 然后添加优化版模型
            for name, model in base_models_opt:
                if model is not None and not any(excluded in name for excluded in ['投票模型', '注意力投票模型']):
                    # 添加"(优化)"标记，以便识别
                    optimized_name = f"{name}(优化)"
                    all_models_for_attention.append((optimized_name, model))
            
            print(f"收集了 {len(all_models_for_attention)} 个基础模型版本用于优化版注意力投票模型")
            
            attention_voting_model_opt, attention_opt_metrics, attention_opt_time = train_attention_voting_model(
                all_models_for_attention, X_train, y_train, X_val, y_val, class_names, 
                f1_threshold=0.85, optimize=True
            )
            if attention_voting_model_opt is not None:
                models_dict_optimized['注意力投票模型(优化)'] = attention_voting_model_opt
                training_times["注意力投票模型(优化)"] = attention_opt_time
    else:
        print("优化投票模型所需的基础模型不足，跳过创建投票模型")

    # 评估优化模型
    print("\n评估优化模型在测试集上的表现...")
    if models_dict_optimized:
        results_df_optimized = evaluate_all_models(models_dict_optimized, X_test, y_test)
        
        # 显示优化结果
        print("\n优化模式模型性能比较:")
        print(results_df_optimized)
    else:
        print("没有可评估的优化模型")
        results_df_optimized = pd.DataFrame()
    
    # 合并结果用于最终比较
    try:
        # 方法1：确保标准模型和优化模型都存在结果
        if hasattr(locals(), 'results_df_standard') and hasattr(locals(), 'results_df_optimized'):
            combined_results = pd.concat([results_df_standard, results_df_optimized])
            combined_results = combined_results.sort_values('F1分数', ascending=False).reset_index(drop=True)
            
            print("\n所有模型性能比较(标准模式和优化模式):")
            print(combined_results)
            
            # 保存结果
            combined_results.to_csv("./models_comparison_all.csv", index=False)
            print("所有模型比较结果已保存到 models_comparison_all.csv")
            
            # 可视化比较
            plot_models_comparison(combined_results)
            print("模型比较图已保存到 ./plots/models_f1_comparison.png")
        # 方法2：合并所有模型到一个字典中
        else:
            # 如果优化模式启用，合并所有模型结果
            if optimize:
                # 尝试合并所有模型到一个字典
                all_models = {}
                
                # 添加标准模型
                if 'models_dict_standard' in locals():
                    for name, model in models_dict_standard.items():
                        all_models[name] = model
                else:
                    # 添加单独训练的标准模型
                    standard_models = {
                        '逻辑回归': lr_model,
                        'SVM': svm_model,
                        '随机森林': rf_model,
                        'XGBoost': xgb_model,
                        'LightGBM': lgb_model,
                        'MLP': mlp_model
                    }
                    for name, model in standard_models.items():
                        if model is not None:
                            all_models[name] = model
                
                # 添加优化模型
                if 'models_dict_optimized' in locals():
                    for name, model in models_dict_optimized.items():
                        all_models[name] = model
                else:
                    # 添加单独训练的优化模型
                    optimized_models = {
                        '逻辑回归(优化)': lr_model_opt, 
                        'SVM(优化)': svm_model_opt,
                        '随机森林(优化)': rf_model_opt,
                        'XGBoost(优化)': xgb_model_opt,
                        'LightGBM(优化)': lgb_model_opt,
                        'MLP(优化)': mlp_model_opt
                    }
                    for name, model in optimized_models.items():
                        if model is not None:
                            all_models[name] = model
                
                # 评估所有模型
                print("\n评估所有模型(标准和优化):")
                combined_results_df = evaluate_all_models(all_models, X_test, y_test)
                
                print("\n所有模型性能比较:")
                print(combined_results_df)
                
                # 保存结果
                combined_results_df.to_csv("./models_comparison_all.csv", index=False)
                print("所有模型比较结果已保存到 models_comparison_all.csv")
                
                # 可视化比较
                plot_models_comparison(combined_results_df)
                print("模型比较图已保存到 ./plots/models_f1_comparison.png")
            else:
                # 如果只有一个结果数据框，直接使用
                if 'results_df' in locals() and not results_df.empty:
                    # 保存结果
                    results_df.to_csv("./models_comparison.csv", index=False)
                    print("模型比较结果已保存到 models_comparison.csv")
                    
                    # 可视化比较
                    plot_models_comparison(results_df)
                    print("模型比较图已保存到 ./plots/models_f1_comparison.png")
    except Exception as e:
        print(f"合并结果时出错: {e}")
        traceback.print_exc()
    
    # 显示训练时间比较
    print("\n可视化模型训练时间比较...")
    if training_times:
        plot_training_time_comparison(training_times)
    else:
        print("没有收集到训练时间数据")
    
    print("\n所有模型训练与评估完成!")

if __name__ == "__main__":
    main() 