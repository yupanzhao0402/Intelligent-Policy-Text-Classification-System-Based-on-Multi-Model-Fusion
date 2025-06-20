#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import csv
import joblib
import torch
from tqdm import tqdm
import time
from transformers import BertModel, BertTokenizer
import scipy.special

print('启动预测脚本，应用新规则...')

# 添加父目录到路径，以便导入模块
sys.path.append('..')

# 检查CUDA可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    print("使用CPU")

# 全局变量，将在加载类别名称时更新
LABEL_NAMES = {}

# 添加新规则的阈值
CONFIDENCE_THRESHOLD = 0.25  # 最低置信度阈值
SIMILARITY_THRESHOLD = 0.1   # 最高概率的10%偏差范围

print(f"规则1: 最低置信度阈值 = {CONFIDENCE_THRESHOLD:.6f} ({CONFIDENCE_THRESHOLD})")
print(f"规则2: 最高概率偏差范围 = {SIMILARITY_THRESHOLD:.6f} (10%)")
print(f"规则3: 使用中文标签名称")

class SimpleFeatureExtractor:
    """简化的特征提取器，直接使用BERT模型提取特征"""
    
    def __init__(self, model_name='bert-base-chinese', device='cpu'):
        """
        初始化特征提取器
        
        参数:
        model_name (str): BERT模型名称
        device (str): 设备
        """
        self.model_name = model_name
        self.device = device
        
        print(f"加载BERT模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        print("BERT模型加载完成")
    
    def extract_features_from_text(self, text):
        """
        从单个文本中提取特征
        
        参数:
        text (str): 输入文本
        
        返回:
        np.ndarray: 特征向量
        """
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

def load_all_models(models_dir):
    """
    加载所有模型
    
    参数:
    models_dir (str): 模型目录
    
    返回:
    dict: 模型字典，键为模型名称，值为模型对象
    """
    print("\n加载所有模型...")
    models_dict = {}
    
    # 遍历模型目录中的所有文件
    for filename in os.listdir(models_dir):
        if filename.endswith(".pkl"):
            model_path = os.path.join(models_dir, filename)
            model_name = os.path.splitext(filename)[0]
            
            try:
                print(f"加载模型: {model_name}")
                model = joblib.load(model_path)
                models_dict[model_name] = model
                print(f"模型 {model_name} 加载成功")
            except Exception as e:
                print(f"加载模型 {model_name} 失败: {e}")
    
    return models_dict

def extract_features_from_texts(texts, feature_extractor):
    """
    从多个文本中提取特征
    
    参数:
    texts (list): 文本列表
    feature_extractor (SimpleFeatureExtractor): 特征提取器
    
    返回:
    np.ndarray: 特征矩阵
    """
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

def predict_with_models(models_dict, X, class_names):
    """
    使用所有模型进行预测，使用新规则
    
    参数:
    models_dict (dict): 模型字典
    X (np.ndarray): 特征
    class_names (list): 类别名称
    
    返回:
    dict: 预测结果字典，键为模型名称，值为预测标签
    dict: 概率字典，键为模型名称，值为预测概率数组
    """
    print("\n使用所有模型进行预测...")
    predictions = {}
    probabilities = {}
    
    for model_name, model in models_dict.items():
        print(f"使用模型 {model_name} 进行预测...")
        try:
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # 尝试使用决策函数获取得分
                try:
                    decision_scores = model.decision_function(X)
                    # 应用softmax将得分转换为概率
                    proba = scipy.special.softmax(decision_scores, axis=1)
                except:
                    # 如果没有决策函数，直接预测标签
                    preds = model.predict(X)
                    # 创建一个伪概率数组
                    proba = np.zeros((len(preds), len(class_names)))
                    for i, pred in enumerate(preds):
                        proba[i, int(pred)] = 1.0

            # 保存预测概率
            probabilities[model_name] = proba
            
            # 应用新规则进行预测
            pred_labels = []
            for i, prob_row in enumerate(proba):
                # 找出最高概率及其索引
                max_prob = np.max(prob_row)
                max_idx = np.argmax(prob_row)
                
                # 输出一些调试信息（仅前几个样本）
                if i < 3:
                    print(f"样本 {i+1} 最高概率: {max_prob:.4f}, 类别: {LABEL_NAMES[max_idx]}")
                
                # 规则1: 如果最高概率小于阈值，标记为未知政策
                if max_prob < CONFIDENCE_THRESHOLD:
                    pred_labels.append("未知政策")
                    if i < 3:
                        print(f"样本 {i+1} 结果: 未知政策 (低于阈值)")
                else:
                    # 规则2: 找出在最高概率10%偏差范围内的所有标签
                    threshold = max_prob - (max_prob * SIMILARITY_THRESHOLD)
                    similar_indices = np.where(prob_row >= threshold)[0]
                    
                    # 按概率降序排序
                    similar_indices = similar_indices[np.argsort(-prob_row[similar_indices])]
                    
                    # 转换为中文标签并用|连接
                    similar_labels = [LABEL_NAMES[int(idx)] for idx in similar_indices]
                    pred_label = "|".join(similar_labels)
                    pred_labels.append(pred_label)
                    
                    if i < 3:
                        print(f"样本 {i+1} 结果: {pred_label}")
            
            predictions[model_name] = pred_labels
            print(f"模型 {model_name} 预测成功")
            
        except Exception as e:
            print(f"模型 {model_name} 预测失败: {e}")
    
    return predictions, probabilities

def load_excel(file_path):
    """
    使用openpyxl加载Excel文件
    
    参数:
    file_path (str): Excel文件路径
    
    返回:
    list of dict: 数据列表
    """
    import openpyxl
    
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

def save_csv(data, headers, file_path):
    """
    保存数据为CSV文件
    
    参数:
    data (list of dict): 数据列表
    headers (list): 表头列表
    file_path (str): 保存路径
    """
    print(f"保存CSV文件: {file_path}")
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print(f"CSV文件保存成功: {file_path}")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

def main():
    """主函数"""
    # 创建结果目录
    os.makedirs("result", exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 加载测试数据
    test_file = "文本分类.xlsx"
    test_data = load_excel(test_file)
    
    if not test_data:
        print("错误: 未能加载测试数据")
        return
    
    # 检查数据
    if 'text' not in test_data[0]:
        print("错误: 测试数据中没有'text'列")
        return
    
    print(f"测试数据加载成功，共 {len(test_data)} 条记录")
    
    # 2. 初始化特征提取器
    print("\n初始化特征提取器...")
    feature_extractor = SimpleFeatureExtractor(model_name='bert-base-chinese', device=device)
    
    # 3. 加载所有模型
    models_dir = "../output_models"
    models_dict = load_all_models(models_dir)
    
    if not models_dict:
        print("错误: 没有加载到任何模型")
        return
    
    # 4. 加载类别名称
    print("\n加载类别名称...")
    global LABEL_NAMES
    try:
        class_names = joblib.load("../extracted_features/class_names.joblib")
        print(f"类别名称加载成功: {class_names}")
        
        # 更新标签名称映射
        LABEL_NAMES = {i: name for i, name in enumerate(class_names)}
        print("标签映射关系已更新:")
        for idx, name in LABEL_NAMES.items():
            print(f"  {idx}: {name}")
    except Exception as e:
        print(f"加载类别名称失败: {e}")
        return
    
    # 5. 从文本中提取特征
    texts = [item['text'] for item in test_data]
    X_test = extract_features_from_texts(texts, feature_extractor)
    
    # 6. 使用所有模型进行预测
    predictions, probabilities = predict_with_models(models_dict, X_test, class_names)
    
    # 7. 只保存注意力投票模型的预测结果
    print("\n保存注意力投票模型的预测结果...")
    
    if 'attention_voting_model' in predictions:
        attention_result_data = []
        for i, item in enumerate(test_data):
            # 只保留text和预测结果
            attention_result_data.append({
                'text': item['text'],
                '预测标签': predictions['attention_voting_model'][i]
            })
        
        # 构建表头
        attention_headers = ['text', '预测标签']
        
        attention_result_path = "result/res_attention_vote_new.csv"
        save_csv(attention_result_data, attention_headers, attention_result_path)
        
        print(f"注意力投票模型结果已保存到 {attention_result_path}")
    else:
        print("警告: 未找到注意力投票模型的预测结果")
    
    # 打印规则信息
    print("\n规则设置:")
    print(f"规则1: 最低置信度阈值 = {CONFIDENCE_THRESHOLD:.6f} ({CONFIDENCE_THRESHOLD})")
    print(f"规则2: 最高概率偏差范围 = {SIMILARITY_THRESHOLD:.6f} (10%)")
    print(f"规则3: 使用中文标签名称")
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time / 60)
    seconds = int(total_time % 60)
    print(f"\n预测完成，总耗时: {minutes}分{seconds}秒")

if __name__ == "__main__":
    main() 