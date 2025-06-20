<!---
Copyright 2023 哈工大赵玉攀课题组. All rights reserved.

Licensed under the MIT License.
-->

<p align="center">
  <img src="https://img.shields.io/badge/政策文本-智能分类系统-blue" alt="政策文本智能分类系统" width="400"/>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion.svg?color=blue"></a>
    <a href="https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion/releases"><img alt="Release" src="https://img.shields.io/badge/release-v1.0-brightgreen"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.8-orange"></a>
    <a href="https://python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue"></a>
</p>

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion/blob/main/README_en.md">English</a>
    </p>
</h4>

<h3 align="center">
    <p>基于多模型融合的政策文本智能分类系统</p>
    <p><i>哈工大赵玉攀课题组</i></p>
</h3>

## 📋 项目概述

本项目是一个基于BERT特征提取和多模型融合的政策文本智能分类系统。系统能够自动对政府政策文件进行分类，支持多种机器学习和深度学习模型，包括逻辑回归、支持向量机、随机森林、XGBoost、LightGBM、多层感知机、预训练Transformer和基于注意力机制的投票模型等。

## ✨ 功能特点

- **多模型支持**：集成了多种主流机器学习和深度学习模型
- **投票集成机制**：包含普通投票和基于注意力机制的加权投票模型
- **预训练Transformer**：利用BERT提取的特征训练自定义Transformer模型
- **模型优化**：支持超参数优化和交叉验证
- **可视化分析**：提供混淆矩阵、精确率-召回率曲线、特征重要性等多种可视化结果
- **预测功能**：支持对新政策文本的分类预测
- **GPU加速**：支持使用GPU加速模型训练和推理

## 🔍 项目结构

```
.
├── models/                   # 模型实现代码
│   ├── attention_voting_model.py   # 注意力投票模型
│   ├── lightgbm_model.py           # LightGBM模型
│   ├── logistic_regression_model.py # 逻辑回归模型
│   ├── mlp_model.py                # 多层感知机模型
│   ├── pretrained_transformer.py   # 预训练Transformer模型
│   ├── random_forest_model.py      # 随机森林模型
│   ├── svm_model.py                # SVM模型
│   ├── voting_model.py             # 投票模型
│   └── xgboost_model.py            # XGBoost模型
├── extracted_features/       # 预提取的BERT特征
│   ├── class_names.joblib    # 类别名称
│   ├── X_train.npy           # 训练集特征
│   ├── X_val.npy             # 验证集特征
│   ├── X_test.npy            # 测试集特征
│   ├── y_train.npy           # 训练集标签
│   ├── y_val.npy             # 验证集标签
│   └── y_test.npy            # 测试集标签
├── plots/                    # 整体模型可视化结果
│   ├── models_f1_comparison.png    # 模型F1分数比较
│   ├── model_training_time_comparison.png  # 模型训练时间比较
│   └── mlp_confusion_matrix.png    # MLP混淆矩阵
├── results/                  # 不同模型的详细结果
├── test/                     # 测试和预测功能
│   ├── predict.py            # 预测脚本
│   └── predict_new.py        # 新数据预测脚本
├── output_models/            # 训练好的模型(由于过大请自行运行，项目具有可复现性)
├── custom_split.py           # 定制化训练集划分
├── install_dependencies.py   # 依赖安装脚本
├── requirements.txt          # 依赖库
├── data_processor.py         # 数据处理模块
├── feature_extractor.py      # 特征提取模块
└── main.py                   # 主程序
```

## 🔄 数据处理流程

1. **数据获取与预处理**：
   - 通过NLP工具对政府政策文件进行分句
   - 进行数据标注、校准和平衡处理
   - 通过多种文本增强技术扩充数据集

2. **特征提取**：
   - 使用BERT模型提取文本的向量表示
   - 将特征存储为numpy数组，便于后续处理

3. **模型训练**：
   - 训练多种不同类型的机器学习和深度学习模型
   - 采用交叉验证和超参数优化提升模型性能
   - 训练投票机制和注意力机制的集成模型

4. **模型评估与可视化**：
   - 生成混淆矩阵、精确率-召回率曲线等评估可视化
   - 比较不同模型的性能指标
   - 分析特征重要性和模型预测依据

## 🚀 安装与使用

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)
- 详细依赖见 `requirements.txt`

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion.git
   cd Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion
   ```

2. 安装依赖
   ```bash
   python install_dependencies.py
   # 或者
   pip install -r requirements.txt
   ```

3. 运行主程序
   ```bash
   python main.py
   ```

### 使用流程

1. **提取特征**：
   ```python
   python feature_extractor.py --input your_data.csv --output ./extracted_features
   ```

2. **训练模型**：
   ```python
   python main.py
   ```

3. **预测新文本**：
   ```python
   python test/predict.py --model attention_voting_model --input your_text_file.xlsx
   ```

4. **使用特定模型预测**：
   ```python
   python test/predict_new.py --model pretrained_transformer --input your_text_file.xlsx
   ```

## 📊 模型性能

根据最新运行结果，各模型性能如下：

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|--------|
| 注意力投票模型(优化) | 0.9386 | 0.9389 | 0.9386 | 0.9385 |
| 预训练Transformer | 0.9367 | 0.9373 | 0.9367 | 0.9365 |
| MLP(优化) | 0.9352 | 0.9359 | 0.9352 | 0.9353 |
| 注意力投票模型 | 0.9338 | 0.9344 | 0.9338 | 0.9338 |
| 投票模型(soft) | 0.9319 | 0.9332 | 0.9319 | 0.9320 |
| MLP | 0.9319 | 0.9326 | 0.9319 | 0.9318 |
| 逻辑回归 | 0.9119 | 0.9130 | 0.9119 | 0.9120 |
| SVM | 0.9095 | 0.9099 | 0.9095 | 0.9092 |
| LightGBM | 0.8957 | 0.8984 | 0.8957 | 0.8964 |
| XGBoost | 0.8881 | 0.8898 | 0.8881 | 0.8886 |
| 随机森林 | 0.8081 | 0.8090 | 0.8081 | 0.8051 |

## 🔧 预训练Transformer模型

本项目实现了一个基于预提取BERT特征的自定义Transformer模型：

- **模型架构**：多头自注意力机制 + 前馈神经网络
- **输入**：BERT预提取的768维特征向量
- **性能**：在测试集上达到93.65%的F1分数
- **优势**：
  - 相比传统机器学习模型有显著性能提升
  - 比完整的BERT微调更高效
  - 与其他模型兼容，易于集成

```python
# 使用预训练Transformer模型
from models.pretrained_transformer import PretrainedTransformerModel

# 初始化模型
model = PretrainedTransformerModel(
    input_dim=768,  # BERT特征维度
    num_classes=len(class_names),
    d_model=768,    # Transformer隐藏层维度
    nhead=8,        # 注意力头数
    num_layers=4,   # Transformer层数
    device='cuda'   # 使用GPU加速
)

# 训练模型
model.train(X_train, y_train, X_val, y_val)

# 预测
predictions = model.predict(X_test)
```

## ⚠️ 注意事项

- 首次运行时会自动下载BERT模型，请确保网络连接正常
- 大型模型(如Transformer和注意力投票模型)训练时建议使用GPU
- 对于大规模数据集，请确保有足够的内存和存储空间

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🤝 贡献

欢迎通过Issue和Pull Request形式贡献代码和提出建议。

## 📚 引用

如果您在研究中使用了本项目，请按以下格式引用：

```bibtex
@misc{policy-text-classification,
  author = {哈工大赵玉攀课题组},
  title = {基于多模型融合的政策文本智能分类系统},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion}
}
```

## 📧 联系方式

如有任何问题或建议，请联系：zhaoyplab@hit.edu.cn
