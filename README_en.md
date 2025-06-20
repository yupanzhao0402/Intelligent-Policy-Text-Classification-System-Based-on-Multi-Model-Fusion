<!---
Copyright 2023 HIT Zhao Yupan Research Group. All rights reserved.

Licensed under the MIT License.
-->

<p align="center">
  <img src="https://img.shields.io/badge/Policy%20Text-Intelligent%20Classification%20System-blue" alt="Policy Text Intelligent Classification System" width="400"/>
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
        <a href="https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion/blob/main/README.md">简体中文</a> |
        <b>English</b>
    </p>
</h4>

<h3 align="center">
    <p>Multi-Model Fusion Based Policy Text Intelligent Classification System</p>
    <p><i>Harbin Institute of Technology, Zhao Yupan Research Group</i></p>
</h3>

## 📋 Project Overview

This project is a policy text intelligent classification system based on BERT feature extraction and multi-model fusion. The system can automatically classify government policy documents, supporting various machine learning and deep learning models, including Logistic Regression, Support Vector Machine, Random Forest, XGBoost, LightGBM, Multi-Layer Perceptron, Pretrained Transformer, and attention-based voting models.

## ✨ Features

- **Multiple Model Support**: Integrates various mainstream machine learning and deep learning models
- **Voting Ensemble Mechanism**: Includes regular voting and attention-based weighted voting models
- **Pretrained Transformer**: Utilizes BERT-extracted features to train custom Transformer models
- **Model Optimization**: Supports hyperparameter optimization and cross-validation
- **Visualization Analysis**: Provides confusion matrices, precision-recall curves, feature importance, and other visualization results
- **Prediction Functionality**: Supports classification prediction for new policy texts
- **GPU Acceleration**: Supports GPU acceleration for model training and inference

## 🔍 Project Structure

```
.
├── models/                   # Model implementation code
│   ├── attention_voting_model.py   # Attention voting model
│   ├── lightgbm_model.py           # LightGBM model
│   ├── logistic_regression_model.py # Logistic regression model
│   ├── mlp_model.py                # Multi-layer perceptron model
│   ├── pretrained_transformer.py   # Pretrained transformer model
│   ├── random_forest_model.py      # Random forest model
│   ├── svm_model.py                # SVM model
│   ├── voting_model.py             # Voting model
│   └── xgboost_model.py            # XGBoost model
├── extracted_features/       # Pre-extracted BERT features
│   ├── class_names.joblib    # Class names
│   ├── X_train.npy           # Training set features
│   ├── X_val.npy             # Validation set features
│   ├── X_test.npy            # Test set features
│   ├── y_train.npy           # Training set labels
│   ├── y_val.npy             # Validation set labels
│   └── y_test.npy            # Test set labels
├── plots/                    # Overall model visualization results
│   ├── models_f1_comparison.png    # Model F1 score comparison
│   ├── model_training_time_comparison.png  # Model training time comparison
│   └── mlp_confusion_matrix.png    # MLP confusion matrix
├── results/                  # Detailed results for different models
├── test/                     # Testing and prediction functionality
│   ├── predict.py            # Prediction script
│   └── predict_new.py        # New data prediction script
├── output_models/            # Trained models
├── install_dependencies.py   # Dependency installation script
├── requirements.txt          # Dependencies
├── data_processor.py         # Data processing module
├── feature_extractor.py      # Feature extraction module
└── main.py                   # Main program
```

## 🔄 Data Processing Flow

1. **Data Acquisition and Preprocessing**:
   - Sentence segmentation of government policy documents using NLP tools
   - Data annotation, calibration, and balancing
   - Data set expansion through various text augmentation techniques

2. **Feature Extraction**:
   - Using BERT model to extract vector representations of text
   - Storing features as numpy arrays for subsequent processing

3. **Model Training**:
   - Training various types of machine learning and deep learning models
   - Using cross-validation and hyperparameter optimization to improve model performance
   - Training ensemble models with voting mechanisms and attention mechanisms

4. **Model Evaluation and Visualization**:
   - Generating evaluation visualizations such as confusion matrices and precision-recall curves
   - Comparing performance metrics of different models
   - Analyzing feature importance and model prediction basis

## 🚀 Installation and Usage

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for detailed dependencies

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion.git
   cd Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion
   ```

2. Install dependencies
   ```bash
   python install_dependencies.py
   # or
   pip install -r requirements.txt
   ```

3. Run the main program
   ```bash
   python main.py
   ```

### Usage Process

1. **Extract Features**:
   ```python
   python feature_extractor.py --input your_data.csv --output ./extracted_features
   ```

2. **Train Models**:
   ```python
   python main.py
   ```

3. **Predict New Text**:
   ```python
   python test/predict.py --model attention_voting_model --input your_text_file.xlsx
   ```

4. **Use Specific Model for Prediction**:
   ```python
   python test/predict_new.py --model pretrained_transformer --input your_text_file.xlsx
   ```

## 📊 Model Performance

According to the latest running results, the performance of each model is as follows:

| Model | Accuracy | Precision | Recall | F1 Score |
|------|--------|--------|--------|--------|
| Attention Voting Model (Optimized) | 0.9386 | 0.9389 | 0.9386 | 0.9385 |
| Pretrained Transformer | 0.9367 | 0.9373 | 0.9367 | 0.9365 |
| MLP (Optimized) | 0.9352 | 0.9359 | 0.9352 | 0.9353 |
| Attention Voting Model | 0.9338 | 0.9344 | 0.9338 | 0.9338 |
| Voting Model (Soft) | 0.9319 | 0.9332 | 0.9319 | 0.9320 |
| MLP | 0.9319 | 0.9326 | 0.9319 | 0.9318 |
| Logistic Regression | 0.9119 | 0.9130 | 0.9119 | 0.9120 |
| SVM | 0.9095 | 0.9099 | 0.9095 | 0.9092 |
| LightGBM | 0.8957 | 0.8984 | 0.8957 | 0.8964 |
| XGBoost | 0.8881 | 0.8898 | 0.8881 | 0.8886 |
| Random Forest | 0.8081 | 0.8090 | 0.8081 | 0.8051 |

## 🔧 Pretrained Transformer Model

This project implements a custom Transformer model based on pre-extracted BERT features:

- **Model Architecture**: Multi-head self-attention mechanism + Feed-forward neural network
- **Input**: 768-dimensional feature vectors pre-extracted by BERT
- **Performance**: Achieves 93.65% F1 score on the test set
- **Advantages**:
  - Significant performance improvement compared to traditional machine learning models
  - More efficient than full BERT fine-tuning
  - Compatible with other models, easy to integrate

```python
# Using the pretrained Transformer model
from models.pretrained_transformer import PretrainedTransformerModel

# Initialize the model
model = PretrainedTransformerModel(
    input_dim=768,  # BERT feature dimension
    num_classes=len(class_names),
    d_model=768,    # Transformer hidden layer dimension
    nhead=8,        # Number of attention heads
    num_layers=4,   # Number of Transformer layers
    device='cuda'   # Use GPU acceleration
)

# Train the model
model.train(X_train, y_train, X_val, y_val)

# Predict
predictions = model.predict(X_test)
```

## ⚠️ Notes

- BERT model will be automatically downloaded when running for the first time, please ensure normal network connection
- GPU is recommended for training large models (such as Transformer and attention voting models)
- For large-scale datasets, please ensure sufficient memory and storage space

## 📄 License

This project is licensed under the MIT License, see the LICENSE file for details.

## 🤝 Contribution

Contributions through Issues and Pull Requests are welcome.

## 📚 Citation

If you use this project in your research, please cite it as follows:

```bibtex
@misc{policy-text-classification,
  author = {HIT Zhao Yupan Research Group},
  title = {Multi-Model Fusion Based Policy Text Intelligent Classification System},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yupanzhao0402/Intelligent-Policy-Text-Classification-System-Based-on-Multi-Model-Fusion}
}
```

## 📧 Contact

For any questions or suggestions, please contact: 20240219@hit.edu.cn
