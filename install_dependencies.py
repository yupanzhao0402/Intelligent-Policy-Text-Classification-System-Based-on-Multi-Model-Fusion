#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
依赖安装脚本
用于安装项目所需的依赖项
"""

import subprocess
import sys
import os

def install_dependencies():
    """安装项目所需的依赖项"""
    print("开始安装依赖项...")
    
    # 检查requirements.txt文件是否存在
    if not os.path.exists("requirements.txt"):
        print("错误: 找不到requirements.txt文件")
        return False
    
    try:
        # 安装依赖项
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖项安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装依赖项时出错: {e}")
        return False

def install_pytorch():
    """安装PyTorch（可选）"""
    print("\n是否安装PyTorch CUDA版本? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("安装PyTorch CUDA版本...")
        try:
            # 安装PyTorch CUDA版本
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
            print("PyTorch CUDA版本安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"安装PyTorch CUDA版本时出错: {e}")
            return False
    else:
        print("跳过PyTorch CUDA版本安装")
        return True

def download_bert_model():
    """下载BERT模型（可选）"""
    print("\n是否下载BERT模型? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("下载BERT模型...")
        try:
            # 安装transformers
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            
            # 下载BERT模型
            import torch
            from transformers import BertModel, BertTokenizer
            
            print("下载BERT分词器...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            print("下载BERT模型...")
            model = BertModel.from_pretrained('bert-base-chinese')
            
            print("BERT模型下载成功")
            return True
        except Exception as e:
            print(f"下载BERT模型时出错: {e}")
            return False
    else:
        print("跳过BERT模型下载")
        return True

def main():
    """主函数"""
    print("政策文本智能分类系统 - 依赖安装脚本")
    print("=" * 50)
    
    # 安装依赖项
    if not install_dependencies():
        print("依赖项安装失败，请手动安装")
        return
    
    # 安装PyTorch
    install_pytorch()
    
    # 下载BERT模型
    download_bert_model()
    
    print("\n安装完成!")
    print("现在您可以运行 python main.py 来启动程序")

if __name__ == "__main__":
    main() 