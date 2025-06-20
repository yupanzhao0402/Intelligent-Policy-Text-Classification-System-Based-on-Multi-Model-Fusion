#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 设置中文字体和负号显示
plt.rcParams["font.family"] = "SimHei"  # 设置默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 设置更好看的风格
plt.style.use('seaborn-v0_8-darkgrid')  # 使用seaborn的暗网格风格

# 设置更大的字号，便于阅读
plt.rcParams['font.size'] = 12  # 设置字体大小
plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 设置轴标签字体大小

# 设置更好的图表尺寸
plt.rcParams['figure.figsize'] = (10, 6)  # 设置默认图表尺寸
plt.rcParams['figure.dpi'] = 100  # 设置默认DPI

# 设置更好的图例
plt.rcParams['legend.frameon'] = True  # 图例边框
plt.rcParams['legend.fontsize'] = 10  # 图例字体大小

# 设置网格线
plt.rcParams['grid.alpha'] = 0.3  # 网格线透明度
plt.rcParams['grid.linestyle'] = '--'  # 网格线样式

# 导出配置函数，在需要单独配置的地方调用
def set_plt_configs():
    """重新设置matplotlib配置，用于在单独的模块中调用"""
    # 设置中文字体和负号显示
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    
    # 其他配置可以按需添加 