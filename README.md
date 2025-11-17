# 葡萄酒品质分析系统

基于Python的葡萄酒识别数据集分析系统，用于分析影响红酒品质的关键特征。

## 功能特性

1. **数据获取**: 使用sklearn内置的Wine recognition dataset
2. **数据预处理**: 自动检测和处理缺失值、异常值
3. **科学计算**: 使用Pandas进行数据分析和特征重要性计算
4. **可视化展示**: 
   - 酒精浓度分布分析
   - 密度与酒精浓度关系散点图
   - 酸性特征分析
   - 甜度相关特征分析
   - 特征相关性热力图
   - 特征重要性排序
   - 类别分布图

## 环境要求

- Python 3.7+
- 依赖包见 `requirements.txt`

## 安装步骤

1. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

直接运行主程序：
```bash
python wine_analysis.py
```

## 输出结果

运行后会在 `output/` 目录下生成：
- `wine_analysis_overview.png`: 综合分析图表
- `feature_importance.png`: 特征重要性排序图
- `class_distribution.png`: 类别分布图
- `feature_importance.csv`: 特征重要性数据表

## 数据集说明

使用sklearn的Wine recognition dataset，包含：
- 178个样本
- 13个特征（酒精浓度、密度、酸性、甜度等）
- 3个类别（Class_0, Class_1, Class_2）

## 主要分析内容

1. **酒精浓度分析**: 分析酒精浓度分布及其与其他特征的关系
2. **密度关系**: 探索密度与酒精浓度的相关性
3. **酸性特征**: 分析挥发性酸度、总二氧化硫等酸性指标
4. **甜度特征**: 分析残糖量对品质的影响
5. **特征重要性**: 识别影响红酒品质最重要的特征

## 开发工具

- PyCharm (推荐)
- Python 3.7+

## 注意事项

- 首次运行会自动创建 `output/` 目录
- 图表支持中文显示（需要系统安装中文字体）
- 如果中文显示异常，可修改代码中的字体设置




