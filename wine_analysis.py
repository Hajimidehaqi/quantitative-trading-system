"""
葡萄酒品质分析系统 - 最小可用版本
使用sklearn内置的Wine recognition dataset进行分析
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import matplotlib.font_manager as fm
import warnings
# 抑制所有字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning)
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体，自动检测系统可用字体"""
    # Windows常见中文字体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        # 设置matplotlib全局字体
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.family'] = 'sans-serif'
        # 设置seaborn的字体参数
        sns.set_style("whitegrid")
        try:
            sns.set(font=selected_font)
        except:
            pass
        print(f"使用字体: {selected_font}")
        return selected_font
    else:
        # 如果找不到中文字体，尝试使用默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("警告: 未找到中文字体，图表中的中文可能显示异常")
        return None
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    return selected_font

# 初始化字体设置
chinese_font_name = setup_chinese_font()

# 创建字体属性对象，用于所有绘图
if chinese_font_name:
    from matplotlib.font_manager import FontProperties
    chinese_font = FontProperties(fname=None, family=chinese_font_name)
else:
    chinese_font = None

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_wine_data():
    """加载葡萄酒数据集"""
    print("=" * 50)
    print("步骤1: 数据获取")
    print("=" * 50)
    
    wine_data = load_wine()
    
    # 转换为DataFrame
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    
    # 添加类别名称
    class_names = ['Class_0', 'Class_1', 'Class_2']
    df['class_name'] = df['target'].map(lambda x: class_names[x])
    
    print(f"数据集形状: {df.shape}")
    print(f"特征数量: {len(wine_data.feature_names)}")
    print(f"样本数量: {len(df)}")
    print(f"\n特征列表:")
    for i, feature in enumerate(wine_data.feature_names, 1):
        print(f"  {i}. {feature}")
    
    print(f"\n数据预览:")
    print(df.head())
    print(f"\n数据基本信息:")
    print(df.info())
    
    return df, wine_data.feature_names


def preprocess_data(df):
    """数据预处理"""
    print("\n" + "=" * 50)
    print("步骤2: 数据预处理")
    print("=" * 50)
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print(f"\n缺失值统计:")
    if missing_values.sum() == 0:
        print("  无缺失值")
    else:
        print(missing_values[missing_values > 0])
    
    # 检查异常值（使用IQR方法）
    print(f"\n异常值检测 (使用IQR方法):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    outliers_info = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            outliers_info.append({
                'feature': col,
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100
            })
    
    if outliers_info:
        outliers_df = pd.DataFrame(outliers_info)
        print(outliers_df.to_string(index=False))
    else:
        print("  未发现明显异常值")
    
    # 数据统计摘要
    print(f"\n数据统计摘要:")
    print(df.describe())
    
    return df


def analyze_and_visualize(df, feature_names):
    """科学计算与可视化"""
    print("\n" + "=" * 50)
    print("步骤3: 科学计算与可视化")
    print("=" * 50)
    
    # 创建输出目录
    import os
    os.makedirs('output', exist_ok=True)
    
    # 1. 分析酒精浓度与其他特征的关系
    print("\n1. 分析酒精浓度与其他特征的关系...")
    alcohol_col = 'alcohol'
    
    # 计算相关性
    correlations = df[feature_names].corr()[alcohol_col].sort_values(ascending=False)
    print(f"\n与酒精浓度的相关性排序:")
    for feature, corr in correlations.items():
        if feature != alcohol_col:
            print(f"  {feature}: {corr:.3f}")
    
    # 2. 可视化：酒精浓度分布
    plt.figure(figsize=(15, 10))
    
    # 子图1: 酒精浓度分布直方图
    plt.subplot(2, 3, 1)
    plt.hist(df[alcohol_col], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('酒精浓度 (Alcohol)', fontproperties=chinese_font)
    plt.ylabel('频数', fontproperties=chinese_font)
    plt.title('酒精浓度分布', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 子图2: 酒精浓度与脯氨酸的散点图（脯氨酸与酒精相关性最高）
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(df['alcohol'], df['proline'], 
                         c=df['target'], cmap='viridis', alpha=0.6)
    plt.xlabel('酒精浓度 (Alcohol)', fontproperties=chinese_font)
    plt.ylabel('脯氨酸 (Proline)', fontproperties=chinese_font)
    plt.title('酒精浓度 vs 脯氨酸', fontproperties=chinese_font)
    cbar = plt.colorbar(scatter)
    cbar.set_label('类别', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 子图3: 不同类别的酒精浓度箱线图
    plt.subplot(2, 3, 3)
    ax = plt.gca()
    df.boxplot(column=alcohol_col, by='class_name', ax=ax)
    plt.xlabel('类别', fontproperties=chinese_font)
    plt.ylabel('酒精浓度', fontproperties=chinese_font)
    plt.title('不同类别的酒精浓度分布', fontproperties=chinese_font)
    plt.suptitle('')  # 移除默认标题
    # 设置x轴刻度标签字体
    if chinese_font:
        for label in ax.get_xticklabels():
            label.set_fontproperties(chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 子图4: 酸性特征分析（苹果酸等）
    plt.subplot(2, 3, 4)
    acid_features = ['malic_acid', 'alcalinity_of_ash', 'ash']
    acid_data = df[acid_features].mean()
    plt.bar(range(len(acid_data)), acid_data.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.xticks(range(len(acid_data)), ['苹果酸', '灰分碱度', '灰分'], 
               rotation=45, ha='right', fontproperties=chinese_font)
    plt.ylabel('平均值', fontproperties=chinese_font)
    plt.title('酸性相关特征分析', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 子图5: 颜色强度分析（影响红酒外观和品质）
    plt.subplot(2, 3, 5)
    color_intensity = df['color_intensity']
    plt.hist(color_intensity, bins=20, color='#FFA07A', edgecolor='black', alpha=0.7)
    plt.xlabel('颜色强度 (Color Intensity)', fontproperties=chinese_font)
    plt.ylabel('频数', fontproperties=chinese_font)
    plt.title('颜色强度分布', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 子图6: 特征相关性热力图
    plt.subplot(2, 3, 6)
    # 选择主要特征进行相关性分析
    main_features = ['alcohol', 'proline', 'color_intensity', 
                    'total_phenols', 'flavanoids', 'malic_acid']
    corr_matrix = df[main_features].corr()
    # 设置字体参数，确保中文正常显示
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                xticklabels=main_features, yticklabels=main_features)
    plt.title('主要特征相关性热力图', fontproperties=chinese_font)
    
    plt.tight_layout()
    plt.savefig('output/wine_analysis_overview.png', dpi=300, bbox_inches='tight')
    print("  已保存: output/wine_analysis_overview.png")
    # 显示交互式窗口
    fig1 = plt.gcf()
    try:
        fig1.canvas.manager.set_window_title('葡萄酒分析 - 综合分析图表')
    except:
        pass
    # 显示第一个图表（阻塞模式，关闭后继续）
    plt.draw()  # 强制绘制
    print("\n【窗口1/3】正在显示综合分析图表...")
    print("提示：关闭此窗口后将显示下一个图表")
    plt.show(block=True)  # 阻塞模式，等待用户关闭窗口
    
    # 3. 特征重要性分析（使用方差和相关性）
    print("\n2. 特征重要性分析...")
    
    # 计算每个特征与目标的相关性
    feature_importance = []
    for feature in feature_names:
        corr = abs(df[feature].corr(df['target']))
        std = df[feature].std()
        # 综合评分：相关性 * 标准差（归一化）
        score = corr * (std / df[feature].mean())
        feature_importance.append({
            'feature': feature,
            'correlation': corr,
            'std': std,
            'importance_score': score
        })
    
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    
    print("\n特征重要性排序（基于与目标的相关性和变异性）:")
    print(importance_df.to_string(index=False))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['importance_score'].values, 
             color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('重要性得分', fontproperties=chinese_font)
    plt.title('Top 10 特征重要性分析', fontproperties=chinese_font)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
    print("  已保存: output/feature_importance.png")
    # 显示交互式窗口
    fig2 = plt.gcf()
    try:
        fig2.canvas.manager.set_window_title('葡萄酒分析 - 特征重要性')
    except:
        pass
    # 显示第二个图表（阻塞模式，关闭后继续）
    plt.draw()  # 强制绘制
    print("\n【窗口2/3】正在显示特征重要性图表...")
    print("提示：关闭此窗口后将显示下一个图表")
    plt.show(block=True)  # 阻塞模式，等待用户关闭窗口
    
    # 4. 类别分布饼图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    class_counts = df['class_name'].value_counts()
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90,
            textprops={'fontproperties': chinese_font})
    plt.title('葡萄酒类别分布', fontproperties=chinese_font)
    
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.bar(class_counts.index, class_counts.values, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.xlabel('类别', fontproperties=chinese_font)
    plt.ylabel('数量', fontproperties=chinese_font)
    plt.title('葡萄酒类别数量', fontproperties=chinese_font)
    # 设置x轴刻度标签字体
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=chinese_font)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('output/class_distribution.png', dpi=300, bbox_inches='tight')
    print("  已保存: output/class_distribution.png")
    # 显示交互式窗口
    fig3 = plt.gcf()
    try:
        fig3.canvas.manager.set_window_title('葡萄酒分析 - 类别分布')
    except:
        pass
    # 显示第三个图表（阻塞模式，关闭后继续）
    plt.draw()  # 强制绘制
    print("\n【窗口3/3】正在显示类别分布图表...")
    print("提示：关闭此窗口后程序将完成")
    plt.show(block=True)  # 阻塞模式，等待用户关闭窗口
    
    return importance_df, [fig1, fig2, fig3]


def result_analysis(df, importance_df):
    """结果分析"""
    print("\n" + "=" * 50)
    print("步骤4: 结果分析")
    print("=" * 50)
    
    print("\n【关键发现】")
    print("-" * 50)
    
    # 1. 最重要的特征
    top_3_features = importance_df.head(3)
    print("\n1. 影响红酒品质最重要的3个特征:")
    for idx, row in top_3_features.iterrows():
        print(f"   {row['feature']}: 重要性得分 {row['importance_score']:.4f}")
    
    # 2. 酒精浓度分析
    print("\n2. 酒精浓度分析:")
    print(f"   平均酒精浓度: {df['alcohol'].mean():.2f}")
    print(f"   酒精浓度范围: {df['alcohol'].min():.2f} - {df['alcohol'].max():.2f}")
    alcohol_corr = df['alcohol'].corr(df['target'])
    print(f"   与品质类别的相关性: {alcohol_corr:.3f}")
    
    # 3. 脯氨酸与酒精的关系
    print("\n3. 脯氨酸与酒精浓度的关系:")
    proline_alcohol_corr = df['proline'].corr(df['alcohol'])
    print(f"   脯氨酸与酒精浓度的相关性: {proline_alcohol_corr:.3f}")
    if abs(proline_alcohol_corr) > 0.5:
        print("   → 脯氨酸与酒精浓度存在较强的相关性")
    
    # 4. 酸性特征
    print("\n4. 酸性特征分析:")
    acid_features = ['malic_acid', 'alcalinity_of_ash', 'ash']
    for feature in acid_features:
        if feature in df.columns:
            corr = abs(df[feature].corr(df['target']))
            feature_name = {'malic_acid': '苹果酸', 'alcalinity_of_ash': '灰分碱度', 'ash': '灰分'}.get(feature, feature)
            print(f"   {feature_name} ({feature}): 与目标相关性 {corr:.3f}")
    
    # 5. 颜色强度特征
    print("\n5. 颜色强度特征分析:")
    color_corr = abs(df['color_intensity'].corr(df['target']))
    print(f"   颜色强度与品质类别的相关性: {color_corr:.3f}")
    print(f"   平均颜色强度: {df['color_intensity'].mean():.2f}")
    
    # 6. 结论
    print("\n" + "=" * 50)
    print("【结论】")
    print("=" * 50)
    print("""
基于数据分析，影响红酒品质最重要的特征包括：

1. 酒精浓度 (Alcohol): 是区分不同品质红酒的关键指标之一
2. 脯氨酸 (Proline): 与酒精浓度高度相关，是重要的品质指标
3. 颜色强度 (Color Intensity): 影响红酒外观和品质特征
4. 苹果酸 (Malic Acid): 酸性特征对红酒品质有重要影响
5. 总酚类 (Total Phenols) 和类黄酮 (Flavanoids): 影响红酒的口感和品质

建议：
- 在评估红酒品质时，应重点关注酒精浓度和脯氨酸指标
- 酸性特征（特别是苹果酸）需要控制在合理范围内
- 颜色强度是区分不同类别红酒的重要视觉指标
- 不同类别的红酒在这些关键特征上存在明显差异
    """)
    
    # 保存分析结果到CSV
    importance_df.to_csv('output/feature_importance.csv', index=False, encoding='utf-8-sig')
    print("\n分析结果已保存到: output/feature_importance.csv")


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("葡萄酒品质分析系统")
    print("=" * 50)
    
    try:
        # 1. 数据获取
        df, feature_names = load_wine_data()
        
        # 2. 数据预处理
        df = preprocess_data(df)
        
        # 3. 科学计算与可视化
        importance_df, figures = analyze_and_visualize(df, feature_names)
        
        # 4. 结果分析
        result_analysis(df, importance_df)
        
        print("\n" + "=" * 50)
        print("分析完成！所有结果已保存到 output/ 目录")
        print("所有可视化窗口已显示完毕")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

