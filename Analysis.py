import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.regressionplots import plot_regress_exog
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tools.tools import add_constant

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 后端为 Agg（适用于无图形界面环境）
import matplotlib
matplotlib.use('Agg')

# 读取数据（请确保文件路径正确）
datas = pd.read_csv('D:/datas.csv', encoding='UTF-8')

# 设置图片中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif', "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学符号字体

# --------------------- 描述性统计分析 ---------------------
# 计算平均值
print('人口平均出生率:', datas['人口出生率'].mean())
print('人口平均死亡率:', datas['人口死亡率'].mean())
print('人口平均自然增长率:', datas['人口自然增长率'].mean())

# 计算中位数
print('人口出生率中位数:', datas['人口出生率'].median())
print('人口死亡率中位数:', datas['人口死亡率'].median())
print('人口自然增长率中位数:', datas['人口自然增长率'].median())

# 计算标准差
print('人口出生率标准差:', datas['人口出生率'].std())
print('人口死亡率标准差:', datas['人口死亡率'].std())
print('人口自然增长率标准差:', datas['人口自然增长率'].std())

# 计算方差
print('人口出生率方差:', datas['人口出生率'].var())
print('人口死亡率方差:', datas['人口死亡率'].var())
print('人口自然增长率方差:', datas['人口自然增长率'].var())

# 计算偏度系数
print('人口出生率偏度系数:', stats.skew(datas['人口出生率']))
print('人口死亡率偏度系数:', stats.skew(datas['人口死亡率']))
print('人口自然增长率偏度系数:', stats.skew(datas['人口自然增长率']))

# 生成频数分布表
print('数据基本统计信息：')
print(datas.describe())

# --------------------- 可视化分析 - 直方图与箱线图 ---------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 15))
sns.histplot(datas['人口出生率'], kde=True, stat='density', ax=axes[0], color='lightblue')
axes[0].set_xlabel('人口出生率')
axes[0].set_ylabel('密度')
axes[0].tick_params(axis='x', rotation=45)
sns.rugplot(datas['人口出生率'], ax=axes[0], color='red')
sns.boxplot(x=datas['人口出生率'], ax=axes[0].inset_axes([0.55, 0.5, 0.4, 0.4]), orient='h')

sns.histplot(datas['人口死亡率'], kde=True, stat='density', ax=axes[1], color='lightblue')
axes[1].set_xlabel('人口死亡率')
axes[1].set_ylabel('密度')
axes[1].tick_params(axis='x', rotation=45)
sns.rugplot(datas['人口死亡率'], ax=axes[1], color='red')
sns.boxplot(x=datas['人口死亡率'], ax=axes[1].inset_axes([0.55, 0.5, 0.4, 0.4]), orient='h')

sns.histplot(datas['人口自然增长率'], kde=True, stat='density', ax=axes[2], color='lightblue')
axes[2].set_xlabel('人口自然增长率')
axes[2].set_ylabel('密度')
axes[2].tick_params(axis='x', rotation=45)
sns.rugplot(datas['人口自然增长率'], ax=axes[2], color='red')
sns.boxplot(x=datas['人口自然增长率'], ax=axes[2].inset_axes([0.55, 0.5, 0.4, 0.4]), orient='h')

plt.tight_layout()
plt.savefig('static/image/直方图_出生率死亡率自然增长率.png')

# --------------------- 可视化分析 - 核密度图 ---------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
sns.kdeplot(datas['人口自然增长率'], ax=axes[0, 0], color='red', linewidth=3)
sns.kdeplot(datas['人口自然增长率'], ax=axes[0, 1], linewidth=3)
axes[0, 1].fill_between(axes[0, 1].lines[0].get_xdata(), axes[0, 1].lines[0].get_ydata(), color='gold')
sns.rugplot(datas['人口自然增长率'], ax=axes[0, 1], color='red')

sns.kdeplot(datas['人口出生率'], ax=axes[1, 0], color='red', linewidth=3)
sns.kdeplot(datas['人口出生率'], ax=axes[1, 1], linewidth=3)
axes[1, 1].fill_between(axes[1, 1].lines[0].get_xdata(), axes[1, 1].lines[0].get_ydata(), color='gold')
sns.rugplot(datas['人口出生率'], ax=axes[1, 1], color='red')

sns.kdeplot(datas['人口死亡率'], ax=axes[2, 0], color='red', linewidth=3)
sns.kdeplot(datas['人口死亡率'], ax=axes[2, 1], linewidth=3)
axes[2, 1].fill_between(axes[2, 1].lines[0].get_xdata(), axes[2, 1].lines[0].get_ydata(), color='gold')
sns.rugplot(datas['人口死亡率'], ax=axes[2, 1], color='red')

plt.tight_layout()
plt.savefig('static/image/核密度图_人口指标分布.png')

# --------------------- 数据预处理 - 时间序列格式转换 ---------------------
# 去除年份中的"年"字并转换为日期格式
datas['年份'] = datas['年份'].str.replace('年', '')
datas['年份'] = pd.to_datetime(datas['年份'], format='%Y')
datas.set_index('年份', inplace=True)

# --------------------- 可视化分析 - 时间序列图 ---------------------
plt.figure(figsize=(10, 6))
plt.plot(datas['人口出生率'], label='人口出生率', linewidth=2, color='red', marker='o')
plt.plot(datas['人口死亡率'], label='人口死亡率', linewidth=2, color='blue', linestyle='--', marker='s')
plt.plot(datas['人口自然增长率'], label='人口自然增长率', linewidth=2, color='green', linestyle='-.', marker='^')
plt.xlabel('年份')
plt.ylabel('比率')
plt.title('人口相关比率时间序列图')
plt.legend()
plt.grid(True)
plt.savefig('static/image/时间序列图_人口比率变化.png')

# --------------------- 可视化分析 - 散点图 ---------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.scatterplot(x='人口死亡率', y='人口出生率', data=datas, ax=axes[0])
axes[0].set_xlabel('人口死亡率')
axes[0].set_ylabel('人口出生率')

sns.scatterplot(x='人口自然增长率', y='人口出生率', data=datas, ax=axes[1])
axes[1].set_xlabel('人口自然增长率')
axes[1].set_ylabel('人口出生率')

sns.scatterplot(x='人口自然增长率', y='人口死亡率', data=datas, ax=axes[2])
axes[2].set_xlabel('人口自然增长率')
axes[2].set_ylabel('人口死亡率')

plt.tight_layout()
plt.savefig('static/image/散点图_人口指标相关性.png')

# --------------------- 相关性分析 ---------------------
# 计算相关系数
print('出生率和死亡率的相关系数:', datas['人口出生率'].corr(datas['人口死亡率']))
print('出生率和自然增长率的相关系数:', datas['人口出生率'].corr(datas['人口自然增长率']))
print('死亡率和自然增长率的相关系数:', datas['人口死亡率'].corr(datas['人口自然增长率']))

# 相关系数显著性检验（Pearson检验）
print('出生率和死亡率的相关系数检验:\n', stats.pearsonr(datas['人口出生率'], datas['人口死亡率']))
print('出生率和自然增长率的相关系数检验:\n', stats.pearsonr(datas['人口出生率'], datas['人口自然增长率']))
print('死亡率和自然增长率的相关系数检验:\n', stats.pearsonr(datas['人口死亡率'], datas['人口自然增长率']))

# --------------------- 回归分析 - 出生率与死亡率 ---------------------
model = ols('人口出生率 ~ 人口死亡率', data=datas).fit()
print('出生率与死亡率回归分析结果：')
print(model.summary())

# 置信区间
print('出生率与死亡率回归模型的置信区间：')
print(model.conf_int(alpha=0.05))

# 方差分析表
print('出生率与死亡率回归模型的方差分析表：')
print(anova_lm(model))

# 绘制拟合图
plt.figure(figsize=(8, 6))
plt.scatter(datas['人口死亡率'], datas['人口出生率'])
plt.plot(datas['人口死亡率'], model.fittedvalues, color='red', linewidth=2)
for i in range(len(datas)):
    plt.plot([datas['人口死亡率'].iloc[i], datas['人口死亡率'].iloc[i]],
             [datas['人口出生率'].iloc[i], model.fittedvalues[i]],
             'k-', alpha=0.3)
plt.xlabel('人口死亡率')
plt.ylabel('人口出生率')
plt.title('出生率与死亡率回归模型拟合图')
plt.savefig('static/image/回归拟合图_出生率与死亡率.png')

# --------------------- 回归分析 - 出生率与自然增长率 ---------------------
model = ols('人口出生率 ~ 人口自然增长率', data=datas).fit()
print('出生率与自然增长率回归分析结果：')
print(model.summary())

# 置信区间
print('出生率与自然增长率回归模型的置信区间：')
print(model.conf_int(alpha=0.05))

# 方差分析表
print('出生率与自然增长率回归模型的方差分析表：')
print(anova_lm(model))

# 绘制拟合图
plt.figure(figsize=(8, 6))
plt.scatter(datas['人口自然增长率'], datas['人口出生率'])
plt.plot(datas['人口自然增长率'], model.fittedvalues, color='red', linewidth=2)
for i in range(len(datas)):
    plt.plot([datas['人口自然增长率'].iloc[i], datas['人口自然增长率'].iloc[i]],
             [datas['人口出生率'].iloc[i], model.fittedvalues[i]],
             'k-', alpha=0.3)
plt.xlabel('人口自然增长率')
plt.ylabel('人口出生率')
plt.title('出生率与自然增长率回归模型拟合图')
plt.savefig('static/image/回归拟合图_出生率与自然增长率.png')

# --------------------- 回归分析 - 死亡率与自然增长率 ---------------------
model = ols('人口死亡率 ~ 人口自然增长率', data=datas).fit()
print('死亡率与自然增长率回归分析结果：')
print(model.summary())

# 置信区间
print('死亡率与自然增长率回归模型的置信区间：')
print(model.conf_int(alpha=0.05))

# 方差分析表
print('死亡率与自然增长率回归模型的方差分析表：')
print(anova_lm(model))

# 绘制拟合图
plt.figure(figsize=(8, 6))
plt.scatter(datas['人口自然增长率'], datas['人口死亡率'])
plt.plot(datas['人口自然增长率'], model.fittedvalues, color='red', linewidth=2)
for i in range(len(datas)):
    plt.plot([datas['人口自然增长率'].iloc[i], datas['人口自然增长率'].iloc[i]],
             [datas['人口死亡率'].iloc[i], model.fittedvalues[i]],
             'k-', alpha=0.3)
plt.xlabel('人口自然增长率')
plt.ylabel('人口死亡率')
plt.title('死亡率与自然增长率回归模型拟合图')
plt.savefig('static/image/回归拟合图_死亡率与自然增长率.png')

# --------------------- 回归预测 ---------------------
model = ols('人口出生率 ~ 人口死亡率', data=datas).fit()
x0 = datas['人口死亡率']
exog = pd.DataFrame({'人口死亡率': x0})
exog = add_constant(exog)
pre_model = model.predict()

# 计算置信区间和预测区间
_, con_int_lower, con_int_upper = wls_prediction_std(model, exog=exog, alpha=0.05, weights=None)
_, pre_int_lower, pre_int_upper = wls_prediction_std(model, exog=exog, alpha=0.05, weights=None)

pre = pd.DataFrame({
    '人口出生率': datas['人口出生率'],
    '点预测值': pre_model,
    '置信下限': con_int_lower,
    '置信上限': con_int_upper,
    '预测下限': pre_int_lower,
    '预测上限': pre_int_upper
})
print('利用回归方程预测结果：')
print(pre)

# --------------------- 残差分析 ---------------------
# 计算预测值、残差和标准化残差
model = ols('人口出生率 ~ 人口死亡率', data=datas).fit()
pre = model.fittedvalues
res = model.resid
zre = model.resid / np.sqrt(model.mse_resid)

mysummary = pd.DataFrame({
    '人口出生率': datas['人口出生率'],
    '点预测值': pre,
    '残差': res,
    '标准化残差': zre
})
print('计算预测值、残差和标准化残差结果：')
print(mysummary)

# --------------------- 成分残差图 ---------------------
model_1 = ols('人口出生率 ~ 人口自然增长率', data=datas).fit()
fig = plot_regress_exog(model_1, '人口自然增长率')
plt.savefig('static/image/成分残差图_出生率与自然增长率.png')

# --------------------- 模型诊断 - 正态性检验 ---------------------
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(model_1.resid)
plt.title('残差随观测值变化图')
plt.xlabel('观测值序号')
plt.ylabel('残差')

plt.subplot(2, 2, 2)
sns.histplot(model_1.resid, kde=True)
plt.title('残差直方图')
plt.xlabel('残差')
plt.ylabel('频率')

plt.subplot(2, 2, 3)
qqplot(model_1.resid, line='s')
plt.title('残差 QQ 图')

plt.subplot(2, 2, 4)
plt.scatter(model_1.fittedvalues, model_1.resid)
plt.title('残差与拟合值关系图')
plt.xlabel('拟合值')
plt.ylabel('残差')

plt.tight_layout()
plt.savefig('static/image/正态性检验图_残差分析.png')

# --------------------- 模型诊断 - 方差齐性检验 ---------------------
bp_test = het_breuschpagan(model_1.resid, model_1.model.exog)
bp_statistic = bp_test[0]
bp_p_value = bp_test[1]
print(f'布罗施-帕甘检验统计量: {bp_statistic}, p 值: {bp_p_value}')

# 绘制散布—水平图
plt.scatter(model_1.fittedvalues, np.sqrt(np.abs(model_1.resid)))
plt.xlabel('拟合值')
plt.xticks(rotation=45)
plt.ylabel('残差绝对值的平方根')
plt.title('散布—水平图')
plt.savefig('static/image/散布水平图_残差与拟合值.png')

# --------------------- 模型诊断 - 残差独立性检验 ---------------------
dw_statistic = durbin_watson(model_1.resid)
print(f'杜宾-沃森统计量: {dw_statistic}')