from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import os

app = Flask(__name__)

# 图片文件夹路径
IMAGE_FOLDER = os.path.join(app.root_path, 'static/image')

# 加载数据
def load_data():
    df = pd.read_csv('data/population.csv')
    # 重命名列并清理年份数据
    df.columns = ['year', 'birth_rate', 'death_rate', 'growth_rate']

    # 清理年份列：移除"年"字并转换为整数
    df['year'] = df['year'].str.replace('年', '').astype(int)

    return df


# 描述性统计
def get_descriptive_stats():
    df = load_data()
    # 选择数值列
    numeric_cols = ['birth_rate', 'death_rate', 'growth_rate']
    numeric_df = df[numeric_cols]

    # 计算基本统计量
    stats_df = numeric_df.describe().round(2)

    # 添加偏度和峰度
    skewness = numeric_df.skew().round(2).to_dict()
    kurtosis = numeric_df.kurtosis().round(2).to_dict()

    return {
        'table': stats_df.to_dict(),
        'skewness': skewness,
        'kurtosis': kurtosis
    }


# 相关性分析
def get_correlation():
    df = load_data()
    # 选择数值列
    numeric_cols = ['birth_rate', 'death_rate', 'growth_rate']
    numeric_df = df[numeric_cols]

    # 计算相关系数矩阵
    corr_matrix = numeric_df.corr().round(3).to_dict()

    # 计算p值
    p_values = {}
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:  # 避免重复计算
                _, p = stats.pearsonr(df[col1], df[col2])
                p_values[f"{col1}_{col2}"] = round(p, 4)

    # 为散点图矩阵准备数据
    scatter_data_points = df[numeric_cols].values.tolist()

    return {'corr_matrix': corr_matrix, 'p_values': p_values, 'scatter_data_points': scatter_data_points, 'fields': numeric_cols}


# 回归分析
# 回归分析
def get_regression():
    df = load_data()

    # 准备自变量和因变量
    X = df[['birth_rate', 'death_rate']]
    y = df['growth_rate']

    # 添加常数项
    X = sm.add_constant(X)

    # 拟合模型
    model = sm.OLS(y, X).fit()

    # 获取回归系数
    const_coef = model.params['const']
    birth_rate_coef = model.params['birth_rate']
    death_rate_coef = model.params['death_rate']

    # 获取模型统计量
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    f_value = model.fvalue
    f_pvalue = model.f_pvalue

    # 获取标准误
    bse_const = model.bse['const']
    bse_birth = model.bse['birth_rate']
    bse_death = model.bse['death_rate']

    # 获取p值
    p_const = model.pvalues['const']
    p_birth = model.pvalues['birth_rate']
    p_death = model.pvalues['death_rate']

    # 获取预测值和残差
    fitted_values = model.fittedvalues
    residuals = model.resid

    # 获取最新年份的实际值和预测值
    actual_value = df.iloc[-1]['growth_rate']
    predicted_value = model.predict(X.iloc[-1])[0]

    result = {
        'equation': "自然增长率 = {const:.2f} + {birth_rate:.2f} × 出生率 - {death_rate:.2f} × 死亡率".format(
            const=const_coef,
            birth_rate=birth_rate_coef,
            death_rate=abs(death_rate_coef)
        ),
        'coefficients': {
            'const': const_coef,
            'birth_rate': birth_rate_coef,
            'death_rate': death_rate_coef
        },
        'r_squared': r2,
        'model_summary': {
            'fvalue': f_value,
            'rsquared_adj': adj_r2,
            'f_pvalue': f_pvalue,
            'bse_const': bse_const,
            'bse_birth_rate': bse_birth,
            'bse_death_rate': bse_death
        },
        'p_values_model': {
            'const': p_const,
            'birth_rate': p_birth,
            'death_rate': p_death
        },
        'actual': actual_value,
        'prediction': predicted_value,
        'fitted_values': fitted_values.tolist(),
        'residuals': residuals.tolist()
    }

    # 确保没有None值
    for key, value in result.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_value is None:
                    result[key][sub_key] = 0
        elif value is None:
            result[key] = 0

    return result

# 获取图表数据
def get_chart_data():
    df = load_data()

    # 计算箱线图所需的五数概括 [min, Q1, median, Q3, max]（保留2位小数）
    def get_box_data(series):
        return [
            round(float(series.min()), 2),
            round(float(series.quantile(0.25)), 2),
            round(float(series.median()), 2),
            round(float(series.quantile(0.75)), 2),
            round(float(series.max()), 2)
        ]

    # 计算3年移动平均线（用于趋势分析，可选）
    def moving_average(data, window=3):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    # 准备时间序列数据
    years = df['year'].astype(str).tolist()
    birth_rates = [round(x, 2) for x in df['birth_rate'].tolist()]
    death_rates = [round(x, 2) for x in df['death_rate'].tolist()]
    growth_rates = [round(x, 2) for x in df['growth_rate'].tolist()]

    # 计算移动平均线（示例：3年窗口）
    ma_birth = [round(x, 2) for x in moving_average(df['birth_rate'])]
    ma_death = [round(x, 2) for x in moving_average(df['death_rate'])]
    ma_years = years[1:-1]  # 移动平均的年份对齐

    return {
        # 原始时间序列数据
        'years': years,
        'birth_rates': birth_rates,
        'death_rates': death_rates,
        'growth_rates': growth_rates,

        # 移动平均数据（可选）
        'ma_birth': ma_birth,
        'ma_death': ma_death,
        'ma_years': ma_years,

        # 新增箱线图数据（关键修改）
        'birth_box': get_box_data(df['birth_rate']),
        'death_box': get_box_data(df['death_rate']),
        'growth_box': get_box_data(df['growth_rate'])
    }


# 路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/descriptive')
def descriptive():
    stats = get_descriptive_stats()
    return render_template('descriptive.html', stats=stats)


@app.route('/correlation')
def correlation():
    corr_data = get_correlation()
    return render_template('correlation.html', corr_data=corr_data)


@app.route('/regression')
def regression():
    # 获取回归分析结果
    reg_data = get_regression()

    # 确保所有必要字段都存在
    required_fields = ['equation', 'coefficients', 'r_squared', 'model_summary',
                       'p_values_model', 'actual', 'prediction', 'fitted_values', 'residuals']

    for field in required_fields:
        if field not in reg_data:
            reg_data[field] = 0  # 或适当的默认值

    return render_template('regression.html', reg_data=reg_data)

@app.route('/about')
def about():
    return render_template('about.html')

# 新增API端点为图表提供数据
@app.route('/chart_data_for_scatter')
def chart_data_for_scatter():
    df = load_data()
    numeric_cols = ['birth_rate', 'death_rate', 'growth_rate']
    scatter_data = df[numeric_cols].values.tolist()
    return jsonify({
        'scatter_data': scatter_data,
        'fields': ['出生率', '死亡率', '自然增长率'] # 确保字段名与前端对应
    })

@app.route('/regression_diagnostic_data')
def regression_diagnostic_data():
    # 复用 get_regression 中的逻辑或重新计算
    df = load_data()
    X_cols = ['birth_rate', 'death_rate']
    X = df[X_cols]
    y = df['growth_rate']
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    fitted_values = model.fittedvalues.tolist()
    residuals = model.resid.tolist()
    return jsonify({
        'fitted_values': fitted_values,
        'residuals': residuals
    })


@app.route('/chart_data')
def chart_data():
    return jsonify(get_chart_data())

# 新增路由用于展示分析图片
@app.route('/analysis')
def analysis():
    image_files = []
    if os.path.exists(IMAGE_FOLDER):
        for filename in os.listdir(IMAGE_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(filename)
    return render_template('analysis.html', images=image_files)

def get_image_title(filename):
    """根据英文文件名返回对应的中文标题"""
    title_map = {
        "fit_plot_birth_death.png": "出生率与死亡率回归分析",
        "fit_plot_birth_growth.png": "出生率与自然增长率回归分析",
        "fit_plot_death_growth.png": "死亡率与自然增长率回归分析",
        "histogram_plot.png": "人口指标直方图分布",
        "kde_plot.png": "核密度估计分布图",
        "normal_test_plot.png": "正态性检验图",
        "partial_residual_plot.png": "偏残差图",
        "scatter_plot.png": "散点图矩阵",
        "spread_level_plot.png": "散布水平图",
        "time_series_plot.png": "时间序列趋势图"
    }
    return title_map.get(filename, filename.split('.')[0].replace('_', ' ').title())

def get_image_description(filename):
    """根据文件名返回对应的分析描述"""
    desc_map = {
        "fit_plot_birth_death.png": "展示出生率与死亡率之间的回归关系，红色直线表示拟合结果，灰色区域表示置信区间。",
        "fit_plot_birth_growth.png": "展示出生率与自然增长率之间的回归关系，点线表示实际观测值，直线表示拟合结果。",
        "fit_plot_death_growth.png": "展示死亡率与自然增长率之间的回归关系，包含残差分布和拟合线。",
        "histogram_plot.png": "展示人口统计指标的直方图分布，包含出生率、死亡率和自然增长率的分布情况。",
        "kde_plot.png": "使用核密度估计展示人口指标的平滑分布曲线，比直方图更能反映数据分布特征。",
        "normal_test_plot.png": "用于检验数据是否符合正态分布，Q-Q图中点越接近直线表示越符合正态性假设。",
        "partial_residual_plot.png": "展示在控制其他变量后，单个变量与因变量的关系，用于诊断回归模型。",
        "scatter_plot.png": "散点图矩阵展示所有变量两两之间的关系，对角线显示各变量的分布。",
        "spread_level_plot.png": "用于检验回归模型的同方差性，若点随机分布则表示满足假设。",
        "time_series_plot.png": "展示2006-2024年人口指标的时间变化趋势，包含出生率、死亡率和自然增长率。"
    }
    return desc_map.get(filename, "这是通过Python统计分析生成的数据可视化图表。")

# 注册为模板全局变量
app.jinja_env.globals.update(
    get_image_title=get_image_title,
    get_image_description=get_image_description
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)