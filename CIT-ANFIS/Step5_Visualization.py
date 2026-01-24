# -*- coding: utf-8 -*-
"""
Step5_Visualization.py
--------------------------
Updates:
1. 修复: 新增 plot_robustness_analysis() 以支持 Step3_Robustness.py 的结果。
2. 保持: 雷达图、预测曲线(Zoom)、超参热力图、消融条形图逻辑不变。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os
from matplotlib.patches import ConnectionPatch

# ==========================================
# 0. 设置学术绘图风格
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

output_dir = 'paper_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 模型颜色与样式配置 (CIT-ANFIS = Deep Blue)
# ==========================================
model_cfg = {
    'Actual': {'color': 'black', 'ls': '-', 'lw': 1.0, 'alpha': 0.6, 'label': 'Actual Load'},
    'CIT_ANFIS': {'color': '#00008B', 'ls': '-', 'lw': 1.2, 'alpha': 0.8, 'label': 'CIT-ANFIS (Ours)'},
    'Transformer': {'color': '#1f77b4', 'ls': '--', 'lw': 1.2, 'alpha': 0.8, 'label': 'Transformer'},
    'LSTM': {'color': '#ff7f0e', 'ls': '--', 'lw': 1.2, 'alpha': 0.8, 'label': 'LSTM'},
    'XGBoost': {'color': '#2ca02c', 'ls': ':', 'lw': 1.2, 'alpha': 0.8, 'label': 'XGBoost'},
    'RandomForest': {'color': '#bcbd22', 'ls': ':', 'lw': 1.2, 'alpha': 0.8, 'label': 'Random Forest'},
    'SVR': {'color': '#9467bd', 'ls': ':', 'lw': 1.2, 'alpha': 0.8, 'label': 'SVR'},
    'KAN': {'color': '#8c564b', 'ls': '-.', 'lw': 1.2, 'alpha': 0.8, 'label': 'KAN'},
    'ANFIS': {'color': '#7f7f7f', 'ls': '-.', 'lw': 1.2, 'alpha': 0.8, 'label': 'Std. ANFIS'},
    'MLP': {'color': '#17becf', 'ls': ':', 'lw': 1.2, 'alpha': 0.8, 'label': 'MLP'}
}


# ==========================================
# 1. 修正后的雷达图 (Refined Radar Chart)
# ==========================================
def plot_radar_chart(df):
    if df is None or df.empty:
        print("Warning: No data for Radar Chart.")
        return

    print("-> Generating Refined Radar Chart...")

    target_models = ['CIT-ANFIS', 'Transformer', 'LSTM', 'XGBoost', 'SVR', 'Standard ANFIS','ANFIS','KAN','RandomForest','MLP']
    available_models = [m for m in target_models if m in df['Model'].unique()]

    if not available_models:
        available_models = df['Model'].unique().tolist()

    # True = 越大越好 (Score), False = 越小越好 (Error)
    metrics_config = {
        'MAPE': False,
        'RMSE': False,
        'MAE': False,
        'R2': True
    }

    categories = [c for c in metrics_config.keys() if c in df.columns]
    if not categories:
        return

    df_sub = df[df['Model'].isin(available_models)].groupby('Model').mean(numeric_only=True).reset_index()
    plot_df = df_sub.copy()

    for feature in categories:
        min_val = plot_df[feature].min()
        max_val = plot_df[feature].max()
        denom = max_val - min_val if max_val != min_val else 1.0

        plot_df[feature] = (plot_df[feature] - min_val) / denom

        if not metrics_config[feature]:
            plot_df[feature] = 1.0 - plot_df[feature]

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='black', size=7, weight='bold')
    ax.set_yticklabels([])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    plt.ylim(0, 1.05)

    colors = ['#00008B', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, model in enumerate(available_models):
        values = plot_df[plot_df['Model'] == model][categories].values.flatten().tolist()
        values += values[:1]

        if 'CIT' in model or 'Proposed' in model:
            c = '#00008B'
            lw = 1
            zorder = 10
            alpha_fill = 0
        else:
            c = colors[i % len(colors)]
            lw = 0.8
            zorder = 1
            alpha_fill = 0

        ax.plot(angles, values, linewidth=lw, linestyle='solid', label=model, color=c, zorder=zorder)
        ax.fill(angles, values, color=c, alpha=alpha_fill)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=5)
    plt.title("Model Performance (Larger Area = Better)", y=1.08, weight='bold', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_radar_chart.png', bbox_inches='tight', dpi=600)
    plt.savefig(f'{output_dir}/1_radar_chart.pdf', bbox_inches='tight')
    plt.close()


# ==========================================
# 2. 最后 180 时间步预测对比图 (Zoom Below)
# ==========================================
def plot_last_168_steps_comparison(dataset_name):
    filename = f"predictions_{dataset_name}.csv"
    if not os.path.exists(filename):
        print(f"Skipping {dataset_name}: {filename} not found.")
        return

    print(f"-> Generating Forecast Plot (Zoom Below) for {dataset_name}...")
    df_pred = pd.read_csv(filename)
    steps_to_plot = min(168, len(df_pred))
    df_plot = df_pred.tail(steps_to_plot).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.5), gridspec_kw={'height_ratios': [2, 1.2]})

    legend_handles = []
    for col in df_plot.columns:
        if col in model_cfg:
            cfg = model_cfg[col]
            ax1.plot(df_plot.index, df_plot[col], label=cfg['label'], color=cfg['color'], linestyle=cfg['ls'], linewidth=cfg['lw'], alpha=cfg['alpha'])
            legend_handles.append(plt.Line2D([], [], color=cfg['color'], linestyle=cfg['ls'], linewidth=cfg['lw']*2.5, alpha=cfg['alpha'], label=cfg['label']))

    ax1.set_title(f"Forecast Overview (Last {steps_to_plot} Hours) - {dataset_name}", weight='bold')
    ax1.set_ylabel("Load Value (MW)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(0, steps_to_plot)
    ax1.tick_params(labelbottom=False)
    ax1.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=5, frameon=False, fontsize=7)

    zoom_start, zoom_end = 80, 130
    if zoom_end > len(df_plot):
        zoom_start, zoom_end = len(df_plot) - 50, len(df_plot)
    df_zoom = df_plot.iloc[zoom_start:zoom_end]

    for col in df_zoom.columns:
        if col in model_cfg:
            cfg = model_cfg[col]
            ax2.plot(df_zoom.index, df_zoom[col], color=cfg['color'], linestyle=cfg['ls'], linewidth=cfg['lw']*1.3, alpha=cfg['alpha'])

    ax2.set_title("Detailed View (Zoom-in)", fontsize=11, fontstyle='italic', loc='left')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Load Value")
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax1.axvspan(zoom_start, zoom_end, color='gray', alpha=0.15, lw=0)
    con1 = ConnectionPatch(xyA=(zoom_start, ax1.get_ylim()[0]), xyB=(zoom_start, ax2.get_ylim()[1]), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="gray", ls="--", alpha=0.5)
    con2 = ConnectionPatch(xyA=(zoom_end, ax1.get_ylim()[0]), xyB=(zoom_end, ax2.get_ylim()[1]), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="gray", ls="--", alpha=0.5)
    fig.add_artist(con1)
    fig.add_artist(con2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'{output_dir}/2_forecast_{dataset_name}.png', bbox_inches='tight', dpi=600)
    plt.savefig(f'{output_dir}/2_forecast_{dataset_name}.pdf', bbox_inches='tight')
    plt.close()


# ==========================================
# 3. 超参数敏感性热力图
# ==========================================
def plot_hyperparam_heatmap():
    if not os.path.exists('hyperparam_sensitivity.csv'):
        return
    print("-> Generating Hyperparameter Heatmap...")
    df = pd.read_csv('hyperparam_sensitivity.csv')

    if 'Dataset' in df.columns:
        ds_list = df['Dataset'].unique()
        df = df[df['Dataset'] == ds_list[0]]

    pivot = df.pivot(index='n_estimators', columns='max_depth', values='MAPE')

    plt.figure(figsize=(3.5, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", cbar_kws={'label': 'MAPE (%)'}, annot_kws={'fontsize': 7})
    plt.title(f"Hyperparameter Sensitivity\n(n_estimators vs max_depth)", weight='bold', fontsize=9)
    plt.ylabel("Number of Rules (Estimators)", fontsize=8)
    plt.xlabel("Tree Depth", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=7)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(labelsize=7)
    cbar.set_label('MAPE (%)', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_sensitivity_heatmap.png', dpi=600)
    plt.savefig(f'{output_dir}/3_sensitivity_heatmap.pdf')
    plt.close()


# ==========================================
# 4. 消融实验条形图
# ==========================================
def plot_ablation_comparison():
    possible_files = ['ablation_results.csv', 'ablation_tuned.csv']
    target_file = None
    for f in possible_files:
        if os.path.exists(f):
            target_file = f
            break

    if not target_file:
        print("[Info] No ablation results found (skipping ablation plot).")
        return

    print(f"-> Generating Ablation Plot using {target_file}...")
    df = pd.read_csv(target_file)
    
    plt.figure(figsize=(3.5, 4.5))
    
    if 'Dataset' in df.columns and 'Variant' in df.columns:
        sns.barplot(data=df, x='Dataset', y='MAPE', hue='Variant', palette='Set2', edgecolor='black', linewidth=0.5)
        plt.legend(loc='upper left', fontsize=6, title_fontsize=7)
    elif 'Variant' in df.columns:
        sns.barplot(data=df, x='Variant', y='MAPE', hue='Variant', palette='Set2', edgecolor='black', linewidth=0.5, legend=False)
        plt.xticks(rotation=15, fontsize=7)
        plt.xlabel("Ablation Variant", fontsize=8)
    else:
        print("[Error] Invalid Ablation CSV format.")
        return

    plt.title("Ablation Study: Component Impact Analysis", weight='bold', fontsize=9)
    plt.ylabel("MAPE (%)", fontsize=8)
    plt.xlabel("Dataset", fontsize=8)
    plt.tick_params(axis='x', labelsize=7)
    plt.tick_params(axis='y', labelsize=7)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_ablation_plot.png', dpi=600)
    plt.savefig(f'{output_dir}/4_ablation_plot.pdf')
    plt.close()


# ==========================================
# 5. [修改] 鲁棒性/噪声测试折线图
# ==========================================
def plot_robustness_analysis():
    # 文件路径
    ablation_file = "ablation_results.csv"
    robustness_file = "ablation_robustness.csv"

    # 1. 准备 Noise=0 的数据 (来自消融实验)
    if not os.path.exists(ablation_file):
        print(f"[Info] {ablation_file} not found. Skipping robustness plot (need baseline).")
        return

    df_ablation = pd.read_csv(ablation_file)
    # 检查列名是否符合要求
    req_cols = {'Dataset', 'Variant', 'MAPE'}
    if not req_cols.issubset(df_ablation.columns):
        print(f"[Error] {ablation_file} missing columns: {req_cols - set(df_ablation.columns)}")
        return

    # 提取所需列并设置 Noise = 0
    df_zero = df_ablation[['Dataset', 'Variant', 'MAPE']].copy()
    df_zero['Noise'] = 0

    # 2. 准备 Noise=10, 20 的数据 (来自鲁棒性测试)
    if not os.path.exists(robustness_file):
        print(f"[Info] {robustness_file} not found. Skipping robustness plot.")
        return

    df_robust = pd.read_csv(robustness_file)

    # [关键修改] 只筛选 Noise 为 10 和 20 的数据
    target_noise = [10, 20]
    df_robust_filtered = df_robust[df_robust['Noise'].isin(target_noise)].copy()

    if df_robust_filtered.empty:
        print("[Warning] No data found for Noise=10 or 20 in robustness file.")
        return

    # 3. 合并数据
    df_final = pd.concat([df_zero, df_robust_filtered], ignore_index=True)

    # 4. 绘图
    datasets = df_final['Dataset'].unique()

    # 定义变体样式
    variant_styles = {
        'Full-Model': ('#00008B', 'o', '-'),  # 深蓝, 实线
        'No-Causal': ('#d62728', 's', '--'),  # 红色, 虚线
        'No-LSE': ('#ff7f0e', '^', '-.')  # 橙色, 点划线
    }

    for ds in datasets:
        # 跳过 Belgium 数据集
        if ds == "Belgium":
            continue
            
        df_ds = df_final[df_final['Dataset'] == ds].sort_values('Noise')

        if df_ds.empty:
            continue

        plt.figure(figsize=(4, 3.2))

        # 分组绘制每一条线
        variants = df_ds['Variant'].unique()
        for variant in variants:
            subset = df_ds[df_ds['Variant'] == variant]

            # 获取样式，如果没有定义则使用默认灰色
            color, marker, ls = variant_styles.get(variant, ('gray', 'x', ':'))

            plt.plot(subset['Noise'], subset['MAPE'],
                     label=variant,
                     color=color,
                     marker=marker,
                     linestyle=ls,
                     linewidth=1.8,
                     markersize=6)

        plt.title(f"Robustness Analysis - {ds}", weight='bold', fontsize=10)
        plt.xlabel(r"Noise Level ($\sigma$)", fontsize=9)
        plt.ylabel("MAPE (%)", fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=8)

        # 设置 X 轴刻度为 0, 10, 20
        plt.xticks([0, 10, 20])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/5_robustness_{ds}.png', bbox_inches='tight', dpi=600)
        plt.savefig(f'{output_dir}/5_robustness_{ds}.pdf', bbox_inches='tight')
        plt.close()
        print(f"   Saved robustness plot for {ds} (Noise levels: 0, 10, 20)")


# ==========================================
# 6. [新增] 导出全量实验数据汇总表
# ==========================================
def export_unified_data_summary():
    print(f"\n-> Exporting Unified Data Summary...")

    all_data = []

    # --- A. 加载基准对比实验 (Benchmark) ---
    bench_file = 'benchmark_results_final.csv'
    if os.path.exists(bench_file):
        df_bench = pd.read_csv(bench_file)
        df_bench['Experiment_Type'] = 'Benchmark Comparison'  # 标记实验类型
        df_bench['Noise_Level'] = 0  # 基准实验默认为无噪声
        # 统一模型名称列
        if 'Model' in df_bench.columns:
            df_bench.rename(columns={'Model': 'Model_Name'}, inplace=True)
        all_data.append(df_bench)
        print(f"   Loaded {len(df_bench)} rows from Benchmark.")

    # --- B. 加载消融实验 (Ablation, Noise=0) ---
    abl_file = 'ablation_results.csv'
    if os.path.exists(abl_file):
        df_abl = pd.read_csv(abl_file)
        df_abl['Experiment_Type'] = 'Ablation Study'
        df_abl['Noise_Level'] = 0
        # 将 Variant 改名为 Model_Name 以便对齐
        if 'Variant' in df_abl.columns:
            df_abl.rename(columns={'Variant': 'Model_Name'}, inplace=True)
        all_data.append(df_abl)
        print(f"   Loaded {len(df_abl)} rows from Ablation.")

    # --- C. 加载鲁棒性实验 (Robustness, Noise>0) ---
    rob_file = 'ablation_robustness.csv'
    if os.path.exists(rob_file):
        df_rob = pd.read_csv(rob_file)
        df_rob['Experiment_Type'] = 'Robustness Test'
        # 鲁棒性文件通常只有 Dataset, Noise, Variant, MAPE
        # 需要重命名列以对齐
        rename_map = {
            'Variant': 'Model_Name',
            'Noise': 'Noise_Level'
        }
        df_rob.rename(columns=rename_map, inplace=True)
        all_data.append(df_rob)
        print(f"   Loaded {len(df_rob)} rows from Robustness.")

    # --- D. 合并与导出 ---
    if not all_data:
        print("[Warning] No result files found to merge.")
        return

    # 上下合并
    df_final = pd.concat(all_data, ignore_index=True)

    # 整理列顺序 (美观)
    # 优先显示的列
    priority_cols = ['Experiment_Type', 'Dataset', 'Model_Name', 'Noise_Level', 'MAPE', 'RMSE', 'MAE', 'R2', 'Time(s)']
    # 数据中实际存在的列
    existing_cols = [c for c in priority_cols if c in df_final.columns]
    # 其他可能存在的列 (放在最后)
    other_cols = [c for c in df_final.columns if c not in priority_cols]

    final_cols = existing_cols + other_cols
    df_final = df_final[final_cols]

    # 填充 NaN (比如鲁棒性测试可能没有 RMSE/Time)
    # df_final.fillna('-', inplace=True) # 可选：填补空值

    # 保存
    output_path = f'{output_dir}/TOTAL_EXPERIMENT_SUMMARY.csv'
    df_final.to_csv(output_path, index=False)
    print(f"   [Success] Combined data saved to: {output_path}")
# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print(f"\n{'=' * 60}\nVISUALIZATION ENGINE STARTED (Refined Radar + Zoom Below)\n{'=' * 60}")

    # 1. 绘制雷达图 & 总条形图
    if os.path.exists('benchmark_results_final.csv'):
        df_main = pd.read_csv('benchmark_results_final.csv')
        plot_radar_chart(df_main)

        plt.figure(figsize=(3.5, 4.5))
        sns.barplot(data=df_main, x='Dataset', y='MAPE', hue='Model', palette='tab10', edgecolor='black')
        plt.title("Global Performance Comparison (MAPE)", weight='bold', fontsize=9)
        plt.ylabel("MAPE (%)", fontsize=8)
        plt.xlabel("Dataset", fontsize=8)
        plt.tick_params(axis='x', labelsize=7)
        plt.tick_params(axis='y', labelsize=7)
        plt.legend(loc='upper right', fontsize=5, title_fontsize=6)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/6_overall_mape.png', dpi=600)
        plt.savefig(f'{output_dir}/6_overall_mape.pdf')
        plt.close()
    else:
        print("[Info] 'benchmark_results_final.csv' not found. Skipping Radar Chart.")

    # 2. 绘制预测曲线 (Zoom Below)
    datasets = ["Malaysia", "ISO-NE", "Belgium"]
    for ds in datasets:
        plot_last_168_steps_comparison(ds)

    # 3. 绘制辅助分析图
    plot_hyperparam_heatmap()     # Step 3 Hyperparam
    plot_ablation_comparison()    # Step 2 Ablation
    plot_robustness_analysis()    # Step 3 Robustness (新增)
    # 4. [新增] 导出总数据表
    export_unified_data_summary()  # <--- 这里调用新函数
    print(f"\n{'=' * 60}\nALL FIGURES SAVED TO '{output_dir}/'\n{'=' * 60}")