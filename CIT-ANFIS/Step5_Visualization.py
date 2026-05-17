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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许 PyTorch 和 Matplotlib/Numpy 共享 OpenMP 库
import torch
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
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
RESULT_DIR = 'result'
MODEL_DIR = 'models'
output_dir = 'paper_figures'
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
    'MLP': {'color': '#17becf', 'ls': ':', 'lw': 1.2, 'alpha': 0.8, 'label': 'MLP'},
    'XMQ_Proposed': {'color': '#FF69B4', 'ls': ':', 'lw': 1.2, 'alpha': 0.8, 'label': 'FDNN-LA'}
}


# ==========================================
# 1. 修正后的雷达图 (Refined Radar Chart)
# ==========================================
def plot_radar_chart(df):
    if df is None or df.empty:
        print("Warning: No data for Radar Chart.")
        return

    print("-> Generating Refined Radar Chart...")

    target_models = ['CIT-ANFIS', 'FDNN-LA' ,'Transformer', 'LSTM', 'XGBoost', 'SVR', 'Standard ANFIS','ANFIS','KAN','RandomForest','MLP']
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

    colors = ['#00008B','#FF69B4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_radar_chart.png', bbox_inches='tight', dpi=600)
    plt.savefig(f'{output_dir}/1_radar_chart.pdf', bbox_inches='tight')
    plt.close()


# ==========================================
# 2. 最后 180 时间步预测对比图 (Zoom Below)
# ==========================================
def plot_last_168_steps_comparison(dataset_name):
    filename = f"result/predictions_{dataset_name}.csv"
    pure_filename = f"result/pure_predictions_{dataset_name}.csv"
    if not os.path.exists(filename):
        print(f"Skipping {dataset_name}: {filename} not found.")
        return

    print(f"-> Generating Forecast Plot (Zoom Below) for {dataset_name}...")
    df_pred = pd.read_csv(filename)
    
    # === 新增：读取并合并 FDNN-LA 模型的预测结果 ===
    if os.path.exists(pure_filename):
        df_pure = pd.read_csv(pure_filename)
        if 'XMQ_Proposed' in df_pure.columns:
            df_pred['XMQ_Proposed'] = df_pure['XMQ_Proposed']
            
    steps_to_plot = min(168, len(df_pred))
    df_plot = df_pred.tail(steps_to_plot).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 3), gridspec_kw={'height_ratios': [2, 1.2]})

    legend_handles = []
    for col in df_plot.columns:
        if col in model_cfg:
            cfg = model_cfg[col]
            ax1.plot(df_plot.index, df_plot[col], label=cfg['label'], color=cfg['color'], linestyle=cfg['ls'], linewidth=cfg['lw'], alpha=cfg['alpha'])
            legend_handles.append(plt.Line2D([], [], color=cfg['color'], linestyle=cfg['ls'], linewidth=cfg['lw']*2.5, alpha=cfg['alpha'], label=cfg['label']))

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
    if not os.path.exists('result/hyperparam_sensitivity.csv'):
        return
    print("-> Generating Hyperparameter Heatmap...")
    df = pd.read_csv('result/hyperparam_sensitivity.csv')

    if 'Dataset' in df.columns:
        ds_list = df['Dataset'].unique()
        for ds in ds_list:
            df_sub = df[df['Dataset'] == ds]
            pivot = df_sub.pivot(index='n_estimators', columns='max_depth', values='MAPE')

            plt.figure(figsize=(3.5, 3.5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", cbar_kws={'label': 'MAPE (%)'}, annot_kws={'fontsize': 7})
            plt.title(f"Hyperparameter Sensitivity ({ds})\n(n_estimators vs max_depth)", weight='bold', fontsize=9)
            plt.ylabel("Number of Rules (Estimators)", fontsize=8)
            plt.xlabel("Tree Depth", fontsize=8)
            plt.tick_params(axis='both', which='major', labelsize=7)
            cbar = plt.gca().collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(labelsize=7)
            cbar.set_label('MAPE (%)', fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/3_sensitivity_heatmap_{ds}.png', dpi=600)
            plt.savefig(f'{output_dir}/3_sensitivity_heatmap_{ds}.pdf')
            plt.close()
    else:
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
    possible_files = ['result/ablation_results.csv', 'result/ablation_tuned.csv']
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
    
    plt.figure(figsize=(3.5, 3.5))
    
    if 'Dataset' in df.columns and 'Variant' in df.columns:
        sns.barplot(data=df, x='Dataset', y='MAPE', hue='Variant', palette='Set2', edgecolor='black', linewidth=0.5)
        plt.legend(loc='upper right', fontsize=6, title_fontsize=7)
    elif 'Variant' in df.columns:
        sns.barplot(data=df, x='Variant', y='MAPE', hue='Variant', palette='Set2', edgecolor='black', linewidth=0.5, legend=False)
        plt.xticks(rotation=15, fontsize=7)
        plt.xlabel("Ablation Variant", fontsize=8)
    else:
        print("[Error] Invalid Ablation CSV format.")
        return

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
# 6. [新增] 导出全量实验数据汇总表
# ==========================================
def export_unified_data_summary():
    print(f"\n-> Exporting Unified Data Summary...")

    all_data = []

    # --- A. 加载基准对比实验 (Benchmark) ---
    bench_file = 'result/benchmark_results_final.csv'
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
# --- C.5 加载外部对比模型实验 (FDNN-LA) ---
    xmq_file = 'result/xmq_only_results_final.csv'
    if os.path.exists(xmq_file):
        df_xmq = pd.read_csv(xmq_file)
        df_xmq['Experiment_Type'] = 'Benchmark Comparison'
        df_xmq['Noise_Level'] = 0
        if 'Model' in df_xmq.columns:
            df_xmq.rename(columns={'Model': 'Model_Name'}, inplace=True)
        # 统一命名为 FDNN-LA
        df_xmq['Model_Name'] = df_xmq['Model_Name'].replace({'XMQ_Proposed (Paper)': 'FDNN-LA'})
        all_data.append(df_xmq)
        print(f"   Loaded {len(df_xmq)} rows from XMQ (FDNN-LA) Benchmark.")
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

def plot_causal_contribution(dataset_name):
    model_path = os.path.join(MODEL_DIR, f"best_model_{dataset_name}.pth")
    if not os.path.exists(model_path):
        print(f"[Skip] Model file not found for {dataset_name}.")
        return
        
    print(f"-> Generating Causal Contribution (Lollipop) for {dataset_name}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    feat_names = checkpoint['feature_names']
    weights = checkpoint['state_dict']['attention_weights'].numpy()
    
    imp_df = pd.DataFrame({'Feature': feat_names, 'Weight': weights}).sort_values(by='Weight', ascending=True)
    
    fig_height = max(4.5, len(feat_names) * 0.15) 
    plt.figure(figsize=(7, fig_height))
    
    plt.hlines(y=imp_df['Feature'], xmin=0, xmax=imp_df['Weight'], color='lightsteelblue', alpha=0.8, linewidth=2)
    
    norm = plt.Normalize(imp_df['Weight'].min(), imp_df['Weight'].max())
    cmap = sns.color_palette("YlGnBu_r", as_cmap=True)
    colors = cmap(norm(imp_df['Weight']))
    
    plt.scatter(imp_df['Weight'], imp_df['Feature'], s=60, color=colors, edgecolors='black', linewidth=0.5, zorder=3)
    
    plt.xlabel('Attention Weight / Causal Importance', fontsize=10)
    plt.ylabel('Features', fontsize=10)
    plt.tick_params(axis='y', labelsize=8)
    plt.grid(axis='y', linestyle='', alpha=0) 
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/7_causal_lollipop_{dataset_name}.png', dpi=600)
    plt.savefig(f'{output_dir}/7_causal_lollipop_{dataset_name}.pdf')
    plt.close()

# ==========================================
# 3. 规则森林矩阵与 双模逻辑(语义模糊 + 精确数学) 导出
# ==========================================
def plot_top_rules_forest(dataset_name, top_n=50):
    model_path = os.path.join(MODEL_DIR, f"best_model_{dataset_name}.pth")
    if not os.path.exists(model_path): return
    
    print(f"-> Extracting Dual-Mode Rules & Matrix for {dataset_name}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    feat_names = checkpoint['feature_names']
    state = checkpoint['state_dict']
    
    # 获取真实的网络结构参数
    feat_idxs = state['rule_feat_idxs'].numpy()
    masks = state['rule_masks'].numpy()
    thresh_tensor = state.get('rule_threshs')
    sign_tensor = state.get('rule_signs')
    cons_tensor = state.get('consequent_params')
    
    num_rules = min(top_n, feat_idxs.shape[0])
    num_feats = len(feat_names)

    def map_count_to_label(count):
        mapping = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
        return mapping.get(count, f"Level {count} Intensity")

    fuzzy_rules_text = [f"====== CLASSIC FUZZY RULE BASE: {dataset_name} ======\n"]
    exact_rules_text = [f"====== EXACT MATHEMATICAL RULES (TSK): {dataset_name} ======\n"]
    
    rule_matrix = np.zeros((num_rules, num_feats))
    
    for r_idx in range(num_rules):
        active_feats_in_this_rule = []
        exact_conditions = []
        
        # 遍历该规则的路径深度
        for j in range(feat_idxs.shape[1]):
            if masks[r_idx, j] > 0:
                f_idx = int(feat_idxs[r_idx, j])
                if f_idx < num_feats:
                    feat_name = feat_names[f_idx]
                    active_feats_in_this_rule.append(feat_name)
                    rule_matrix[r_idx, f_idx] = 1 
                    
                    # 提取精确的切分阈值和方向 (用于数学公式)
                    if thresh_tensor is not None and sign_tensor is not None:
                        val_thresh = round(thresh_tensor[r_idx, j].item(), 4)
                        val_sign = sign_tensor[r_idx, j].item()
                        operator = "< " if val_sign <= 0 else ">="
                        exact_conditions.append(f"{feat_name} {operator} {val_thresh}")
                    else:
                        exact_conditions.append(f"{feat_name} ? Thresh")
        
        if active_feats_in_this_rule:
            feat_counts = Counter(active_feats_in_this_rule)
            
            # --- 分支 1：生成经典模糊规则 (语义化) ---
            condition_parts_fuzzy = [f"{feat} is {map_count_to_label(count)}" for feat, count in feat_counts.items()]
            fuzzy_cond_str = " AND ".join(condition_parts_fuzzy)
            fuzzy_str = f"Rule {r_idx + 1:02d}: IF [{fuzzy_cond_str}] THEN Load_Forecast is Pattern_{r_idx + 1:02d}"
            fuzzy_rules_text.append(fuzzy_str)

            # --- 分支 2：生成精确数学规则 (TSK 方程) ---
            tsk_terms = []
            unique_feats = list(feat_counts.keys())
            
            if cons_tensor is not None:
                rule_weights = cons_tensor[r_idx] 
                for feat in unique_feats:
                    f_idx = feat_names.index(feat)
                    try:
                        val_w = round(rule_weights[f_idx].item(), 3)
                        tsk_terms.append(f"{val_w} · {feat}")
                    except Exception:
                        tsk_terms.append(f"W_{feat} · {feat}")
                try:
                    val_b = round(rule_weights[-1].item(), 3) # Bias 通常在最后
                    bias_str = f"+ {val_b}" if val_b >= 0 else f"- {abs(val_b)}"
                    tsk_terms.append(bias_str)
                except Exception:
                    tsk_terms.append("+ Bias")
            else:
                for feat in unique_feats: tsk_terms.append(f"W_{feat} · {feat}")
                tsk_terms.append("+ Bias")
            
            exact_cond_str = " AND\n  ".join(exact_conditions)
            
            if tsk_terms:
                equation_str = tsk_terms[0]
                for t in tsk_terms[1:]:
                    equation_str += f" + {t}" if not t.startswith("-") and not t.startswith("+") else f" {t}"
            else:
                equation_str = "0"
            
            exact_str = (f"Rule {r_idx + 1:02d}:\nIF \n  {exact_cond_str}\n"
                         f"THEN \n  Load = {equation_str}\n{'-'*40}")
            exact_rules_text.append(exact_str)
    
    # 保存文本文件
    fuzzy_out_path = os.path.join(output_dir, f'7_classic_fuzzy_rules_{dataset_name}.txt')
    with open(fuzzy_out_path, 'w', encoding='utf-8') as f: f.write("\n".join(fuzzy_rules_text))
        
    exact_out_path = os.path.join(output_dir, f'7_exact_math_rules_{dataset_name}.txt')
    with open(exact_out_path, 'w', encoding='utf-8') as f: f.write("\n".join(exact_rules_text))
        
    print(f"   [Success] Fuzzy Rules exported to {fuzzy_out_path}")
    print(f"   [Success] Exact Math Rules exported to {exact_out_path}")

    # --- B. 绘制规则-特征映射矩阵图 ---
    active_feat_cols = np.where(rule_matrix.sum(axis=0) > 0)[0]
    plot_matrix = rule_matrix[:, active_feat_cols]
    active_feat_names = [feat_names[i] for i in active_feat_cols]
    
    fig_h = max(5, num_rules * 0.12)
    plt.figure(figsize=(max(6, len(active_feat_names)*0.3), fig_h))
    
    cmap = LinearSegmentedColormap.from_list("RuleCmap", ["#f0f0f0", "#1f77b4"])
    ax = sns.heatmap(plot_matrix, cmap=cmap, cbar=False, linewidths=0.5, linecolor='white')
    
    ax.set_xticks(np.arange(len(active_feat_names)) + 0.5)
    ax.set_xticklabels(active_feat_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(num_rules) + 0.5)
    ax.set_yticklabels([f"Rule {i+1}" for i in range(num_rules)], rotation=0, fontsize=7)

    plt.xlabel('Features Selected in Rules', fontsize=10, weight='bold')
    plt.ylabel('Extracted Top Rules', fontsize=10, weight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', edgecolor='w', label='Feature Included in Rule Condition')]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/8_rule_matrix_{dataset_name}.png', dpi=600)
    plt.savefig(f'{output_dir}/8_rule_matrix_{dataset_name}.pdf')
    plt.close()
def plot_consequent_weight_boxplot(dataset_name, top_n=10):
    """
    展示前 N 个关键特征在所有规则的 THEN 线性方程中，其系数(W_x)的分布情况。
    """
    model_path = os.path.join(MODEL_DIR, f"best_model_{dataset_name}.pth")
    if not os.path.exists(model_path): return
    print(f"-> Generating Consequent Boxplot for {dataset_name}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    feat_names = checkpoint['feature_names']
    cons_params = checkpoint['state_dict']['consequent_params'].numpy()
    
    # 提取特征权重 (去掉最后的 Bias)
    feat_weights = cons_params[:, :len(feat_names)]
    
    # 根据绝对值中位数找出最重要的 top_n 特征
    median_abs_weights = np.median(np.abs(feat_weights), axis=0)
    top_indices = np.argsort(median_abs_weights)[::-1][:top_n]
    
    top_feats = [feat_names[i] for i in top_indices]
    plot_data = feat_weights[:, top_indices]
    
    plt.figure(figsize=(7.16, 3))
    
    sns.boxplot(data=plot_data, orient='h', palette='Set3', linewidth=1.2, fliersize=3)
    
    # 添加一条 0 刻度线作为参考
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.yticks(np.arange(top_n), top_feats, fontsize=9)
    plt.xlabel('Coefficient Value in TSK Equations')
    plt.ylabel('Top Contributing Features')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/9_consequent_boxplot_{dataset_name}.png', dpi=600)
    plt.savefig(f'{output_dir}/9_consequent_boxplot_{dataset_name}.pdf')
    plt.close()
# ==========================================
# 5. [新增] 鲁棒性推理趋势图 (Robustness Analysis)
# ==========================================
def plot_robustness_analysis():
    file_path = 'result/robustness_inference.csv'
    if not os.path.exists(file_path):
        print(f"[Info] Robustness results not found at {file_path} (skipping robustness plot).")
        return

    print("-> Generating Inference Robustness Plot...")
    df = pd.read_csv(file_path)

    # 绘制折线图
    plt.figure(figsize=(4.0, 3.5))
    
    # 使用 seaborn 绘制带有标记的折线图，展示 MAPE 随噪声变化趋势
    sns.lineplot(
        data=df, 
        x='Noise_Level', 
        y='MAPE', 
        hue='Dataset', 
        style='Dataset', 
        markers=True, 
        dashes=False, 
        linewidth=1.5, 
        markersize=7, 
        palette='Set1'
    )

    plt.xlabel('Noise Level (σ)', fontsize=8)
    plt.ylabel('MAPE (%)', fontsize=8)
    
    # 根据 Step 4 中设定的噪声等级设置 X 轴刻度
    noise_levels = sorted(df['Noise_Level'].unique())
    plt.xticks(noise_levels, fontsize=7)
    plt.yticks(fontsize=7)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Dataset', title_fontsize=7, fontsize=6, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_robustness_inference.png', dpi=600)
    plt.savefig(f'{output_dir}/5_robustness_inference.pdf')
    plt.close()
    print(f"   [Success] Robustness plot saved to {output_dir}/5_robustness_inference.[png/pdf]")
# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print(f"\n{'=' * 60}\nVISUALIZATION ENGINE STARTED (Refined Radar + Zoom Below)\n{'=' * 60}")

# 1. 绘制雷达图 & 总条形图
    if os.path.exists('result/benchmark_results_final.csv'):
        df_main = pd.read_csv('result/benchmark_results_final.csv')
        
# [已有逻辑]：从消融实验中提取 Full-Model 数据合并为 CIT-ANFIS
        ablation_file = 'result/ablation_results.csv'
        if os.path.exists(ablation_file):
            df_abl = pd.read_csv(ablation_file)
            df_ours = df_abl[df_abl['Variant'] == 'Full-Model'].copy()
            df_ours.rename(columns={'Variant': 'Model'}, inplace=True)
            df_ours['Model'] = 'CIT-ANFIS'
            df_main = pd.concat([df_main, df_ours], ignore_index=True)
            
        # =========================================================
        # [新增逻辑]：合并 FDNN-LA 的对比实验数据
        # =========================================================
        xmq_file = 'result/xmq_only_results_final.csv'
        if os.path.exists(xmq_file):
            df_xmq = pd.read_csv(xmq_file)
            # 将原脚本中的名字替换为学术简称 FDNN-LA
            df_xmq['Model'] = df_xmq['Model'].replace({'XMQ_Proposed (Paper)': 'FDNN-LA'})
            df_main = pd.concat([df_main, df_xmq], ignore_index=True)
            print("   [Info] Successfully injected 'XMQ_Proposed' as 'FDNN-LA'.")
        # =========================================================

        plot_radar_chart(df_main)

        plt.figure(figsize=(4.2, 3.5)) # 略微调宽画板以容纳新柱子
        sns.barplot(data=df_main, x='Dataset', y='MAPE', hue='Model', palette='tab10', edgecolor='black')
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
    # 4. [新增] 导出总数据表
    export_unified_data_summary()  # <--- 这里调用新函数
    plot_robustness_analysis()
    print(f"\n{'=' * 60}\nALL FIGURES SAVED TO '{output_dir}/'\n{'=' * 60}")
    print("interpretation visualizers...")
    # 实际使用时需确保 models/best_model_xxx.pth 存在
    for ds in ["Malaysia", "ISO-NE", "Belgium"]:
        plot_causal_contribution(ds)
        plot_top_rules_forest(ds, top_n=50)
        plot_consequent_weight_boxplot(ds, top_n=8)