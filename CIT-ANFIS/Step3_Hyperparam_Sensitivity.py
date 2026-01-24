# -*- coding: utf-8 -*-
"""
Step3_Hyperparam_Sensitivity.py
----------------------------------------------
目标：分析超参数 n_estimators (规则数) 和 max_depth (规则复杂度) 对 CIT-ANFIS (Order=1) 性能的影响。
输出：hyperparam_sensitivity1.csv
"""

import torch
import pandas as pd
import numpy as np
import time
import sys
import random
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# 导入工具
try:
    if '.' not in sys.path:
        sys.path.append('.')
    from dataprepare import load_iso_ne, load_malaysia, load_north_american, enhance_features, split_dataset_by_paper
    from xganfis.model_lse import TreeANFIS
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)


# 固定随机种子
def setup_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_sensitivity_analysis(dataset_name, loader_func, device):
    print(f"\n{'=' * 60}")
    print(f"Hyperparam Sensitivity Analysis on: {dataset_name}")
    print(f"{'=' * 60}")

    # 1. 数据准备
    df = loader_func()
    if df is None: return []
    df_fe = enhance_features(df, dataset_name)

    # North American 使用真实数据 (不加噪声)
    if dataset_name == "North American":
        print(f"   [Info] Using REAL data for {dataset_name}.")

    train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_fe, dataset_name)
    feature_cols = [c for c in df_fe.columns if
                    c not in ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(train_df[feature_cols].values)
    y_train = scaler_y.fit_transform(train_df['Total Load'].values.reshape(-1, 1)).flatten()
    X_val = scaler_x.transform(ind_val_df[feature_cols].values)
    y_val = scaler_y.transform(ind_val_df['Total Load'].values.reshape(-1, 1)).flatten()
    X_test = scaler_x.transform(test_df[feature_cols].values)
    y_test = test_df['Total Load'].values

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 2. 定义参数网格
    # 你可以根据需要调整这里的列表
    n_estimators_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # 树的数量 (规则数)
    max_depth_list = [3,4,5,6,7,8]  # 树的深度 (前提复杂度)

    # 固定你的模型配置 (Order=1, Causal=True)
    fixed_order = 1
    fixed_causal = True

    results = []

    # 3. 网格搜索循环
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            config_name = f"Trees={n_est}, Depth={depth}"
            print(f"-> Testing: {config_name} ...")

            setup_seed(2024)  # 每次重置种子，保证公平对比
            t0 = time.time()

            model = TreeANFIS(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=0.01,
                order=fixed_order,  # 一阶模型
                use_causal=fixed_causal,  # 开启因果
                interaction_threshold=0.15  # Order=1时此参数无效，但不影响运行
            )

            try:
                # 训练流程
                model.identify_structure(X_train, y_train, feature_names=feature_cols)
                model = model.to(device)
                model.initialize_consequents(X_train_t, y_train_t)
                model.optimize_rule_base(X_train_t)
                model.train_neuro_fuzzy(
                    X_train_t, y_train_t,
                    X_val=X_val_t, y_val=y_val_t,
                    epochs=100,  # 保持一致
                    patience=10,
                    batch_size=2048
                )

                # 预测
                with torch.no_grad():
                    pred_s = model(X_test_t).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()

                epsilon = 1e-7
                mape = np.mean(np.abs((y_test - pred) / np.maximum(np.abs(y_test), epsilon))) * 100
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                # 记录结果
                results.append({
                    "Dataset": dataset_name,
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "MAPE": mape,
                    "RMSE": rmse,
                    "Time": time.time() - t0
                })
                print(f"   [Result] MAPE={mape:.3f}%, RMSE={rmse:.2f}")

            except Exception as e:
                print(f"   [Error] {e}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    all_res = []
    try:
        # 为了节省时间，你可以只跑 Malaysia 或 ISO-NE，或者全跑
        # 1. Malaysia
        all_res.extend(run_sensitivity_analysis("Malaysia", load_malaysia, device))
        # 2. ISO-NE
        # all_res.extend(run_sensitivity_analysis("ISO-NE", load_iso_ne, device))
        #3. North American
        #all_res.extend(run_sensitivity_analysis("North American", load_north_american, device))

    except Exception as e:
        print(e)

    if all_res:
        df_res = pd.DataFrame(all_res)
        print("\n" + "=" * 80)
        print("HYPERPARAMETER SENSITIVITY RESULTS")
        print("=" * 80)
        print(df_res.to_string(index=False))
        df_res.to_csv("hyperparam_sensitivity.csv", index=False)
        print("\n[Success] Results saved to 'hyperparam_sensitivity.csv'")