# -*- coding: utf-8 -*-
"""
Step3_Ablation_Robustness.py
------------------
功能：消融实验的鲁棒性测试 (Ablation Robustness)
目标：验证各个模块 (尤其是 Causal) 是否提升了模型的抗噪能力。
逻辑：
    在不同噪声等级下 (Sigma=0,5)，对比以下变体的性能衰减速度：
    1. Full-Model (基准)
    2. No-Causal (预期：抗噪能力最差)
    3. No-LSE
"""

import time
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# 导入依赖
try:
    from dataprepare import load_iso_ne, load_malaysia, enhance_features, split_dataset_by_paper
    from CITanfis.model_lse import TreeANFIS

    print("[System] Modules imported successfully.")
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)


# ==========================================
# 0. 基础工具
# ==========================================
def setup_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calculate_metrics(y_true, y_pred):
    epsilon = 1e-7
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAPE": mape, "RMSE": rmse}


def inject_temperature_noise(df, noise_std):
    """注入温度噪声"""
    df_noisy = df.copy()
    temp_cols = [c for c in df.columns if 'emp' in c.lower() or 'drybulb' in c.lower()]
    if not temp_cols: return None

    np.random.seed(42)  # 保证所有变体使用的是同一份加噪数据
    for col in temp_cols:
        noise = np.random.normal(0, noise_std, len(df_noisy))
        df_noisy[col] = df_noisy[col] + noise
    return df_noisy


# ==========================================
# 1. 消融鲁棒性测试主逻辑
# ==========================================
def run_ablation_robustness(dataset_name, df_loader_func, device):
    print(f"\n{'-' * 30} Ablation Robustness: {dataset_name} {'-' * 30}")

    # 1. 加载原始数据
    df_raw = df_loader_func()
    if df_raw is None: return []

    # 2. 基础特征工程
    df_base = enhance_features(df_raw, dataset_name)

    # 定义噪声等级
    noise_levels = [5,10,20]

    # 定义消融变体
    variants = ['Full-Model', 'No-Causal', 'No-LSE']

    results = []

    # 设定超参数 (固定住，控制变量)
    if dataset_name == "Malaysia":
        base_params = {"n_estimators": 20, "max_depth": 6}
    else:  # ISO-NE
        base_params = {"n_estimators": 20, "max_depth": 6}

    # --- 循环 A: 噪声等级 ---
    for sigma in noise_levels:
        print(f"\n   >> [Noise Level] Sigma = {sigma}")

        # 注入噪声 (特征层面的干扰)
        df_curr = inject_temperature_noise(df_base, sigma)
        if df_curr is None:
            print("      Skipping (No Temperature data).")
            return []

        # 数据切分
        train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_curr, dataset_name)

        exclude_cols = ['Datetime', 'Total Load', 'date', 'demand', 'Season']
        feature_cols = [c for c in df_curr.columns if c not in exclude_cols and c != 'Total Load']
        target_col = 'Total Load'

        # 标准化
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_x.fit_transform(train_df[feature_cols].values)
        y_train = scaler_y.fit_transform(train_df[target_col].values.reshape(-1, 1)).flatten()
        X_val = scaler_x.transform(ind_val_df[feature_cols].values)
        y_val = scaler_y.transform(ind_val_df[target_col].values.reshape(-1, 1)).flatten()
        X_test = scaler_x.transform(test_df[feature_cols].values)
        y_test = test_df[target_col].values

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        # --- 循环 B: 消融变体 ---
        for variant in variants:
            # 配置开关
            p_causal = True
            p_lse = True

            if variant == 'No-Causal':
                p_causal = False
            elif variant == 'No-LSE':
                p_lse = False

            # 初始化模型
            model = TreeANFIS(
                n_estimators=base_params['n_estimators'],
                max_depth=base_params['max_depth'],
                learning_rate=0.01,
                order=2,
                use_causal=p_causal,  # 变量 1
                interaction_threshold=0.05
            )

            try:
                # 1. 结构辨识
                model.identify_structure(X_train, y_train, feature_names=feature_cols)
                model = model.to(device)

                # 2. LSE 初始化 (变量 2)
                if p_lse:
                    model.initialize_consequents(X_train_t, y_train_t)


                # 4. BP 微调 (稍微加速，epoch减少)
                model.train_neuro_fuzzy(
                    X_train_t, y_train_t,
                    X_val=X_val_t, y_val=y_val_t,
                    epochs=60, batch_size=2048, patience=8  # 静默模式
                )

                # 5. 评估
                model.eval()
                with torch.no_grad():
                    pred_s = model(X_test_t).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()

                metrics = calculate_metrics(y_test, pred)

                # 打印当前结果
                print(f"      |-- {variant:<12} MAPE: {metrics['MAPE']:.2f}%")

                results.append({
                    "Dataset": dataset_name,
                    "Noise": sigma,
                    "Variant": variant,
                    "MAPE": metrics['MAPE']
                })

            except Exception as e:
                print(f"      [Error] {variant} Failed: {e}")

    return results


if __name__ == "__main__":
    setup_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    all_results = []

    try:
        # 只跑 Malaysia 和 ISO-NE (因为有温度数据)
        all_results.extend(run_ablation_robustness("Malaysia", load_malaysia, device))
        all_results.extend(run_ablation_robustness("ISO-NE", load_iso_ne, device))

    except Exception as e:
        print(f"Global Error: {e}")
        import traceback

        traceback.print_exc()

    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("FINAL ABLATION ROBUSTNESS RESULTS")
        print("=" * 60)

        # 打印透视表
        pivot = df_res.pivot_table(index=['Dataset', 'Noise'], columns='Variant', values='MAPE')
        print(pivot)

        df_res.to_csv("ablation_robustness.csv", index=False)
        print("\nSaved to 'ablation_robustness.csv'")
