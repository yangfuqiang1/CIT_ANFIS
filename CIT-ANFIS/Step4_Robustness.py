# -*- coding: utf-8 -*-
"""
Step4_Robustness.py
------------------
功能：基于保存的最优模型的温度鲁棒性测试 (Inference Robustness)
目标：验证已经训练好的最优模型在面对不同测试集噪声时的性能下降情况。
更新：
1. 直接加载 Step 2 中保存的最优模型权重 (.pth)。
2. 动态读取模型保存时的 feature_names，确保网络结构 (Rule 数量) 100% 严格对齐，解决 size mismatch！
3. 仅在测试阶段 (Inference) 引入噪声。
4. 噪声等级：[0, 5, 10, 15, 20] (0代表无噪声基线)。
"""

import time
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats, linalg
import warnings
warnings.filterwarnings('ignore')

# 导入依赖
try:
    from dataprepare import load_iso_ne, load_malaysia, enhance_features, split_dataset_by_paper
    from CITanfis.model_lse import TreeANFIS
    print("[System] Modules imported successfully.")
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

# ==========================================
# 0. 基础工具与因果算法 (用于重构模型结构)
# ==========================================
def setup_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class TruePCMCI:
    """重算因果权重，确保加载模型时网络结构(Tree分割)与训练时完全一致"""
    def __init__(self, alpha=0.05, max_cond_depth=1):
        self.alpha = alpha
        self.max_cond_depth = max_cond_depth

    def _partial_corr(self, x, y, Z_matrix):
        if Z_matrix is None or Z_matrix.shape[1] == 0:
            return stats.pearsonr(x, y)
        combined = np.column_stack([x, y, Z_matrix])
        try:
            cov = np.cov(combined, rowvar=False)
            prec = linalg.inv(cov)
            r_val = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
            r_val = np.clip(r_val, -0.9999, 0.9999)
            df = combined.shape[0] - Z_matrix.shape[1] - 2
            z = 0.5 * np.log((1 + r_val) / (1 - r_val))
            p_val = 2 * (1 - stats.norm.cdf(abs(z * np.sqrt(df))))
            return r_val, p_val
        except:
            return 0.0, 1.0

    def fit(self, X, y, feature_names):
        n_features = X.shape[1]
        initial_parents = []
        for i in range(n_features):
            _, p = stats.pearsonr(X[:, i], y)
            if p < self.alpha: initial_parents.append(i)
        
        final_weights = np.zeros(n_features)
        for idx in initial_parents:
            others = [p for p in initial_parents if p != idx]
            Z = X[:, others[:self.max_cond_depth]] if others else None
            r_mci, p_mci = self._partial_corr(X[:, idx], y, Z)
            if p_mci < self.alpha:
                final_weights[idx] = 1.0 + abs(r_mci)
        return final_weights

def calculate_metrics(y_true, y_pred):
    epsilon = 1e-7
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAPE": mape, "RMSE": rmse}

def inject_temperature_noise(df, noise_std):
    """仅向 DataFrame 注入温度噪声"""
    df_noisy = df.copy()
    temp_cols = [c for c in df.columns if 'emp' in c.lower() or 'drybulb' in c.lower()]
    if not temp_cols: return None

    for col in temp_cols:
        noise = np.random.normal(0, noise_std, len(df_noisy))
        df_noisy[col] = df_noisy[col] + noise
    return df_noisy


# ==========================================
# 1. 鲁棒性推理主逻辑
# ==========================================
def run_inference_robustness(dataset_name, df_loader_func, device):
    print(f"\n{'-' * 40}\n Model Robustness Inference: {dataset_name} \n{'-' * 40}")

    model_path = f"models/best_model_{dataset_name}.pth"
    if not os.path.exists(model_path):
        print(f"   [Error] 找不到模型文件 {model_path}，请确保已运行 Step 2。")
        return []

    # --- 1. 加载模型存档与核心配置 ---
    print(f"[*] Loading saved model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt['config']
    
    # 核心修复点：直接读取当时保存的确切特征名称列表，保证前后维度完全一致
    feature_cols = ckpt['feature_names'] 
    
    print(f"    Saved Config: Trees={config['n_estimators']}, Depth={config['max_depth']}")
    print(f"    Feature Dimensions Aligned: {len(feature_cols)} features.")

    df_raw = df_loader_func()
    if df_raw is None: return []

    # 基础特征工程与数据划分
    df_base = enhance_features(df_raw, dataset_name)
    train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_base, dataset_name)
    target_col = 'Total Load'

    # --- 2. 基于干净的 Train 数据拟合 Scaler ---
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # 严格使用 feature_cols 切片，避免混入无关列
    X_train_clean = scaler_x.fit_transform(train_df[feature_cols].values)
    y_train_clean = scaler_y.fit_transform(train_df[target_col].values.reshape(-1, 1)).flatten()
    
    y_test = test_df[target_col].values

    # --- 3. 重构模型结构并加载权重 ---
    pcmci = TruePCMCI(alpha=0.05, max_cond_depth=1)
    causal_weights = pcmci.fit(X_train_clean, y_train_clean, feature_cols)

    model = TreeANFIS(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        order=config['order'],
        learning_rate=0.01,
        use_causal=True
    )
    
    # 必须传入与之前完全相同的 feature_names 和尺寸一致的 X_train_clean
    model.identify_structure(X_train_clean, y_train_clean, feature_names=feature_cols, causal_weights=causal_weights)
    
    # 此时 size 绝对匹配，安全加载
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    print("    [Success] Model architecture reconstructed and weights loaded seamlessly.")

    # ==========================================
    # 4. 噪声评估阶段
    # ==========================================
    noise_levels = [0, 5, 10, 15, 20]
    results = []

    for sigma in noise_levels:
        if sigma == 0:
            test_df_noisy = test_df.copy()
            print(f"\n   >> [Noise Level] Sigma = {sigma} (Baseline)")
        else:
            test_df_noisy = inject_temperature_noise(test_df, sigma)
            if test_df_noisy is None:
                print("      [Warning] No temperature features found to inject noise.")
                return []
            print(f"\n   >> [Noise Level] Sigma = {sigma}")

        # 使用之前拟合好的干净 Scaler 对加噪数据进行标准化，并且严格按 feature_cols 提取
        X_test_noisy = scaler_x.transform(test_df_noisy[feature_cols].values)
        X_test_t = torch.tensor(X_test_noisy, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_s = model(X_test_t).cpu().numpy()
        
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()

        metrics = calculate_metrics(y_test, pred)
        print(f"      |-- MAPE: {metrics['MAPE']:.2f}%  |  RMSE: {metrics['RMSE']:.2f}")

        results.append({
            "Dataset": dataset_name,
            "Noise_Level": sigma,
            "MAPE": metrics['MAPE'],
            "RMSE": metrics['RMSE']
        })

    return results

if __name__ == "__main__":
    setup_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    all_results = []

    try:
        all_results.extend(run_inference_robustness("Malaysia", load_malaysia, device))
        all_results.extend(run_inference_robustness("ISO-NE", load_iso_ne, device))
    except Exception as e:
        print(f"Global Error: {e}")
        import traceback
        traceback.print_exc()

    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("FINAL INFERENCE ROBUSTNESS RESULTS")
        print("=" * 60)

        pivot = df_res.pivot_table(index='Dataset', columns='Noise_Level', values='MAPE')
        print(pivot)

        if not os.path.exists("result"):
            os.makedirs("result")
            
        df_res.to_csv("result/robustness_inference.csv", index=False)
        print("\nSaved to 'result/robustness_inference.csv'")