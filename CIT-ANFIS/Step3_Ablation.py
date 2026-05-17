# -*- coding: utf-8 -*-
import time
import sys
import os
import random
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

try:
    if '.' not in sys.path:
        sys.path.append('.')
    from dataprepare import load_iso_ne, load_malaysia, enhance_features, split_dataset_by_paper
    from CITanfis.model_lse import TreeANFIS
except ImportError as e:
    print(f"[Error] 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 0. 环境与目录准备
# ==========================================
if not os.path.exists('models/ablation'):
    os.makedirs('models/ablation')

def setup_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# 复用 Step 2 的 Belgium 加载逻辑
def load_belgium_data(filepath='data/ods001.csv'):
    if not os.path.exists(filepath): return None
    df = pd.read_csv(filepath, sep=';') if ';' in open(filepath).read(100) else pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_localize(None)
    target = 'Total Load'
    data = df[['Datetime', target]].resample('1h', on='Datetime').mean().interpolate().reset_index()
    return data[(data['Datetime'] >= '2024-01-01') & (data['Datetime'] <= '2025-12-31')].reset_index(drop=True)

def enhance_features_belgium(df):
    df = df.copy()
    target = 'Total Load'
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['Year'] = df['Datetime'].dt.year 
    df['Season'] = (df['Month'] % 12 + 3) // 3
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0) 
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if 7 <= x <= 22 else 0)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)  
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Lag_1_Hour'] = df[target].shift(1)
    df['Lag_2_Hour'] = df[target].shift(2)  
    df['Lag_24_Hour'] = df[target].shift(24)  
    df['Lag_1_Week'] = df[target].shift(24 * 7)  
    shifted_target = df[target].shift(1)
    df['Roll_Mean_24H'] = shifted_target.rolling(window=24).mean()
    df['EMA_12H'] = shifted_target.ewm(span=12, adjust=False).mean()  
    df['Diff_1H'] = df['Lag_1_Hour'] - df['Lag_2_Hour']  
    df.dropna(inplace=True)
    return df

def split_belgium_data(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:], df.iloc[train_end:val_end]

# ==========================================
# 1. 消融实验主逻辑 (含变体模型保存)
# ==========================================
def run_ablation(dataset_name, df_loader_func, device):
    print(f"\n{'=' * 50}\nStep 3: Ablation Study & Saving Variants on {dataset_name}\n{'=' * 50}")

    with open('result/optimal_params.json', 'r') as f:
        opt_cfg = json.load(f)
    
    params = opt_cfg.get(dataset_name, {"n_estimators": 15, "max_depth": 5})
    n_est = params["n_estimators"]
    depth = params["max_depth"]

    # 数据准备
    if dataset_name == "Belgium":
        df = load_belgium_data()
        df_fe = enhance_features_belgium(df)
        train_df, val_df, test_df, ind_val_df = split_belgium_data(df_fe)
    else:
        df = df_loader_func()
        df_fe = enhance_features(df, dataset_name)
        train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_fe, dataset_name)

    feat_cols = [c for c in df_fe.columns if c not in ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']]
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler_x.fit_transform(train_df[feat_cols].values)
    y_train = scaler_y.fit_transform(train_df['Total Load'].values.reshape(-1, 1)).flatten()
    X_val = scaler_x.transform(ind_val_df[feat_cols].values)
    y_val = scaler_y.transform(ind_val_df['Total Load'].values.reshape(-1, 1)).flatten()
    X_test = scaler_x.transform(test_df[feat_cols].values)
    y_test = test_df['Total Load'].values

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 注入 Full-Model (由 Step 2 完成，此处仅处理变体)
    results = []
    if "MAPE" in params:
        results.append({"Dataset": dataset_name, "Variant": "Full-Model", "MAPE": params["MAPE"], "RMSE": params["RMSE"], "MAE": params["MAE"], "R2": params["R2"]})

    variants = ['No-Causal', 'No-LSE', 'No-Order2', 'No-Pruning']

    for var in variants:
        print(f"   >> 运行变体: {var}")
        setup_seed(2024)
        t0 = time.time()

        p_order = 1 if var == 'No-Order2' else 2
        p_causal = False if var == 'No-Causal' else True
        p_lse = False if var == 'No-LSE' else True
        p_pruning = False if var == 'No-Pruning' else True

        model = TreeANFIS(n_estimators=n_est, max_depth=depth, order=p_order, use_causal=p_causal)
        model.identify_structure(X_train, y_train, feature_names=feat_cols)
        model = model.to(device)

        if p_lse: model.initialize_consequents(X_train_t, y_train_t)
        if p_pruning: model.optimize_rule_base(X_train_t)

        model.train_neuro_fuzzy(X_train_t, y_train_t, X_val=X_val_t, y_val=y_val_t, epochs=100, patience=10)

        # 保存消融变体模型
        var_save_path = f"models/ablation/{dataset_name}_{var}.pth"
        torch.save({
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'feature_names': feat_cols,
            'variant': var
        }, var_save_path)
        
        with torch.no_grad():
            pred_s = model(X_test_t).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
        
        results.append({
            "Dataset": dataset_name, 
            "Variant": var, 
            "MAPE": np.mean(np.abs((y_test - pred) / np.maximum(np.abs(y_test), 1e-7))) * 100,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred)
        })
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(2024)
    
    final_results = []
    datasets = [("Malaysia", load_malaysia), ("ISO-NE", load_iso_ne), ("Belgium", None)]
    
    for name, loader in datasets:
        final_results.extend(run_ablation(name, loader, device))
        
    if final_results:
        df_final = pd.DataFrame(final_results)
        df_final.to_csv("result/ablation_results.csv", index=False)
        print("\n[成功] 消融实验结果及模型文件已存至 result/ 和 models/ablation/")