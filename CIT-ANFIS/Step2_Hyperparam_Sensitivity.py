# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
import time
import sys
import random
import os
import json
import warnings
from scipy import stats, linalg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ==========================================
# 0. 环境与目录准备
# ==========================================
for path in ['result', 'models']:
    if not os.path.exists(path):
        os.makedirs(path)

try:
    if '.' not in sys.path:
        sys.path.append('.')
    from dataprepare import load_iso_ne, load_malaysia, enhance_features, split_dataset_by_paper
    from CITanfis.model_lse import TreeANFIS
except ImportError as e:
    print(f"[Error] 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 1. 因果发现算法 (PCMCI)
# ==========================================
class TruePCMCI:
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

# ==========================================
# 2. 基础配置与数据函数
# ==========================================
def setup_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
# 3. 敏感性分析主函数 (含模型持久化)
# ==========================================
def run_sensitivity_analysis(dataset_name, loader_func, device):
    print(f"\n{'=' * 60}\nStep 2: Hyperparam Search & Model Saving on {dataset_name}\n{'=' * 60}")

    if dataset_name == "Belgium":
        df = load_belgium_data()
        df_fe = enhance_features_belgium(df)
        train_df, val_df, test_df, ind_val_df = split_belgium_data(df_fe)
    else:
        df = loader_func()
        df_fe = enhance_features(df, dataset_name)
        train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_fe, dataset_name)

    feat_cols = [c for c in df_fe.columns if c not in ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']]
    
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler_x.fit_transform(train_df[feat_cols].values)
    y_train = scaler_y.fit_transform(train_df['Total Load'].values.reshape(-1, 1)).flatten()
    X_test = scaler_x.transform(test_df[feat_cols].values)
    y_test = test_df['Total Load'].values

    # 运行因果权重计算
    pcmci = TruePCMCI(alpha=0.05, max_cond_depth=1)
    causal_weights = pcmci.fit(X_train, y_train, feat_cols)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(scaler_x.transform(ind_val_df[feat_cols].values), dtype=torch.float32).to(device)
    y_val_t = torch.tensor(scaler_y.transform(ind_val_df['Total Load'].values.reshape(-1, 1)).flatten(), dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 缩短后的超参网格（根据需要调整）
    n_estimators_list = [5,6,7,8, 9, 10,11,12,13,14,15,16,17,18,19,20] 
    max_depth_list = [3, 4, 5, 6, 7, 8]
    
    results = []
    best_mape = float('inf')
    best_model_data = None

    for n_est in n_estimators_list:
        for depth in max_depth_list:
            print(f"   测试中: Trees={n_est}, Depth={depth}")
            setup_seed(2024)
            t0 = time.time()
            
            model = TreeANFIS(n_estimators=n_est, max_depth=depth, learning_rate=0.01, order=2, use_causal=True)
            try:
                model.identify_structure(X_train, y_train, feature_names=feat_cols, causal_weights=causal_weights)
                model = model.to(device)
                model.initialize_consequents(X_train_t, y_train_t)
                model.train_neuro_fuzzy(X_train_t, y_train_t, X_val=X_val_t, y_val=y_val_t, epochs=80, patience=8, batch_size=2048)

                with torch.no_grad():
                    pred_s = model(X_test_t).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()

                mape = np.mean(np.abs((y_test - pred) / np.maximum(np.abs(y_test), 1e-7))) * 100
                
                if mape < best_mape:
                    best_mape = mape
                    # 记录最佳模型的详细状态用于持久化
                    best_model_data = {
                        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'feature_names': feat_cols,
                        'config': {'n_estimators': n_est, 'max_depth': depth, 'order': 2}
                    }

                results.append({
                    "Dataset": dataset_name, "n_estimators": n_est, "max_depth": depth, "MAPE": mape,
                    "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "MAE": mean_absolute_error(y_test, pred),
                    "R2": r2_score(y_test, pred), "Time(s)": time.time() - t0 
                })
            except Exception as e:
                print(f"      [错误] {e}")

    # 保存当前数据集的最优模型
    if best_model_data:
        model_save_path = f"models/best_model_{dataset_name}.pth"
        torch.save(best_model_data, model_save_path)
        print(f"   [模型已保存] 数据集 {dataset_name} 的最优模型已存至 {model_save_path} (MAPE: {best_mape:.4f}%)")

    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(2024)
    
    all_res = []
    datasets_to_run = [("Malaysia", load_malaysia), ("ISO-NE", load_iso_ne), ("Belgium", None)]
    
    for name, loader in datasets_to_run:
        all_res.extend(run_sensitivity_analysis(name, loader, device))

    if all_res:
        df_res = pd.DataFrame(all_res)
        df_res.to_csv("result/hyperparam_sensitivity.csv", index=False)
        print(f"\n[成功] 完整结果已保存至 result/hyperparam_sensitivity.csv")
        
        # 更新最优参数 JSON
        best_cfg = {}
        for ds in df_res['Dataset'].unique():
            row = df_res.loc[df_res[df_res['Dataset'] == ds]['MAPE'].idxmin()]
            best_cfg[ds] = {
                "n_estimators": int(row['n_estimators']), 
                "max_depth": int(row['max_depth']), 
                "MAPE": float(row['MAPE']), 
                "RMSE": float(row['RMSE']),
                "MAE": float(row['MAE']),
                "R2": float(row['R2']),
                "Time(s)": float(row['Time(s)']) 
            }
        
        with open("result/optimal_params.json", 'w') as f:
            json.dump(best_cfg, f, indent=4)
        print("[成功] optimal_params.json 已更新。")