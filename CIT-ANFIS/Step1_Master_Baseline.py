# -*- coding: utf-8 -*-
"""
Step1_Master_Baseline.py
------------------
[Final Manual Config Version]
Features:
1. DL Models: Imports Pro versions (Bi-LSTM, Transformer-PE, ResNet-MLP, SiLU-KAN) from dl_models.
2. Bit-level Reproducibility: Full seeding.
3. Smart Pruning: Adaptive Threshold.
4. CIT-ANFIS: Locked to Order=1.
"""

import time
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from tqdm import tqdm

# ==========================================
# 0. 导入依赖
# ==========================================
warnings.filterwarnings('ignore')

print(f"{'=' * 60}")
print("INITIALIZING BENCHMARK SUITE (Pro DL Models)")
print(f"{'=' * 60}")

# 1. 导入数据工具
try:
    from dataprepare import load_iso_ne, load_malaysia, load_north_american, enhance_features, split_dataset_by_paper
    print("[System] Data loaders imported successfully.")
except ImportError:
    print("[Critical Error] Cannot find 'dataprepare.py'.")
    sys.exit(1)

# 2. 导入 CIT-ANFIS 库
try:
    from tanfis_lib.model import FirstTSK
    from xganfis.model_lse import TreeANFIS
    print("[System] ANFIS Models imported successfully.")
except ImportError as e:
    print(f"\n[Critical Error] Model import failed: {e}")
    sys.exit(1)

# 3. 导入增强版深度学习模型 (Pro Versions)
try:
    # [FIX] 确保 dl_models 文件夹存在，且里面有 models.py
    from dl_models.model import LSTMModel, TransformerModel, DeepMLP, SimpleKAN
    print("[System] All Pro DL Models (Bi-LSTM, Trans-PE, DeepMLP, SiLU-KAN) imported.")
except ImportError as e:
    print(f"[Critical Error] Cannot find 'dl_models/model.py': {e}")
    sys.exit(1)



def setup_seed(seed=2024):
    """
    全方位锁定随机种子，确保 PyTorch 模型比特级复现
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"[System] Random Seed set to {seed} (Deterministic Mode On)")


def load_belgium_data(filepath='ods001.csv'):
    """
    读取比利时电网数据 (2020-2025) - 增强健壮性版
    """
    print(f"Loading Belgium Data from {filepath}...")

    if not os.path.exists(filepath):
        print(f"[Error] File '{filepath}' not found!")
        sys.exit(1)

    try:
        # 优先尝试分号 (ODS 常见格式)
        df = pd.read_csv(filepath, sep=';')
        if df.shape[1] < 2:  # 如果分号解析失败
            df = pd.read_csv(filepath, sep=',')
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        sys.exit(1)

    # [Fix 1] 去除列名空格
    df.columns = df.columns.str.strip()

    # 检查是否有时间列
    if 'Datetime' not in df.columns:
        print(f"[Error] 'Datetime' column missing. Found: {df.columns.tolist()}")
        sys.exit(1)

    # [Fix 2] 解析时间并转换为布鲁塞尔时间
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Europe/Brussels')
    except:
        # 如果转换失败（比如系统不支持），退回 UTC
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

    # 排序
    df = df.sort_values('Datetime').reset_index(drop=True)

    # 检查目标列
    target_col = 'Total Load'
    if target_col not in df.columns:
        # 尝试模糊匹配，比如 'Load', 'GridLoad' 等
        candidates = [c for c in df.columns if 'Load' in c]
        if candidates:
            print(f"[Warning] '{target_col}' not found. Using '{candidates[0]}' instead.")
            target_col = candidates[0]
        else:
            print(f"[Error] Column '{target_col}' not found. Available: {df.columns.tolist()}")
            sys.exit(1)

    data = df[['Datetime', target_col]].copy()

    # 重命名统一方便后续处理
    data.columns = ['Datetime', 'Total Load']

    # 重采样为 1小时
    print(f"   Original shape: {data.shape} (Raw)")
    data.set_index('Datetime', inplace=True)
    # [Fix 3] 既然有夏令时，重采样后去掉时区信息，方便计算
    data = data.resample('1h').mean().interpolate()
    data.reset_index(inplace=True)

    # 去除时区信息 (变成 Naive Time)，防止后续 pytorch 转换报错
    data['Datetime'] = data['Datetime'].dt.tz_localize(None)

    print(f"   Resampled shape: {data.shape} (1-Hour)")

    # 筛选2024-2025年的数据
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-12-31')
    data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)].copy()
    data = data.reset_index(drop=True)

    print(f"   Filtered shape: {data.shape} (2024-2025)")

    return data


def enhance_features_belgium(df):
    """
    比利时数据特征工程 (终极增强版)
    针对无温度场景，大幅增强时序特征挖掘。
    """
    df = df.copy()
    target = 'Total Load'

    # ==========================================
    # 1. 基础日历特征 (Calendar)
    # ==========================================
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['Year'] = df['Datetime'].dt.year  # 捕捉长期增长趋势

    # 季度与周末标识
    df['Season'] = (df['Month'] % 12 + 3) // 3
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 周末为1，工作日为0

    # 将一天划分为更细的时间段 (Peak/Off-Peak)
    # 假设 7-22点是高峰期 (根据比利时工业习惯可微调)
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if 7 <= x <= 22 else 0)

    # ==========================================
    # 2. 周期性编码 (Cyclical Encoding)
    # ==========================================
    # 保持原有的，这是深度学习捕捉周期的神器
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)  # 增加年周期
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    # ==========================================
    # 3. 增强型滞后特征 (Advanced Lags)
    # ==========================================
    # A. 短期记忆 (最近几小时)
    df['Lag_1_Hour'] = df[target].shift(1)
    df['Lag_2_Hour'] = df[target].shift(2)  # 新增
    df['Lag_3_Hour'] = df[target].shift(3)  # 新增

    # B. 中期记忆 (昨天/前天)
    df['Lag_24_Hour'] = df[target].shift(24)  # 昨天同一时刻
    df['Lag_25_Hour'] = df[target].shift(25)  # 昨天前一小时 (捕捉趋势)
    df['Lag_48_Hour'] = df[target].shift(48)  # 前天同一时刻 (新增)

    df['Lag_1_Week'] = df[target].shift(24 * 7)  # 上周


    # ==========================================
    # 4. 滚动统计特征 (Rolling Window Statistics)
    # ==========================================

    shifted_target = df[target].shift(1)

    # A. 过去 24 小时 (日级别统计)
    roll_24 = shifted_target.rolling(window=24)
    df['Roll_Mean_24H'] = roll_24.mean()
    df['Roll_Std_24H'] = roll_24.std()  
    df['Roll_Max_24H'] = roll_24.max()
    df['Roll_Min_24H'] = roll_24.min()

    # B. 过去 7 天 (周级别统计) - 新增
    # 捕捉更长期的基准线
    roll_168 = shifted_target.rolling(window=24 * 7)
    df['Roll_Mean_7D'] = roll_168.mean()
    df['Roll_Std_7D'] = roll_168.std()  # 周波动率

    # C. 指数移动平均 (EMA) - 对近期数据权重更高
    df['EMA_12H'] = shifted_target.ewm(span=12, adjust=False).mean()  # 半日趋势
    df['EMA_168H'] = shifted_target.ewm(span=168, adjust=False).mean()  # 周趋势

    # ==========================================
    # 5. 差分与变化率特征 (Difference & Rate) - 新增
    # ==========================================
    # 捕捉负荷是在“爬坡”还是“下降”
    # 当前时刻预测值无法获知，但我们可以知道“上一小时的变化量”
    df['Diff_1H'] = df['Lag_1_Hour'] - df['Lag_2_Hour']  # 最近一小时的变化
    df['Diff_24H'] = df['Lag_1_Hour'] - df['Lag_24_Hour']  # 与昨天相比的变化

    # ==========================================
    # 6. 数据清洗
    # ==========================================
    # 鉴于你有 10 年数据，丢弃第一年作为“预热期”是完全划算的
    df.dropna(inplace=True)

    return df


def split_belgium_data(df):
    """
    按时间顺序切分：70% 训练, 15% 验证, 15% 测试
    """
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df, val_df
# ==========================================
# 1. 统一训练器 (带 Generator)
# ==========================================

def train_torch_model(model, X_train, y_train, X_val, y_val, device, epochs=100, lr=0.001, batch_size=2048, name="Model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 锁定 DataLoader 的随机性
    g = torch.Generator()
    g.manual_seed(2024)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    best_loss = float('inf')
    patience = 12
    no_imp = 0
    best_weights = None

    pbar = tqdm(range(epochs), desc=f"Training {name}", leave=False, unit="ep")

    for ep in pbar:
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        pbar.set_postfix({'val_mse': f'{val_loss:.4f}'})

        if val_loss < best_loss:
            best_loss = val_loss
            no_imp = 0
            best_weights = model.state_dict()
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def calculate_metrics(y_true, y_pred, time_taken):
    epsilon = 1e-7
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAPE": mape, "RMSE": rmse, "MAE": mae, "R2": r2, "Time(s)": time_taken}


# ==========================================
# 2. 主实验循环 (修复版)
# ==========================================

def run_experiment(dataset_name, df_loader_func, device):
    print(f"\n{'-' * 30} Processing {dataset_name} {'-' * 30}")

    # 超参数配置
    HYPERPARAMS = {
        "Malaysia": {"n_estimators": 10, "max_depth": 6},
        "ISO-NE": {"n_estimators": 20, "max_depth": 6},
        "Belgium": {"n_estimators": 5, "max_depth": 8}
    }
    current_params = HYPERPARAMS.get(dataset_name, {"n_estimators": 15, "max_depth": 6})

    # === [关键修复] 数据加载逻辑分支 ===
    # 必须确保这里的字符串 "Belgium" 和你底部调用时传入的一模一样
    if dataset_name == "Belgium":
        # 专门处理 Belgium 数据，不使用传入的 None
        df = load_belgium_data('ods001.csv')
        if df is None: return []

        # 使用 Belgium 专属的特征工程 (无温度)
        df_fe = enhance_features_belgium(df)

        # 使用 Belgium 专属的数据切分
        train_df, val_df, test_df, ind_val_df = split_belgium_data(df_fe)

        # 排除非特征列
        exclude = ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']
        feature_cols = [c for c in df_fe.columns if c not in exclude]

    else:
        # === 其他数据集的逻辑 (防止报错) ===
        if df_loader_func is None:
            print(f"[Error] No loader function provided for {dataset_name}!")
            return []

        df = df_loader_func()
        if df is None: return []

        df_fe = enhance_features(df, dataset_name)
        train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_fe, dataset_name)
        feature_cols = [c for c in df_fe.columns if
                        c not in ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']]

    target_col = 'Total Load'
    print(f"Features Used: {feature_cols}")

    # --- 以下标准化和转换代码保持不变 ---
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(train_df[feature_cols].values)
    y_train = scaler_y.fit_transform(train_df[target_col].values.reshape(-1, 1)).flatten()
    X_val = scaler_x.transform(ind_val_df[feature_cols].values)
    y_val = scaler_y.transform(ind_val_df[target_col].values.reshape(-1, 1)).flatten()
    X_test = scaler_x.transform(test_df[feature_cols].values)
    y_test = test_df[target_col].values

    # Tensor 转换
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    predictions_data = {'Actual': y_test}
    results_list = []

    #--- [从这里开始往下是模型训练代码，保持原样即可] ---
    #1. SVR
    print("1. Training SVR...")
    t0 = time.time()
    model_svr = SVR(kernel='rbf', C=100)
    if len(X_train) > 20000:
        idx = np.random.choice(len(X_train), 20000, replace=False)
        model_svr.fit(X_train[idx], y_train[idx])
    else:
        model_svr.fit(X_train, y_train)
    pred = scaler_y.inverse_transform(model_svr.predict(X_test).reshape(-1, 1)).flatten()
    predictions_data['SVR'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'SVR';
    results_list[-1]['Dataset'] = dataset_name

    # 2. RandomForest
    print("2. Training Random Forest...")
    t0 = time.time()
    model_rf = RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    model_rf.fit(X_train, y_train)
    pred = scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1, 1)).flatten()
    predictions_data['RandomForest'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'RandomForest';
    results_list[-1]['Dataset'] = dataset_name

    # 3. XGBoost
    print("3. Training XGBoost...")
    t0 = time.time()
    model_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)
    model_xgb.fit(X_train, y_train)
    pred = scaler_y.inverse_transform(model_xgb.predict(X_test).reshape(-1, 1)).flatten()
    predictions_data['XGBoost'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'XGBoost';
    results_list[-1]['Dataset'] = dataset_name

    # 4. Bi-LSTM
    print("4. Training LSTM ...")
    t0 = time.time()
    model_lstm = train_torch_model(LSTMModel(X_train.shape[1]), X_train_t, y_train_t, X_val_t, y_val_t, device,
                                   name="Bi-LSTM")
    model_lstm.eval()
    with torch.no_grad():
        pred_s = model_lstm(X_test_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    predictions_data['LSTM'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'LSTM';
    results_list[-1]['Dataset'] = dataset_name

    # 5. Transformer
    print("5. Training Transformer...")
    t0 = time.time()
    model_trans = train_torch_model(TransformerModel(X_train.shape[1]), X_train_t, y_train_t, X_val_t, y_val_t, device,
                                    name="Transformer")
    model_trans.eval()
    with torch.no_grad():
        pred_s = model_trans(X_test_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    predictions_data['Transformer'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'Transformer';
    results_list[-1]['Dataset'] = dataset_name

    # 6. Deep MLP
    print("6. Training MLP (ResNet)...")
    t0 = time.time()
    model_mlp = train_torch_model(DeepMLP(X_train.shape[1]), X_train_t, y_train_t, X_val_t, y_val_t, device,
                                  name="DeepMLP")
    model_mlp.eval()
    with torch.no_grad():
        pred_s = model_mlp(X_test_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    predictions_data['MLP'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'MLP';
    results_list[-1]['Dataset'] = dataset_name

    # 7. KAN
    print("7. Training KAN (SiLU)...")
    t0 = time.time()
    model_kan = train_torch_model(SimpleKAN(X_train.shape[1]), X_train_t, y_train_t, X_val_t, y_val_t, device,
                                  name="KAN")
    model_kan.eval()
    with torch.no_grad():
        pred_s = model_kan(X_test_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    predictions_data['KAN'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'KAN';
    results_list[-1]['Dataset'] = dataset_name

    # 8. Standard ANFIS
    print("8. Training Standard ANFIS (Baseline)...")
    t0 = time.time()
    model_std = FirstTSK(in_dim=X_train.shape[1], out_dim=1, num_fuzzy_set=2, mf='Gaussian', tnorm='prod').to(
        device).double()
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=0.05)
    X_train_d, y_train_d = X_train_t.double(), y_train_t.view(-1, 1).double()
    X_val_d, y_val_d = X_val_t.double(), y_val_t.view(-1, 1).double()
    try:
        model_std.est_con_param(X_train_d, y_train_d)
    except:
        pass

    best_loss_std = float('inf')
    no_imp_std = 0
    for _ in tqdm(range(30), desc="Training Std-ANFIS", leave=False):
        model_std.train()
        optimizer_std.zero_grad()
        out, _ = model_std(X_train_d)
        loss = F.mse_loss(out, y_train_d)
        loss.backward()
        optimizer_std.step()
        model_std.eval()
        with torch.no_grad():
            val_out, _ = model_std(X_val_d)
            val_loss = F.mse_loss(val_out, y_val_d).item()
        if val_loss < best_loss_std:
            best_loss_std = val_loss
            no_imp_std = 0
        else:
            no_imp_std += 1
            if no_imp_std >= 5: break

    with torch.no_grad():
        pred_s, _ = model_std(X_test_t.double())
    pred = scaler_y.inverse_transform(pred_s.cpu().numpy().reshape(-1, 1)).flatten()
    predictions_data['Standard_ANFIS'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'ANFIS';
    results_list[-1]['Dataset'] = dataset_name

    # 9. CIT-ANFIS
    print("9. Training CIT-ANFIS (Proposed)...")
    t0 = time.time()
    n_est = current_params['n_estimators']
    max_d = current_params['max_depth']
    print(f"   -> Using Params: n_estimators={n_est}, max_depth={max_d}")

    model_cit = TreeANFIS(
        n_estimators=n_est,
        max_depth=max_d,
        learning_rate=0.1,
        order=2,
        use_causal=True,
        interaction_threshold=0.05
    )
    model_cit.identify_structure(X_train, y_train, feature_names=feature_cols)
    model_cit = model_cit.to(device)
    model_cit.initialize_consequents(X_train_t, y_train_t)

    n_rules = model_cit.rule_feat_idxs.shape[0]
    if n_rules > 0:
        adaptive_thresh = (1.0 / n_rules) * 0.1
        model_cit.optimize_rule_base(X_train_t, threshold=adaptive_thresh)

    model_cit.train_neuro_fuzzy(
        X_train_t, y_train_t,
        X_val=X_val_t, y_val=y_val_t,
        epochs=150,
        batch_size=2048,
        patience=15
    )

    with torch.no_grad():
        pred_s = model_cit(X_test_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    predictions_data['CIT_ANFIS'] = pred
    results_list.append(calculate_metrics(y_test, pred, time.time() - t0))
    results_list[-1]['Model'] = 'CIT-ANFIS';
    results_list[-1]['Dataset'] = dataset_name

    print(f"Finished {dataset_name}. CIT-ANFIS MAPE: {results_list[-1]['MAPE']:.2f}%")
    df_preds = pd.DataFrame(predictions_data)
    df_preds.to_csv(f"predictions_{dataset_name}.csv", index=False)

    return results_list


if __name__ == "__main__":
    setup_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}\n")

    all_results = []
    try:
        all_results.extend(run_experiment("Malaysia", load_malaysia, device))
        all_results.extend(run_experiment("ISO-NE", load_iso_ne, device))
        all_results.extend(run_experiment("Belgium", None, device))
    except Exception as e:
        import traceback
        traceback.print_exc()

    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("FINAL BENCHMARK RESULTS")
        print("=" * 80)
        print(df_res.to_string(index=False))
        df_res.to_csv("benchmark_results_final.csv", index=False)