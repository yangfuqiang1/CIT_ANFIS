# -*- coding: utf-8 -*-
"""
Step2_Ablation.py
------------------
功能：消融实验 (Ablation Study) - 验证 CIT-ANFIS 核心模块的有效性
更新：
1. 数据集适配：完全集成 Belgium (ods001.csv) 的加载与清洗逻辑。
2. 参数对齐：针对 Belgium 使用 n_estimators=15, max_depth=6 (与 Step 1 保持一致)。
3. 模块验证：对比 Full-Model, No-Causal, No-LSE, No-Pruning。
"""

import time
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 导入自定义模块
try:
    from dataprepare import load_iso_ne, load_malaysia, enhance_features, split_dataset_by_paper
    from xganfis.model_lse import TreeANFIS

    print("[System] Modules imported successfully.")
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)


# ==========================================
# 0. 随机种子锁定 (Bit-level Reproducibility)
# ==========================================
def setup_seed(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"[System] Random Seed set to {seed}")


# ==========================================
# 1. Belgium 数据处理模块 (与 Step 1 完全一致)
# ==========================================
def load_belgium_data(filepath='ods001.csv'):
    """
    读取比利时电网数据 (2020-2025) - 增强健壮性版
    """
    print(f"Loading Belgium Data from {filepath}...")

    if not os.path.exists(filepath):
        print(f"[Error] File '{filepath}' not found!")
        sys.exit(1)

    try:
        df = pd.read_csv(filepath, sep=';')
        if df.shape[1] < 2:
            df = pd.read_csv(filepath, sep=',')
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        sys.exit(1)

    # 去除列名空格
    df.columns = df.columns.str.strip()

    if 'Datetime' not in df.columns:
        print(f"[Error] 'Datetime' column missing. Found: {df.columns.tolist()}")
        sys.exit(1)

    # 解析时间并处理时区
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Europe/Brussels')
    except:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

    df = df.sort_values('Datetime').reset_index(drop=True)

    target_col = 'Total Load'
    if target_col not in df.columns:
        candidates = [c for c in df.columns if 'Load' in c]
        if candidates:
            target_col = candidates[0]
        else:
            sys.exit(1)

    data = df[['Datetime', target_col]].copy()
    data.columns = ['Datetime', 'Total Load']

    # 重采样为 1小时
    data.set_index('Datetime', inplace=True)
    data = data.resample('1h').mean().interpolate()
    data.reset_index(inplace=True)

    # 去除时区信息
    data['Datetime'] = data['Datetime'].dt.tz_localize(None)

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

    # C. 长期记忆 (周/月/年) - 既然有10年数据，这部分极其重要！
    df['Lag_1_Week'] = df[target].shift(24 * 7)  # 上周
    df['Lag_2_Week'] = df[target].shift(24 * 7 * 2)  # 两周前 (新增)
    df['Lag_4_Week'] = df[target].shift(24 * 7 * 4)  # 上个月

    # [关键新增] 年同比特征 (Year-over-Year)
    # 比较今年和去年的同一时刻（使用 52 周近似一年，以此对齐星期几）
    df['Lag_52_Week'] = df[target].shift(24 * 7 * 52)

    # ==========================================
    # 4. 滚动统计特征 (Rolling Window Statistics)
    # ==========================================
    # 计算时必须 shift(1) 以防数据泄露 (Data Leakage)
    shifted_target = df[target].shift(1)

    # A. 过去 24 小时 (日级别统计)
    roll_24 = shifted_target.rolling(window=24)
    df['Roll_Mean_24H'] = roll_24.mean()
    df['Roll_Std_24H'] = roll_24.std()  # 新增：捕捉波动率
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
    # 由于引入了 Lag_52_Week (约1年)，前一年的数据会变成 NaN
    # 鉴于你有 10 年数据，丢弃第一年作为“预热期”是完全划算的
    df.dropna(inplace=True)

    return df


def split_belgium_data(df):
    """70% Train, 15% Val, 15% Test"""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:], df.iloc[train_end:val_end]


def calculate_metrics(y_true, y_pred, time_taken):
    epsilon = 1e-7
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAPE": mape, "RMSE": rmse, "MAE": mae, "R2": r2, "Time(s)": time_taken}


# ==========================================
# 2. 消融实验主逻辑
# ==========================================

def run_ablation(dataset_name, df_loader_func, device):
    print(f"\n{'=' * 40}")
    print(f"Processing Ablation: {dataset_name}")
    print(f"{'=' * 40}")

    # 1. 数据准备
    if dataset_name == "Belgium":
        df = load_belgium_data('ods001.csv')
        if df is None: return []
        df_fe = enhance_features_belgium(df)
        train_df, val_df, test_df, ind_val_df = split_belgium_data(df_fe)
        exclude = ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']
        feature_cols = [c for c in df_fe.columns if c not in exclude]
        # 针对 Belgium 的参数配置 (与 Step 1 一致)
        n_est_cfg = 30
        max_d_cfg = 6
    else:
        # 兼容旧数据集
        df = df_loader_func()
        if df is None: return []
        df_fe = enhance_features(df, dataset_name)
        train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_fe, dataset_name)
        feature_cols = [c for c in df_fe.columns if
                        c not in ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']]
        n_est_cfg = 20
        max_d_cfg = 6

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

    # 转 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    results = []

    # ==========================================
    # 定义消融变体
    # ==========================================
    variants = ['Full-Model', 'No-Causal', 'No-LSE']

    for variant in variants:
        print(f"\n   >> [Variant] Running: {variant}")
        t0 = time.time()

        # --- 1. 配置开关 ---
        p_order = 2  # 始终为 1
        p_causal = True
        p_lse = True

        if variant == 'No-Causal':
            p_causal = False
        elif variant == 'No-LSE':
            p_lse = False


        # --- 2. 初始化模型 ---
        model = TreeANFIS(
            n_estimators=n_est_cfg,
            max_depth=max_d_cfg,
            learning_rate=0.01,
            order=p_order,
            use_causal=p_causal,
            interaction_threshold=0.05
        )

        # --- 3. 结构辨识 ---
        try:
            model.identify_structure(X_train, y_train, feature_names=feature_cols)
        except Exception as e:
            print(f"      [Error] Structure ID failed: {e}")
            continue

        model = model.to(device)

        # --- 4. 参数初始化 (LSE 消融点) ---
        if p_lse:
            # 正常使用 LSE
            model.initialize_consequents(X_train_t, y_train_t)
        else:
            print("      [Ablation] Skipping LSE. Using random weights.")
            # 即使跳过 LSE，TreeANFIS 类内部通常也会有随机初始化，
            # 这里只需确保不调用 initialize_consequents 即可。
            # 或者可以显式地重新随机化后件参数 (视 model_lse 实现而定)



        # --- 6. 训练 (BP 微调) ---
        # 保持与 Step 1 一致的训练参数
        model.train_neuro_fuzzy(
            X_train_t, y_train_t,
            X_val=X_val_t, y_val=y_val_t,
            epochs=150,
            batch_size=2048,
            patience=15
        )

        # --- 7. 预测与评估 ---
        model.eval()
        with torch.no_grad():
            pred_s = model(X_test_t).cpu().numpy()

        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()

        metrics = calculate_metrics(y_test, pred, time.time() - t0)
        metrics['Dataset'] = dataset_name
        metrics['Variant'] = variant
        results.append(metrics)

        print(f"      -> {variant} Result: MAPE={metrics['MAPE']:.2f}%, RMSE={metrics['RMSE']:.2f}")

    return results


if __name__ == "__main__":
    setup_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    all_results = []

    try:
        # 主要运行 Belgium 数据集
        all_results.extend(run_ablation("Belgium", None, device))
        all_results.extend(run_ablation("ISO-NE", load_iso_ne, device))
        # 兼容旧代码，如有需要可取消注释
        all_results.extend(run_ablation("Malaysia", load_malaysia, device))

    except Exception as e:
        print(f"Global Error: {e}")
        import traceback

        traceback.print_exc()

    if all_results:
        df_res = pd.DataFrame(all_results)
        cols = ['Dataset', 'Variant', 'MAPE', 'RMSE', 'MAE', 'R2', 'Time(s)']
        df_res = df_res[cols]

        print("\n" + "=" * 60)
        print("FINAL ABLATION RESULTS")
        print("=" * 60)
        print(df_res.to_string(index=False))

        output_file = "ablation_results.csv"
        df_res.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")


