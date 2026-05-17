# -*- coding: utf-8 -*-
"""
Step1_XMQ_Only.py
------------------
[Proposed Model Dedicated Version]
Features:
1. Focused Execution: Stripped of all 8 baseline models. Focuses SOLELY on XMQ Proposed Model.
2. 3-Dataset Benchmark: Automatically trains and evaluates on Malaysia, ISO-NE, and Belgium.
3. Architecture: Reconstructs Stage I (Fuzzy + LSTM + Attention) & Stage II (ResNetPlus) in PyTorch.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from tqdm import tqdm

# ==========================================
# 0. 环境与依赖初始化
# ==========================================
warnings.filterwarnings('ignore')

print(f"{'=' * 60}")
print("DEDICATED RUNNER: XMQ PROPOSED MODEL ONLY")
print(f"{'=' * 60}")

# 导入数据工具
try:
    from dataprepare import load_iso_ne, load_malaysia, enhance_features, split_dataset_by_paper
    print("[System] Data loaders imported successfully.")
except ImportError:
    print("[Critical Error] Cannot find 'dataprepare.py'. 请确保该文件在同一目录下。")
    sys.exit(1)


def setup_seed(seed=2024):
    """ Lock random seeds for reproducibility """
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


# ==============================================================================
# 1. 论文核心架构 PyTorch 实现 (Fuzzy + LSTM + Attention + ResNetPlus)
# ==============================================================================

class PyTorchResNetPlusBlock(nn.Module):
    """
    对应论文图 4 与原版 get_res_layer 函数 [cite: 216, 237]：
    包含 4 组独立的 Dense(20)->Dense(24) 并行残差块 [cite: 239]，采用 SELU 激活函数 [cite: 277]。
    """
    def __init__(self, channels=24):
        super().__init__()
        self.d11 = nn.Linear(channels, 20)
        self.d12 = nn.Linear(20, channels)
        
        self.d21 = nn.Linear(channels, 20)
        self.d22 = nn.Linear(20, channels)
        
        self.d31 = nn.Linear(channels, 20)
        self.d32 = nn.Linear(20, channels)
        
        self.d41 = nn.Linear(channels, 20)
        self.d42 = nn.Linear(20, channels)

    def forward(self, x):
        r1 = self.d12(F.selu(self.d11(x)))
        r2 = self.d22(F.selu(self.d21(x)))
        r3 = self.d32(F.selu(self.d31(x)))
        r4 = self.d42(F.selu(self.d41(x)))
        return r1 + r2 + r3 + r4 + x  # 4组输出与原始输入相加 [cite: 214, 239]


class XMQProposedModel(nn.Module):
    """
    本文提出的核心预测网络 [cite: 60]
    Stage I: 负荷数据由 LSTM 层与模糊逻辑层独立并行处理 [cite: 75, 268]，并通过自注意力机制动态调整权重融合 。
    Stage II: 融合特征输入修改后的深层残差网络 (ResNetPlus) 迭代输出最终预测 [cite: 76, 236]。
    """
    def __init__(self, in_dim):
        super().__init__()
        # ----------------- Stage I 组件 -----------------
        # LSTM 分支捕捉时序长期依赖 [cite: 7, 269]
        self.lstm_branch = nn.LSTM(input_size=in_dim, hidden_size=32, batch_first=True)
        
        # 模糊逻辑分支：高斯隶属度函数参数中心 c 和标准差 sigma [cite: 159, 165]
        self.fuzzy_centers = nn.Parameter(torch.randn(in_dim, 32) * 0.1)
        self.fuzzy_sigmas = nn.Parameter(torch.ones(in_dim, 32))
        self.fuzzy_fc = nn.Linear(32, 32)
        
        # 自注意力机制：用于融合两个独立分支的输出特征 [cite: 75, 223]
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.stage1_fc = nn.Linear(32, 24)
        
        # ----------------- Stage II 组件 -----------------
        # ResNetPlus 网络结构，这里默认设为 15 层 [cite: 238, 428]
        self.resnet_layers = nn.ModuleList([PyTorchResNetPlusBlock(channels=24) for _ in range(15)])
        self.final_projection = nn.Linear(24, 1)

    def forward(self, x):
        # 1. 运行时序 LSTM 分支 [cite: 268]
        lstm_out, _ = self.lstm_branch(x.unsqueeze(1))
        lstm_out = F.selu(lstm_out.squeeze(1))
        
        # 2. 运行高斯模糊层分支 [cite: 166, 268]：u(x) = exp( - (x - c)^2 / sigma^2 ) [cite: 161]
        x_expanded = x.unsqueeze(-1).expand(-1, -1, 32)
        fuzzy_membership = torch.exp(-torch.sum(((x_expanded - self.fuzzy_centers) ** 2) / (self.fuzzy_sigmas ** 2 + 1e-6), dim=1))
        fuzzy_out = F.selu(self.fuzzy_fc(fuzzy_membership))
        
        # 3. 自注意力机制交互融合 
        stacked_features = torch.stack([lstm_out, fuzzy_out], dim=1)
        attn_out, _ = self.attention(stacked_features, stacked_features, stacked_features)
        stage1_output = self.stage1_fc(attn_out.mean(dim=1))  # 融合输出 preliminary 结果 [cite: 75]
        
        # 4. Stage II: ResNetPlus 密集平均残差层迭代传递 [cite: 76, 236]
        res_current = stage1_output
        history_outputs = [res_current]
        
        for layer in self.resnet_layers:
            block_output = layer(res_current)
            history_outputs.append(block_output)
            # 严格对应原本的计算逻辑：当前块输出与历史所有块进行 average 操作 [cite: 238]
            res_current = torch.stack(history_outputs, dim=0).mean(dim=0)
            
        return self.final_projection(res_current).squeeze(-1)


# ==========================================
# 2. 比利时数据集专用数据流
# ==========================================

def load_belgium_data(filepath='data/ods001.csv'):
    """ 读取并预处理比利时电网数据 """
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

    df.columns = df.columns.str.strip()
    if 'Datetime' not in df.columns:
        print(f"[Error] 'Datetime' column missing. Found: {df.columns.tolist()}")
        sys.exit(1)

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
            print(f"[Error] Column '{target_col}' not found. Available: {df.columns.tolist()}")
            sys.exit(1)

    data = df[['Datetime', target_col]].copy()
    data.columns = ['Datetime', 'Total Load']
    data.set_index('Datetime', inplace=True)
    data = data.resample('1h').mean().interpolate()
    data.reset_index(inplace=True)
    data['Datetime'] = data['Datetime'].dt.tz_localize(None)

    # 截取 2024-2025 年数据 [cite: 456]
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-12-31')
    data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)].copy()
    return data.reset_index(drop=True)


def enhance_features_belgium(df):
    """ 比利时无温度场景下的时序特征工程挖掘 """
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

    df['Lag_1_Hour'] = df[target].shift(1)
    df['Lag_2_Hour'] = df[target].shift(2)
    df['Lag_3_Hour'] = df[target].shift(3)
    df['Lag_24_Hour'] = df[target].shift(24)
    df['Lag_25_Hour'] = df[target].shift(25)
    df['Lag_48_Hour'] = df[target].shift(48)
    df['Lag_1_Week'] = df[target].shift(24 * 7)

    shifted_target = df[target].shift(1)
    roll_24 = shifted_target.rolling(window=24)
    df['Roll_Mean_24H'] = roll_24.mean()
    df['Roll_Std_24H'] = roll_24.std()  
    df['Roll_Max_24H'] = roll_24.max()
    df['Roll_Min_24H'] = roll_24.min()

    df['EMA_12H'] = shifted_target.ewm(span=12, adjust=False).mean()
    df['EMA_168H'] = shifted_target.ewm(span=168, adjust=False).mean()
    df['Diff_1H'] = df['Lag_1_Hour'] - df['Lag_2_Hour']
    df['Diff_24H'] = df['Lag_1_Hour'] - df['Lag_24_Hour']

    df.dropna(inplace=True)
    return df


def split_belgium_data(df):
    """ 划分比例：70% 训练, 15% 验证, 15% 测试 """
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:], df.iloc[train_end:val_end]


# ==========================================
# 3. 统一 PyTorch 模型训练引擎
# ==========================================

def train_torch_model(model, X_train, y_train, X_val, y_val, device, epochs=150, lr=0.001, batch_size=32, name="Model"):
    """ 默认采用原论文中推荐的 Adam 优化器 [cite: 277] 与 Batch Size = 32 [cite: 428] """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # [cite: 277]
    criterion = nn.MSELoss()

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
    """ 计算负荷预测核心评价指标：MAPE, RMSE, MAE, R2 """
    epsilon = 1e-7
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100  # [cite: 320]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # [cite: 326]
    mae = mean_absolute_error(y_true, y_pred)  # [cite: 320]
    r2 = r2_score(y_true, y_pred)  # [cite: 326]
    return {"MAPE": mape, "RMSE": rmse, "MAE": mae, "R2": r2, "Time(s)": time_taken}


# ==========================================
# 4. 专属单模型实验逻辑
# ==========================================

def run_single_model_experiment(dataset_name, df_loader_func, device):
    print(f"\n{'-' * 25} Isolated Processing {dataset_name} {'-' * 25}")

    # 数据加载分流
    if dataset_name == "Belgium":
        df = load_belgium_data('data/ods001.csv')
        if df is None: return []
        df_fe = enhance_features_belgium(df)
        train_df, val_df, test_df, ind_val_df = split_belgium_data(df_fe)
        exclude = ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']
        feature_cols = [c for c in df_fe.columns if c not in exclude]
    else:
        if df_loader_func is None: return []
        df = df_loader_func()
        if df is None: return []
        df_fe = enhance_features(df, dataset_name)
        train_df, val_df, test_df, ind_val_df = split_dataset_by_paper(df_fe, dataset_name)
        feature_cols = [c for c in df_fe.columns if c not in ['Datetime', 'Total Load', 'date', 'demand', 'temperature', 'Season']]

    target_col = 'Total Load'
    print(f"Features Dimension Count: {len(feature_cols)}")

    # 数据标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(train_df[feature_cols].values)
    y_train = scaler_y.fit_transform(train_df[target_col].values.reshape(-1, 1)).flatten()
    X_val = scaler_x.transform(ind_val_df[feature_cols].values)
    y_val = scaler_y.transform(ind_val_df[target_col].values.reshape(-1, 1)).flatten()
    X_test = scaler_x.transform(test_df[feature_cols].values)
    y_test = test_df[target_col].values

    # 转换为 PyTorch 张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 核心模型训练
    print(f"-> Training XMQ Proposed Model on {dataset_name}...")
    t0 = time.time()
    
    model_xmq = XMQProposedModel(in_dim=X_train.shape[1]).to(device)
    model_xmq = train_torch_model(model_xmq, X_train_t, y_train_t, X_val_t, y_val_t, device,
                                  epochs=150, lr=0.001, batch_size=32, name="XMQ_Proposed")
    
    model_xmq.eval()
    with torch.no_grad():
        pred_scaled = model_xmq(X_test_t).cpu().numpy()
        
    # 逆标准化
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    time_taken = time.time() - t0

    # 记录数据
    metrics = calculate_metrics(y_test, pred, time_taken)
    metrics['Model'] = 'XMQ_Proposed (Paper)'
    metrics['Dataset'] = dataset_name

    # 导出单模型预测曲线数据
    os.makedirs("result", exist_ok=True)
    df_preds = pd.DataFrame({'Actual': y_test, 'XMQ_Proposed': pred})
    df_preds.to_csv(f"result/pure_predictions_{dataset_name}.csv", index=False)

    return [metrics]


if __name__ == "__main__":
    setup_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}\n")

    all_results = []
    try:
        all_results.extend(run_single_model_experiment("Malaysia", load_malaysia, device))
        all_results.extend(run_experiment_func := run_single_model_experiment("ISO-NE", load_iso_ne, device))
        all_results.extend(run_single_model_experiment("Belgium", None, device))
    except Exception as e:
        import traceback
        traceback.print_exc()

    # 输出纯净版结果
    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("XMQ PROPOSED MODEL EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        print(df_res[['Dataset', 'Model', 'MAPE', 'RMSE', 'MAE', 'R2', 'Time(s)']].to_string(index=False))
        df_res.to_csv("result/xmq_only_results_final.csv", index=False)