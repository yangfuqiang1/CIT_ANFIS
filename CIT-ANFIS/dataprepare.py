# -*- coding: utf-8 -*-
"""
执行脚本：使用 Tree-ANFIS 模型对三个真实电力负荷数据集进行预测
包含：ISO-NE, Malaysia, North American

修改说明：
1. 所有数据集均采用“独立验证集”策略（从训练集末尾划分 5% 用于 Early Stopping）。
2. 适配动态学习率与早停机制。
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
import time
import random
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar

# 导入节假日库（需提前安装：pip install holidays）
try:
    import holidays
except ImportError:
    print("[Warning] 未安装 holidays 库，节假日特征将无法生成，请运行：pip install holidays")
    raise

# 忽略警告
warnings.filterwarnings('ignore')


def setup_seed(seed):
    """固定随机种子，保证实验复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[System] Global Random Seed set to: {seed}")


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差（论文核心指标）"""
    epsilon = np.finfo(np.float64).eps
    ape = np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))
    return np.mean(ape) * 100


# ==========================================
# 2. 数据加载与预处理 (严格对齐论文时间范围)
# ==========================================

def load_iso_ne():
    """加载ISO-NE数据集（2023-2024），论文要求时间范围"""
    print("Loading ISO-NE (2023-2024)...")
    try:
        df = pd.read_csv('data/ISO-NE (2023-2024).csv')
        df['date'] = pd.to_datetime(df['date'])
        df['Datetime'] = df['date'] + pd.to_timedelta(df['hour'] - 1, unit='h')

        # 重命名为统一格式
        df = df[['Datetime', 'demand', 'temperature']]
        df.columns = ['Datetime', 'Total Load', 'Temperature']

        # 过滤论文要求的时间范围（2023.1.1-2024.12.31）
        df = df[(df['Datetime'] >= pd.to_datetime("2023-01-01")) &
                (df['Datetime'] <= pd.to_datetime("2024-12-31"))]

        # 去重、排序
        df = df.drop_duplicates('Datetime').sort_values('Datetime').reset_index(drop=True)
        print(f"ISO-NE Data Loaded: {len(df)} hours (target: 17544)")
        return df
    except Exception as e:
        print(f"Error loading ISO-NE: {e}")
        return None


def load_malaysia():
    """加载马来西亚数据集（2009-2010），论文要求时间范围"""
    print("Loading Malaysia (2009-2010)...")
    try:
        df = pd.read_csv('data/NEW-Malaysia.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['Datetime'] = df['date'] + pd.to_timedelta(df['hour'] - 1, unit='h')

        # 重命名为统一格式
        df = df[['Datetime', 'demand', 'temperature']]
        df.columns = ['Datetime', 'Total Load', 'Temperature']

        # 过滤论文要求的时间范围（2009.1.1-2010.12.31）
        df = df[(df['Datetime'] >= pd.to_datetime("2009-01-01")) &
                (df['Datetime'] <= pd.to_datetime("2010-12-31"))]

        # 去重、排序
        df = df.drop_duplicates('Datetime').sort_values('Datetime').reset_index(drop=True)
        print(f"Malaysia Data Loaded: {len(df)} hours (target: 17520)")
        return df
    except Exception as e:
        print(f"Error loading Malaysia: {e}")
        return None


def load_north_american():
    """
    加载北美数据集（1985-1992年，论文实际数据范围，目标68208小时）
    """
    print("Loading North American (1985-1992)...")
    try:
        # 验证数据集文件是否存在
        required_files = ['data/output.txt', 'data/input.txt']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"缺少数据集文件：{file}，请从论文GitHub下载完整版本")

        # 读取负荷数据（output.txt）和温度数据（input.txt）
        try:
            df_out = pd.read_csv('output.txt', sep='\s+', header=None)
        except:
            df_out = pd.read_csv('output.txt', header=None)

        try:
            df_temp = pd.read_csv('input.txt', sep='\s+', header=None)
        except:
            df_temp = pd.read_csv('input.txt', header=None)

        dates = []
        loads = []
        temps = []
        invalid_dates = 0

        for idx, row in df_out.iterrows():
            date_str = str(row[0]).strip()
            daily_loads = row[1:25].values
            daily_temps = df_temp.iloc[idx, 1:25].values if idx < len(df_temp) else np.zeros(24)

            # 年份解析
            try:
                curr_date = pd.to_datetime(date_str, format='%m/%d/%Y', errors='raise')
            except:
                try:
                    curr_date = pd.to_datetime(date_str, format='%m/%d/%y', errors='raise')
                except:
                    curr_date = pd.to_datetime(date_str, errors='coerce')

            if pd.isna(curr_date):
                invalid_dates += 1
                continue

            # 强制修正年份
            target_year_range = [1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992]
            if curr_date.year not in target_year_range:
                year_last_two = curr_date.year % 100
                if 85 <= year_last_two <= 92:
                    corrected_year = 1900 + year_last_two
                elif year_last_two < 85:
                    corrected_year = 1985
                else:
                    corrected_year = 1985

                if corrected_year < 1985:
                    corrected_year = 1985
                elif corrected_year > 1992:
                    corrected_year = 1992
                curr_date = curr_date.replace(year=corrected_year)

            start_date = pd.to_datetime("1985-01-01")
            end_date = pd.to_datetime("1992-10-12")
            if curr_date < start_date or curr_date > end_date:
                continue

            for h in range(24):
                hourly_datetime = curr_date + datetime.timedelta(hours=h)
                dates.append(hourly_datetime)
                val_l = daily_loads[h]
                if val_l == 0 and len(loads) > 0:
                    val_l = loads[-1]
                loads.append(val_l)
                val_t = daily_temps[h] if not np.isnan(daily_temps[h]) else 0.0
                temps.append(val_t)

        df = pd.DataFrame({
            'Datetime': dates,
            'Total Load': loads,
            'Temperature': temps
        })
        df = df.drop_duplicates(subset=['Datetime'], keep='first')
        df = df.sort_values('Datetime').reset_index(drop=True)

        total_hours = len(df)
        target_hours = 68208
        print(f"North American Data Loaded: {total_hours} hours (target: {target_hours})")
        return df

    except Exception as e:
        print(f"Error loading North American dataset: {str(e)}")
        return None


# ==========================================
# 3. 特征工程 (完全对齐论文表I输入特征)
# ==========================================

def enhance_features(df, dataset_name):
    """补充论文要求的所有输入特征"""
    if 'Datetime' not in df.columns:
        raise ValueError("Dataframe must have 'Datetime' column")

    df = df.copy()

    # 1. 基础时间特征
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df['Season'] = pd.cut(
        df['Month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Spring', 'Summer', 'Autumn', 'Winter']
    ).astype(str)

    # 2. 独热编码
    df = pd.get_dummies(df, columns=['Season'], prefix='Season')
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(float)

    if dataset_name == "North American" or dataset_name == "ISO-NE":
        cal = USFederalHolidayCalendar()
        holiday_dates = cal.holidays(start=df['Datetime'].min(), end=df['Datetime'].max())
        df['Is_Holiday'] = df['Datetime'].dt.date.isin(holiday_dates).astype(float)
    elif dataset_name == "Malaysia":
        my_holidays = holidays.MY(years=df['Datetime'].dt.year.unique())
        df['Is_Holiday'] = df['Datetime'].dt.date.isin(my_holidays).astype(float)

    # 3. 周期性编码
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # 4. 滞后特征（负荷）
    target = 'Total Load'
    df['Lag_1_Hour'] = df[target].shift(1)
    df['Lag_24_Hour'] = df[target].shift(24)
    df['Lag_1_Week'] = df[target].shift(24 * 7)
    df['Lag_4_Week'] = df[target].shift(24 * 7 * 4)
    df['Lag_8_Week'] = df[target].shift(24 * 7 * 8)
    df['Lag_12_Week'] = df[target].shift(24 * 7 * 12)

    # 5. 滞后特征（温度）
    df['Temp_Lag_24_Hour'] = df['Temperature'].shift(24)
    df['Temp_Lag_1_Week'] = df['Temperature'].shift(24 * 7)
    df['Temp_Lag_4_Week'] = df['Temperature'].shift(24 * 7 * 4)
    df['Temp_Lag_8_Week'] = df['Temperature'].shift(24 * 7 * 8)

    # 6. 滚动统计
    roll_day = df[target].shift(1).rolling(window=24)
    df['Roll_Mean_Day'] = roll_day.mean()
    df['Roll_Max_Day'] = roll_day.max()
    df['Roll_Min_Day'] = roll_day.min()
    df['EMA_24H'] = df[target].shift(1).ewm(span=24, adjust=False).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# ==========================================
# 4. 数据集划分 (核心修改：均使用独立验证集)
# ==========================================

def split_dataset_by_paper(df, dataset_name):
    """
    1. 按照论文时间范围进行 Train/Val/Test 硬划分。
    2. [修改] 对所有数据集，从 Train 中再切分出最后 5% 作为独立验证集 (Independent Val)。
       用于模型训练时的 Early Stopping，防止对论文定义的测试集/验证集过拟合。
    """
    df = df.sort_values('Datetime').reset_index(drop=True)
    train_df, val_df, test_df = None, None, None

    # --- 第一步：按论文时间轴切分 Train/Val/Test ---
    if dataset_name == "North American":
        train_end = pd.to_datetime("1990-08-17")
        val_end = pd.to_datetime("1990-10-12")
        test_end = pd.to_datetime("1992-10-12")
    elif dataset_name == "Malaysia":
        train_end = pd.to_datetime("2010-10-01")
        val_end = pd.to_datetime("2010-10-31")
        test_end = pd.to_datetime("2010-12-31")
    elif dataset_name == "ISO-NE":
        train_end = pd.to_datetime("2024-10-01")
        val_end = pd.to_datetime("2024-10-31")
        test_end = pd.to_datetime("2024-12-31")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_df = df[df['Datetime'] <= train_end]
    val_df = df[(df['Datetime'] > train_end) & (df['Datetime'] <= val_end)]
    test_df = df[(df['Datetime'] > val_end) & (df['Datetime'] <= test_end)]

    # --- 第二步：从训练集中构建独立验证集 (所有数据集通用) ---
    # 提取训练集最后 5% 作为内部验证集，用于 Early Stopping
    independent_val_size = int(len(train_df) * 0.05)
    if independent_val_size < 24:  # 保护性检查，防止数据太少
        independent_val_size = 24

    independent_val_df = train_df.iloc[-independent_val_size:]
    train_df = train_df.iloc[:-independent_val_size]

    print(f"\n{dataset_name} Split Results (with Independent Validation):")
    print(
        f"  Train: {len(train_df)} | Indep-Val: {len(independent_val_df)} | Paper-Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df, independent_val_df


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

    print(f"   Original shape: {data.shape} (Raw)")
    # 去除时区信息 (变成 Naive Time)，防止后续 pytorch 转换报错
    data['Datetime'] = data['Datetime'].dt.tz_localize(None)
    print(f"   Resampled shape: {data.shape} (Original Resolution)")

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

def add_temperature_noise_north_american(df):
    """对北美数据集添加论文要求的温度噪声"""
    np.random.seed(42)
    df['Temperature_Original'] = df['Temperature'].copy()
    noise = np.random.normal(loc=0, scale=1, size=len(df))
    df['Temperature_Noisy'] = df['Temperature'] + noise
    print("Added Gaussian noise to North American temperature (μ=0, σ=1)")
    return df




