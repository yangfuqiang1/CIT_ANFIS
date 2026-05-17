# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats, linalg
from itertools import combinations

class PCMCI:
    """ 针对时间序列设计的 PCMCI 因果发现算法 """

    def __init__(self, alpha=0.05, max_lag=24, max_cond_depth=2, verbose=False):
        self.alpha = alpha
        self.max_lag = max_lag
        self.max_cond_depth = max_cond_depth
        self.verbose = verbose

    def _get_lagged_matrix(self, data, target_name):
        """ 【修复】：如果 max_lag 为 0，说明输入 X 已经由外部处理过滞后 """
        if self.max_lag == 0:
            return data.rename(columns={target_name: f"{target_name}_t0"}).dropna()
        
        # 原始滞后矩阵构造逻辑
        var_names = data.columns.tolist()
        lagged_data = {f"{target_name}_t0": data[target_name].values[self.max_lag:]}
        for var in var_names:
            for lag in range(1, self.max_lag + 1):
                lagged_data[f"{var}_t-{lag}"] = data[var].shift(lag).values[self.max_lag:]
        return pd.DataFrame(lagged_data).dropna()

    def _partial_corr(self, x, y, Z_matrix):
        """ 计算偏相关系数及 p-value """
        if Z_matrix is None or Z_matrix.shape[1] == 0:
            return stats.pearsonr(x, y)
        combined = np.column_stack([x, y, Z_matrix])
        try:
            cov = np.cov(combined, rowvar=False)
            prec = linalg.inv(cov)
            r_val = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
            r_val = np.clip(r_val, -0.999999, 0.999999)
            df = combined.shape[0] - Z_matrix.shape[1] - 2
            if df <= 0: return 0.0, 1.0
            z = 0.5 * np.log((1 + r_val) / (1 - r_val))
            p_val = 2 * (1 - stats.norm.cdf(abs(z * np.sqrt(df))))
            return r_val, p_val
        except: return 0.0, 1.0

    def fit(self, df, target_name):
        """ 执行 PCMCI """
        lagged_df = self._get_lagged_matrix(df, target_name)
        Y_t = lagged_df[f"{target_name}_t0"].values
        X_candidates = lagged_df.drop(columns=[f"{target_name}_t0"])
        feat_names = X_candidates.columns.tolist()

        # Phase 1: PC 阶段
        pc_parents = feat_names.copy()
        for k in range(self.max_cond_depth + 1):
            to_remove = []
            for feat in pc_parents:
                others = [p for p in pc_parents if p != feat]
                if len(others) < k: continue
                for cond_set in combinations(others, k):
                    Z = X_candidates[list(cond_set)].values
                    _, p_val = self._partial_corr(X_candidates[feat].values, Y_t, Z)
                    if p_val > self.alpha:
                        to_remove.append(feat)
                        break
            for r in to_remove:
                if r in pc_parents: pc_parents.remove(r)
            if not to_remove: break

        # Phase 2: MCI 阶段
        mci_results, final_weights = [], np.zeros(len(feat_names))
        for feat in pc_parents:
            Z_names = [p for p in pc_parents if p != feat]
            Z_matrix = X_candidates[Z_names].values
            r_mci, p_mci = self._partial_corr(X_candidates[feat].values, Y_t, Z_matrix)
            if p_mci < self.alpha:
                idx = feat_names.index(feat)
                final_weights[idx] = 1.0 + abs(r_mci)
                mci_results.append(feat)
        return final_weights, mci_results