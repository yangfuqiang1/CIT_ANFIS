# 文件路径: xganfis/pcmci_discovery.py

import numpy as np
import pandas as pd
from scipy import stats, linalg


class PCMCI_CausalDiscovery:
    """
    轻量级 PCMCI (Phase 1: PC-Stable) 实现。
    用于时间序列特征选择，通过条件独立性测试剔除伪相关。

    逻辑:
    1. 筛选: 先找出所有与 Y 相关的变量 (Marginal Correlation)。
    2. 验证: 对于每个特征 X，控制其他强特征 Z，计算偏相关 P(X, Y | Z)。
       如果 p-value > alpha，说明 X 对 Y 的影响被 Z 解释了，剔除 X。
    """

    def __init__(self, alpha=0.05, max_cond_depth=1, verbose=False):
        """
        Args:
            alpha: 显著性水平 (p-value threshold)，默认 0.05。
            max_cond_depth: 条件集的最大深度。设为 1 通常足够去除主要混淆因子且速度快。
        """
        self.alpha = alpha
        self.max_cond_depth = max_cond_depth
        self.verbose = verbose

    def _get_partial_corr_p_value(self, x, y, Z):
        """
        利用精度矩阵（协方差矩阵的逆）高效计算偏相关系数及其 P 值。
        """
        # 数据堆叠: [x, y, Z]
        if Z is None or Z.shape[1] == 0:
            r, p = stats.pearsonr(x, y)
            return r, p

        data = np.column_stack([x, y, Z])
        n_samples = data.shape[0]

        # 1. 计算协方差和精度矩阵
        try:
            cov = np.cov(data, rowvar=False)
            prec = linalg.inv(cov)  # Precision matrix
        except linalg.LinAlgError:
            # 如果矩阵奇异（共线性），保守返回不显著
            return 0.0, 1.0

        # 2. 从精度矩阵提取偏相关
        # Formula: rho_xy.z = -p_xy / sqrt(p_xx * p_yy)
        p_xx = prec[0, 0]
        p_yy = prec[1, 1]
        p_xy = prec[0, 1]

        if p_xx * p_yy <= 0:
            return 0.0, 1.0

        r_val = -p_xy / np.sqrt(p_xx * p_yy)
        r_val = np.clip(r_val, -0.99999, 0.99999)  # 数值稳定截断

        # 3. Fisher Z-Transform 计算显著性
        k = Z.shape[1]
        df = n_samples - k - 2  # 自由度

        if df <= 0: return 0.0, 1.0

        z_score = 0.5 * np.log((1 + r_val) / (1 - r_val))
        se = 1.0 / np.sqrt(df)
        statistic = z_score / se

        # 双尾检验 p-value
        p_val = 2 * (1 - stats.norm.cdf(abs(statistic)))

        return r_val, p_val

    def fit(self, X, y, feature_names=None):
        """
        执行特征选择，返回每个特征的权重。
        """
        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(n_features)]

        # 输出权重容器
        weights = np.zeros(n_features)
        df_X = pd.DataFrame(X, columns=feature_names)

        if self.verbose:
            print(f">>> [PCMCI] Starting Causal Discovery on {n_features} features (alpha={self.alpha})...")

        # ==========================================================
        # 第一阶段: 初步筛选 (Marginal Correlation)
        # ==========================================================
        initial_parents = []
        correlations = []

        for feat in feature_names:
            r, p = stats.pearsonr(df_X[feat], y)
            if p < self.alpha:
                initial_parents.append(feat)
                correlations.append(abs(r))
            else:
                weights[feature_names.index(feat)] = 0.0  # 直接剔除不相关的

        # 按相关性强度排序 (PC-Stable 的关键启发式: 先控制强变量)
        sorted_indices = np.argsort(correlations)[::-1]
        current_parents = [initial_parents[i] for i in sorted_indices]

        if self.verbose:
            print(f"    [Phase 1] Kept {len(current_parents)} features based on marginal correlation.")

        # ==========================================================
        # 第二阶段: 条件独立性测试 (Conditional Independence)
        # ==========================================================
        final_selected = []

        for feat in current_parents:
            feat_idx = feature_names.index(feat)
            is_spurious = False

            # 找到潜在的混淆因子 (除了自己以外的其他强相关特征)
            other_parents = [p for p in current_parents if p != feat]

            if not other_parents:
                # 如果没有其他变量干扰，它就是因果
                weights[feat_idx] = 1.0 + abs(correlations[initial_parents.index(feat)])
                final_selected.append(feat)
                continue

            # 取最强的 k 个变量作为条件集 Z
            cond_set_names = other_parents[:self.max_cond_depth]
            Z_matrix = df_X[cond_set_names].values
            x_vec = df_X[feat].values

            # 核心测试: X 和 Y 在已知 Z 的情况下还相关吗？
            p_corr, p_val = self._get_partial_corr_p_value(x_vec, y, Z_matrix)

            if p_val > self.alpha:
                # p > 0.05 -> 不显著 -> 独立 -> 说明是伪相关
                is_spurious = True
                weights[feat_idx] = 0.1  # 软剪枝 (给予极低权重)
                if self.verbose:
                    print(f"    [Pruned] {feat:<15} | Indep given {cond_set_names} (p={p_val:.3f})")
            else:
                # p < 0.05 -> 显著 -> 依赖 -> 可能是真因果
                # 权重 = 1.0 + 偏相关强度
                weights[feat_idx] = 1.0 + abs(p_corr)
                final_selected.append(feat)

        if self.verbose:
            print(f">>> [PCMCI] Finished. Verified {len(final_selected)} causal drivers.")

        return weights