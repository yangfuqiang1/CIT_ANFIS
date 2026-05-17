# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from .converter import parse_xgb_rules

# 【修复 1】：统一类名导入并解决路径问题
try:
    from .pcmci_discovery import PCMCI as PCMCI_CausalDiscovery
except ImportError:
    try:
        from CITanfis.pcmci_discovery import PCMCI as PCMCI_CausalDiscovery
    except ImportError:
        PCMCI_CausalDiscovery = None
        print("[Warning] pcmci_discovery.py not found. Causal Discovery disabled.")

class TreeANFIS(nn.Module):
    def __init__(self, n_estimators=50, max_depth=4, learning_rate=0.1, objective='reg:squarederror',
                 order=1, use_causal=False, interaction_threshold=0.05):
        super(TreeANFIS, self).__init__()
        self.struct_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': objective,
            'n_jobs': -1,
            'random_state': 42
        }
        self.order = order
        self.use_causal = use_causal
        self.interaction_threshold = interaction_threshold
        self.is_initialized = False
        self.n_features = 0

        self.register_buffer('rule_feat_idxs', None)
        self.register_buffer('rule_threshs', None)
        self.register_buffer('rule_signs', None)
        self.register_buffer('rule_masks', None)
        self.register_buffer('interaction_pairs', None)

        self.premise_params = None
        self.consequent_params = None
        self.attention_weights = None

    def identify_structure(self, X_np, y_np, feature_names=None, causal_weights=None):
        """ Phase 1: 结构辨识 """
        print(f">>> [Phase 1] Structure Identification (Order={self.order})...")
        self.n_features = X_np.shape[1]
        init_attn = torch.ones(self.n_features)

        # 优先级 1: 外部传入权重
        if causal_weights is not None:
            print("   [Internal] Applying external PCMCI causal weights...")
            init_attn = torch.tensor(causal_weights, dtype=torch.float32)
        
        # 优先级 2: 运行内部 PCMCI
        elif self.use_causal and PCMCI_CausalDiscovery is not None:
            print("   [Internal] Running internal PCMCI discovery...")
            try:
                if feature_names is None:
                    feature_names = [f"f{i}" for i in range(self.n_features)]
                
                # 【修复 2】：适配 TruePCMCI 的 fit(df, target) 接口
                tmp_df = pd.DataFrame(X_np, columns=feature_names)
                tmp_df['target'] = y_np
                
                # max_lag=0 因为 X_np 已经包含滞后特征
                pcmci = PCMCI_CausalDiscovery(alpha=0.05, max_lag=0, max_cond_depth=1, verbose=False)
                weights_np, _ = pcmci.fit(tmp_df, 'target')
                init_attn = torch.tensor(weights_np, dtype=torch.float32)

                if init_attn.sum() < 0.1:
                    init_attn = torch.ones(self.n_features)
            except Exception as e:
                print(f"   [Warning] Internal PCMCI failed: {e}. Using default ones.")
                init_attn = torch.ones(self.n_features)

        self.attention_weights = nn.Parameter(torch.nan_to_num(init_attn, nan=1.0))
        X_weighted_np = X_np * self.attention_weights.detach().cpu().numpy()

        # XGBoost 建树提取规则
        model = xgb.XGBRegressor(**self.struct_params)
        model.fit(X_weighted_np, y_np)

        raw_rules = parse_xgb_rules(model)
        n_rules = len(raw_rules)
        max_len = max(len(r['conditions']) for r in raw_rules) if raw_rules else 0

        # 二阶交互挖掘
        if self.order == 2 and self.interaction_threshold > 0:
            print(f"   [Interactions] Mining pairs (threshold={self.interaction_threshold})...")
            pair_counter = Counter()
            for r in raw_rules:
                feats_in_path = sorted(list(set([int(cond[0]) for cond in r['conditions']])))
                for f1, f2 in combinations(feats_in_path, 2):
                    pair_counter[(f1, f2)] += 1
            min_count = int(n_rules * self.interaction_threshold)
            selected_pairs = [pair for pair, count in pair_counter.items() if count >= min_count]
            self.interaction_pairs = torch.tensor(selected_pairs, dtype=torch.long) if selected_pairs else None
        else:
            self.interaction_pairs = None

        # 构造规则张量
        if n_rules > 0:
            feat_idxs_t = torch.zeros((n_rules, max_len), dtype=torch.long)
            threshs_t = torch.zeros((n_rules, max_len), dtype=torch.float32)
            signs_t = torch.zeros((n_rules, max_len), dtype=torch.float32)
            masks_t = torch.zeros((n_rules, max_len), dtype=torch.float32)
            for i, r in enumerate(raw_rules):
                for j, (f_idx, sign, thresh) in enumerate(r['conditions']):
                    feat_idxs_t[i, j], threshs_t[i, j], signs_t[i, j], masks_t[i, j] = int(f_idx), float(thresh), float(sign), 1.0
        else:
            feat_idxs_t, threshs_t, signs_t, masks_t = torch.zeros((1,1), dtype=torch.long), torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1))

        self.rule_feat_idxs, self.rule_threshs, self.rule_signs, self.rule_masks = feat_idxs_t, threshs_t, signs_t, masks_t
        self.premise_params = nn.Parameter(torch.ones(max(n_rules, 1)))
        
        dim_con = 1 + self.n_features + (self.n_features if self.order == 2 else 0) + (self.interaction_pairs.shape[0] if self.interaction_pairs is not None else 0)
        self.consequent_params = nn.Parameter(torch.randn(max(n_rules, 1), dim_con) * 0.01)
        self.is_initialized = True
        print(f">>> [Hybrid] Base initialized with {n_rules} rules (Poly Dim={dim_con}).")

    def _apply_attention(self, x):
        if self.attention_weights.device != x.device:
            self.attention_weights.data = self.attention_weights.data.to(x.device)
        return x * self.attention_weights

    def fuzzify_and_infer(self, x):
        device = x.device
        selected_x = x[:, self.rule_feat_idxs]
        beta = self.premise_params.view(1, -1, 1)
        z = beta * (selected_x - self.rule_threshs.unsqueeze(0)) * self.rule_signs.unsqueeze(0)
        mf_values = torch.sigmoid(z)
        masked_mf = mf_values * self.rule_masks.unsqueeze(0) + (1.0 - self.rule_masks.unsqueeze(0))
        firing_strength = masked_mf.prod(dim=2)
        return firing_strength / (firing_strength.sum(dim=1, keepdim=True) + 1e-8)

    def _get_tsk_polynomials(self, x):
        batch_size = x.shape[0]
        components = [x, torch.ones((batch_size, 1), device=x.device)]
        if self.order == 2:
            components.insert(1, x.pow(2))
            if self.interaction_pairs is not None:
                interactions = x[:, self.interaction_pairs[:, 0]] * x[:, self.interaction_pairs[:, 1]]
                components.insert(2, interactions)
        return torch.cat(components, dim=1)

    def initialize_consequents(self, X_t, y_t, rcond=1e-2, batch_size=2048):
        print(f"   [Phase 2] Optimizing Consequents (Robust LSE)...")
        device = X_t.device
        n_rules, n_poly = self.rule_feat_idxs.shape[0], self.consequent_params.shape[1]
        XtX_local = torch.zeros((n_rules, n_poly, n_poly), device=device)
        XtY_local = torch.zeros((n_rules, n_poly, 1), device=device)
        rule_activity = torch.zeros(n_rules, device=device)
        self.eval()
        with torch.no_grad():
            for i in range(0, X_t.shape[0], batch_size):
                bx, by = X_t[i:i + batch_size], y_t[i:i + batch_size].view(-1, 1)
                bx_att = bx * self.attention_weights.to(device)
                w, p = self.fuzzify_and_infer(bx_att), self._get_tsk_polynomials(bx_att)
                rule_activity += w.sum(dim=0)
                XtX_local += torch.einsum('br,bi,bj->rij', w, p, p)
                XtY_local += torch.einsum('br,bi,bo->rio', w, p, by)
        I = torch.eye(n_poly, device=device).unsqueeze(0)
        for r in range(n_rules):
            if rule_activity[r] > 1.0:
                try:
                    self.consequent_params.data[r] = torch.linalg.solve(XtX_local[r] + rcond * I, XtY_local[r]).squeeze()
                except: pass
        print(f"   [Phase 2] LSE Solved.")

    def optimize_rule_base(self, X, threshold=0.01, batch_size=2048):
        print(f">>> [Pruning] Optimizing Rule Base...")
        self.eval()
        X = X.to(self.attention_weights.device) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32, device=self.attention_weights.device)
        rule_imp = torch.zeros(self.rule_feat_idxs.shape[0], device=X.device)
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                bx = X[i:i + batch_size] * self.attention_weights
                rule_imp += self.fuzzify_and_infer(bx).sum(dim=0)
        keep_mask = (rule_imp / X.shape[0] > threshold).cpu()
        if keep_mask.sum() > 0:
            self.rule_feat_idxs, self.rule_threshs = self.rule_feat_idxs[keep_mask], self.rule_threshs[keep_mask]
            self.rule_signs, self.rule_masks = self.rule_signs[keep_mask], self.rule_masks[keep_mask]
            self.premise_params = nn.Parameter(self.premise_params[keep_mask])
            self.consequent_params = nn.Parameter(self.consequent_params[keep_mask])
            print(f"   [Pruning] Kept {keep_mask.sum().item()} rules.")

    def forward(self, x):
        x_w = self._apply_attention(x)
        w, p = self.fuzzify_and_infer(x_w), self._get_tsk_polynomials(x_w)
        return (w * torch.einsum('bd,rd->br', p, self.consequent_params)).sum(dim=1)

    def train_neuro_fuzzy(self, X, y, X_val=None, y_val=None, epochs=100, lr=0.01, batch_size=2048, patience=10):
        print(f">>> [Phase 3] Training...")
        device = self.attention_weights.device
        opt = torch.optim.Adam([{'params': self.premise_params, 'lr': lr}, {'params': self.consequent_params, 'lr': lr}, {'params': self.attention_weights, 'lr': lr * 0.1}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        loss_fn, best_loss, no_imp, best_w = nn.MSELoss(), float('inf'), 0, None
        for ep in tqdm(range(epochs), desc="Training", unit="ep"):
            self.train()
            perm = torch.randperm(X.size(0), device=device)
            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i + batch_size]
                bx, by = X[idx], y[idx]
                opt.zero_grad()
                loss = loss_fn(self.forward(bx), by.view(-1)) + 1e-5 * self.attention_weights.abs().sum()
                loss.backward()
                self.attention_weights.data.clamp_(min=0.0)
                opt.step()
            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.forward(X_val), y_val.view(-1)).item() if X_val is not None else 0
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss, best_w, no_imp = val_loss, copy.deepcopy(self.state_dict()), 0
            elif no_imp >= patience: break
            else: no_imp += 1
        if best_w: self.load_state_dict(best_w)