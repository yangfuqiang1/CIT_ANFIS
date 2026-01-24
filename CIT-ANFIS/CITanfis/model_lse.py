import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm  # 进度条库
from collections import Counter
from itertools import combinations
from .converter import parse_xgb_rules
from .causal_graph import HeuristicCausalGraph # 这行可以保留也可以注释
try:
    from .pcmci_discovery import PCMCI_CausalDiscovery
except ImportError:
    print("[Warning] pcmci_discovery.py not found using relative import.")

class TreeANFIS(nn.Module):
    """
    Tree-Structured ANFIS [V8: Final Benchmark 3 + Progress Bars]
    Features:
    1. Attention Alignment Fix
    2. Soft Init & Robust LSE
    3. Threshold-Based Interactions (interaction_threshold)
    4. TQDM Progress Bars
    """

    def __init__(self, n_estimators=50, max_depth=4, learning_rate=0.1, objective='reg:squarederror',
                 order=1, use_causal=False, interaction_threshold=0.05):
        """
        Args:
            interaction_threshold: Float (0.0 - 1.0).
                                   Include interaction term (xi*xj) if the pair appears
                                   in more than this fraction of XGBoost rules.
        """
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

        # Buffers
        self.register_buffer('rule_feat_idxs', None)
        self.register_buffer('rule_threshs', None)
        self.register_buffer('rule_signs', None)
        self.register_buffer('rule_masks', None)
        # Interaction pairs buffer
        self.register_buffer('interaction_pairs', None)

        # Parameters
        self.premise_params = None
        self.consequent_params = None
        self.attention_weights = None

    def identify_structure(self, X_np, y_np, feature_names=None):
        """ Phase 1: Structure Identification (Updated with PCMCI) """
        print(f">>> [Phase 1] Structure Identification (Order={self.order})...")
        self.n_features = X_np.shape[1]

        # --- A. Causal Attention Initialization (Upgrade!) ---
        init_attn = torch.ones(self.n_features)

        if self.use_causal:
            print("   [Internal] Initializing Causal Attention via PCMCI (PC-Stable)...")
            try:
                if feature_names is None:
                    feature_names = [f"f{i}" for i in range(self.n_features)]

                # 1. 实例化 PCMCI
                # alpha=0.05: 统计学标准
                # max_cond_depth=1: 控制最强的一个混淆因子 (高效且有效)
                pcmci = PCMCI_CausalDiscovery(alpha=0.05, max_cond_depth=1, verbose=True)

                # 2. 运行因果发现
                weights_np = pcmci.fit(X_np, y_np, feature_names=feature_names)

                # 3. 转换为 Tensor
                init_attn = torch.tensor(weights_np, dtype=torch.float32)

                # 4. 兜底保护：万一所有特征都被剪枝了（极端情况），回退到全1
                if init_attn.sum() < 0.1:
                    print("   [Warning] PCMCI pruned too aggressively. Reverting to ones.")
                    init_attn = torch.ones(self.n_features)

            except Exception as e:
                print(f"   [Warning] PCMCI failed ({e}). Reverting to default initialization.")
                import traceback
                traceback.print_exc()
                init_attn = torch.ones(self.n_features)

        # 确保数值安全
        init_attn = torch.nan_to_num(init_attn, nan=1.0)
        self.attention_weights = nn.Parameter(init_attn)

        # Attention Alignment
        # 将学习到的因果权重应用到输入数据上
        X_weighted_np = X_np * init_attn.detach().cpu().numpy()

        # --- B. XGBoost Training ---
        # 这里的逻辑不变：用加权后的数据训练树
        model = xgb.XGBRegressor(**self.struct_params)
        model.fit(X_weighted_np, y_np)

        raw_rules = parse_xgb_rules(model)
        n_rules = len(raw_rules)
        max_len = max(len(r['conditions']) for r in raw_rules) if raw_rules else 0

        # --- [Interaction Mining] ---
        if self.order == 2 and self.interaction_threshold > 0:
            print(f"   [Interactions] Mining pairs appearing in > {self.interaction_threshold * 100}% of rules...")
            pair_counter = Counter()
            for r in raw_rules:
                feats_in_path = sorted(list(set([int(cond[0]) for cond in r['conditions']])))
                for f1, f2 in combinations(feats_in_path, 2):
                    pair_counter[(f1, f2)] += 1

            min_count = int(n_rules * self.interaction_threshold)
            selected_pairs = [pair for pair, count in pair_counter.items() if count >= min_count]

            if selected_pairs:
                self.interaction_pairs = torch.tensor(selected_pairs, dtype=torch.long)
                print(f"      -> Found {len(selected_pairs)} strong interaction pairs.")
            else:
                self.interaction_pairs = None
                print(f"      -> No interactions met threshold.")
        else:
            self.interaction_pairs = None

        # Build Rule Tensors
        if n_rules > 0:
            feat_idxs_t = torch.zeros((n_rules, max_len), dtype=torch.long)
            threshs_t = torch.zeros((n_rules, max_len), dtype=torch.float32)
            signs_t = torch.zeros((n_rules, max_len), dtype=torch.float32)
            masks_t = torch.zeros((n_rules, max_len), dtype=torch.float32)

            for i, r in enumerate(raw_rules):
                for j, (f_idx, sign, thresh) in enumerate(r['conditions']):
                    feat_idxs_t[i, j] = int(f_idx)
                    threshs_t[i, j] = float(thresh)
                    signs_t[i, j] = float(sign)
                    masks_t[i, j] = 1.0
        else:
            # 处理没有规则生成的极端情况
            feat_idxs_t = torch.zeros((1, 1), dtype=torch.long)
            threshs_t = torch.zeros((1, 1), dtype=torch.float32)
            signs_t = torch.zeros((1, 1), dtype=torch.float32)
            masks_t = torch.zeros((1, 1), dtype=torch.float32)

        self.rule_feat_idxs = feat_idxs_t
        self.rule_threshs = threshs_t
        self.rule_signs = signs_t
        self.rule_masks = masks_t

        # --- C. Initialization ---
        self.premise_params = nn.Parameter(torch.ones(max(n_rules, 1)) * 1.0)  # Soft Init

        dim_con = 1 + self.n_features
        if self.order == 2:
            dim_con += self.n_features
            if self.interaction_pairs is not None:
                dim_con += self.interaction_pairs.shape[0]

        self.consequent_params = nn.Parameter(torch.randn(max(n_rules, 1), dim_con) * 0.01)
        self.is_initialized = True
        print(f">>> [Hybrid] Rule Base initialized with {n_rules} rules (Poly Dim={dim_con}).")

    def _apply_attention(self, x):
        if self.attention_weights.device != x.device:
            self.attention_weights.data = self.attention_weights.data.to(x.device)
        return x * self.attention_weights

    def fuzzify_and_infer(self, x):
        device = x.device
        if self.rule_feat_idxs.device != device:
            self.to(device)

        selected_x = x[:, self.rule_feat_idxs]
        threshs = self.rule_threshs.unsqueeze(0)
        signs = self.rule_signs.unsqueeze(0)
        mask = self.rule_masks.unsqueeze(0)
        beta = self.premise_params.view(1, -1, 1)

        z = beta * (selected_x - threshs) * signs
        mf_values = torch.sigmoid(z)
        masked_mf = mf_values * mask + (1.0 - mask)
        firing_strength = masked_mf.prod(dim=2)

        norm_factor = firing_strength.sum(dim=1, keepdim=True) + 1e-8
        normalized_w = firing_strength / norm_factor
        return normalized_w

    def _get_tsk_polynomials(self, x):
        """ Generate [1, x, x^2, selected_interactions] """
        batch_size = x.shape[0]
        ones = torch.ones((batch_size, 1), device=x.device)
        components = [x, ones]

        if self.order == 2:
            components.insert(1, x.pow(2))
            if self.interaction_pairs is not None:
                idx1 = self.interaction_pairs[:, 0]
                idx2 = self.interaction_pairs[:, 1]
                interactions = x[:, idx1] * x[:, idx2]
                components.insert(2, interactions)

        return torch.cat(components, dim=1)

    def initialize_consequents(self, X_t, y_t, rcond=1e-2, batch_size=2048):
        print(f"   [Phase 2] Optimizing Consequents (Robust LSE)...")
        device = X_t.device
        n_rules = self.rule_feat_idxs.shape[0]
        n_poly = self.consequent_params.shape[1]

        XtX_local = torch.zeros((n_rules, n_poly, n_poly), device=device)
        XtY_local = torch.zeros((n_rules, n_poly, 1), device=device)
        rule_activity = torch.zeros(n_rules, device=device)

        self.eval()
        with torch.no_grad():
            # [PROGRESS BAR]
            pbar = tqdm(range(0, X_t.shape[0], batch_size), desc="     LSE Accumulation", leave=False)
            for i in pbar:
                bx = X_t[i:i + batch_size]
                by = y_t[i:i + batch_size].view(-1, 1)

                bx_att = bx * self.attention_weights.to(device)
                w = self.fuzzify_and_infer(bx_att)
                p = self._get_tsk_polynomials(bx_att)

                rule_activity += w.sum(dim=0)
                XtX_batch = torch.einsum('br,bi,bj->rij', w, p, p)
                XtY_batch = torch.einsum('br,bi,bo->rio', w, p, by)

                XtX_local += XtX_batch
                XtY_local += XtY_batch
                del XtX_batch, XtY_batch, w, p

        new_weights = self.consequent_params.data.clone()
        active_count = 0
        I = torch.eye(n_poly, device=device).unsqueeze(0)

        for r in range(n_rules):
            if rule_activity[r] > 1.0:
                A = XtX_local[r] + rcond * I
                B = XtY_local[r]
                try:
                    theta = torch.linalg.solve(A, B)
                    new_weights[r] = theta.squeeze()
                    active_count += 1
                except:
                    pass

        self.consequent_params.data = new_weights
        print(f"   [Phase 2] LSE Solved for {active_count}/{n_rules} active rules.")

    def optimize_rule_base(self, X, threshold=0.01, batch_size=2048):
        print(f">>> [Pruning] Optimizing Rule Base...")
        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.attention_weights.device)
        else:
            X = X.to(self.attention_weights.device)

        rule_imp = torch.zeros(self.rule_feat_idxs.shape[0], device=X.device)
        with torch.no_grad():
            # [PROGRESS BAR]
            pbar = tqdm(range(0, X.shape[0], batch_size), desc="     Calculating Importance", leave=False)
            for i in pbar:
                bx = X[i:i + batch_size]
                bx = bx * self.attention_weights
                w = self.fuzzify_and_infer(bx)
                rule_imp += w.sum(dim=0)

        rule_imp /= X.shape[0]
        keep_mask = (rule_imp > threshold).cpu()

        if keep_mask.sum() > 0:
            kept = keep_mask.sum().item()
            print(f"   [Pruning] Kept {kept} rules (Pruned {len(keep_mask) - kept}).")
            self.rule_feat_idxs = self.rule_feat_idxs[keep_mask]
            self.rule_threshs = self.rule_threshs[keep_mask]
            self.rule_signs = self.rule_signs[keep_mask]
            self.rule_masks = self.rule_masks[keep_mask]
            self.premise_params = nn.Parameter(self.premise_params[keep_mask])
            self.consequent_params = nn.Parameter(self.consequent_params[keep_mask])
        else:
            print("   [Pruning] Threshold too high, keeping all rules.")

    def forward(self, x):
        x_w = self._apply_attention(x)
        w = self.fuzzify_and_infer(x_w)
        p = self._get_tsk_polynomials(x_w)
        rule_out = torch.einsum('bd,rd->br', p, self.consequent_params)
        output = (w * rule_out).sum(dim=1)
        return output

    def train_neuro_fuzzy(self, X, y, X_val=None, y_val=None, epochs=100, lr=0.01, batch_size=2048, patience=10):
        print(f">>> [Phase 3] Training...")
        if not isinstance(X, torch.Tensor): X = torch.tensor(X, dtype=torch.float32,
                                                             device=self.attention_weights.device)
        if not isinstance(y, torch.Tensor): y = torch.tensor(y, dtype=torch.float32,
                                                             device=self.attention_weights.device)
        if y.dim() == 1: y = y.view(-1, 1)

        monitor_val = False
        if X_val is not None:
            monitor_val = True
            if not isinstance(X_val, torch.Tensor): X_val = torch.tensor(X_val, dtype=torch.float32, device=X.device)
            if not isinstance(y_val, torch.Tensor): y_val = torch.tensor(y_val, dtype=torch.float32, device=y.device)

        opt = torch.optim.Adam([
            {'params': self.premise_params, 'lr': lr},
            {'params': self.consequent_params, 'lr': lr},
            {'params': self.attention_weights, 'lr': lr * 0.1}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        loss_fn = nn.MSELoss()

        best_loss = float('inf')
        no_imp = 0
        best_w = None

        # [PROGRESS BAR]
        epoch_pbar = tqdm(range(epochs), desc="Training", unit="ep")

        for ep in epoch_pbar:
            self.train()
            perm = torch.randperm(X.size(0), device=X.device)
            ep_loss = 0

            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i + batch_size]
                bx, by = X[idx], y[idx]

                opt.zero_grad()
                out = self.forward(bx)
                loss = loss_fn(out, by.view(-1))
                loss += 1e-5 * self.attention_weights.abs().sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.attention_weights.data.clamp_(min=0.0)
                opt.step()
                ep_loss += loss.item() * bx.size(0)

            train_loss = ep_loss / X.size(0)

            self.eval()
            cur_loss = train_loss
            val_loss_display = "-"

            if monitor_val:
                with torch.no_grad():
                    vout = self.forward(X_val)
                    cur_loss = loss_fn(vout, y_val.view(-1)).item()
                    val_loss_display = f"{cur_loss:.4f}"

            scheduler.step(cur_loss)
            cur_lr = opt.param_groups[0]['lr']

            # Update Progress Bar
            epoch_pbar.set_postfix({
                "Train MSE": f"{train_loss:.4f}",
                "Val MSE": val_loss_display,
                "LR": f"{cur_lr:.1e}"
            })

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_w = copy.deepcopy(self.state_dict())
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= patience:
                tqdm.write(f"   [Early Stop] No improvement for {patience} epochs.")
                break

        if best_w: self.load_state_dict(best_w)


LSE_XGANFIS = TreeANFIS