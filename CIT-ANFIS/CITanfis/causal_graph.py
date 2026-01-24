import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class HeuristicCausalGraph:
    def __init__(self, threshold=0.1, verbose=False):
        self.threshold = threshold
        self.verbose = verbose
        self.feature_names = None
        self.causal_roles = {}
        self.corr_matrix = None
        self.target_name = "Target"

    def fit(self, X, y, feature_names=None):
        print(f">>> [Heuristic Causal] Building dense causal graph (Threshold={self.threshold})...")
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"X{i}" for i in range(X.shape[1])]

        X_df = pd.DataFrame(X, columns=self.feature_names)
        X_df[self.target_name] = y

        # 使用 Spearman 计算相关性，并处理 NaN
        self.corr_matrix = X_df.corr(method='spearman').fillna(0.0)

        self.causal_roles = {}
        target_corrs = self.corr_matrix[self.target_name].drop(self.target_name)

        for feat in self.feature_names:
            idx = self.feature_names.index(feat)
            corr_val = abs(target_corrs[feat])

            # 强制清洗 NaN/Inf
            if np.isnan(corr_val) or np.isinf(corr_val):
                corr_val = 0.0

            relation = "Unrelated"
            if corr_val < self.threshold:
                relation = "Unrelated"
            else:
                if "Lag" in feat or "Roll" in feat:
                    relation = "Cause (History)"
                elif "Hour" in feat or "Day" in feat or "Month" in feat or "Is_" in feat:
                    relation = "Cause (Time)"
                elif "Diff" in feat or "EMA" in feat:
                    relation = "Cause (Trend)"
                else:
                    relation = "Correlated"

            self.causal_roles[idx] = relation
            if self.verbose:
                print(f"   {feat:<15} -> {self.target_name} : {corr_val:.4f} ({relation})")

        return self.causal_roles

    def plot_graph(self, save_path=None):
        if self.corr_matrix is None:
            print("Please run fit() first.")
            return

        G = nx.DiGraph()
        target_node = "Total Load"
        G.add_node(target_node, type='target')

        for idx, role in self.causal_roles.items():
            feat_name = self.feature_names[idx]
            if role == "Unrelated":
                continue

            weight = abs(self.corr_matrix.loc[feat_name, self.target_name])
            # [Fix] 再次确保权重有效
            if np.isnan(weight) or np.isinf(weight): weight = 0.001

            G.add_node(feat_name, type='feature', role=role)
            if "Cause" in role:
                G.add_edge(feat_name, target_node, weight=weight, color='#555555')
            elif "Correlated" in role:
                G.add_edge(feat_name, target_node, weight=weight, color='#999999', style='dashed')

        plt.figure(figsize=(14, 10), dpi=150)

        # [Fix] 布局计算可能产生 NaN，需要处理
        pos = nx.spring_layout(G, k=0.5, seed=42)
        pos[target_node] = np.array([0.0, 0.0])

        # 检查并修复坐标
        for node in pos:
            if not np.all(np.isfinite(pos[node])):
                pos[node] = np.array([np.random.rand(), np.random.rand()])

        # 圆形布局优化
        others = [n for n in G.nodes() if n != target_node]
        if others:
            radius = 1.0
            others_sorted = sorted(others, key=lambda x: G.nodes[x].get('role', ''))
            angle_step = 2 * np.pi / len(others_sorted)
            for i, node in enumerate(others_sorted):
                pos[node] = np.array([radius * np.cos(i * angle_step), radius * np.sin(i * angle_step)])

        # 颜色映射
        node_colors = []
        node_sizes = []
        for n in G.nodes():
            if n == target_node:
                node_colors.append('#FFD700')
                node_sizes.append(6000)
            else:
                role = G.nodes[n].get('role', '')
                if "History" in role:
                    node_colors.append('#90EE90')
                elif "Time" in role:
                    node_colors.append('#87CEFA')
                elif "Trend" in role:
                    node_colors.append('#FFB6C1')
                else:
                    node_colors.append('#D3D3D3')
                node_sizes.append(4000)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='#333333', linewidths=2)

        labels = {n: n.replace('_', '\n') if len(n) > 10 else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

        edges = G.edges(data=True)
        # [Fix] 确保绘图权重有限
        weights = [np.nan_to_num(d['weight']) * 5 for u, v, d in edges]
        edge_colors = [d['color'] for u, v, d in edges]

        nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors, arrowstyle='-|>', arrowsize=25,
                               connectionstyle='arc3,rad=0.1')

        legend_patches = [
            mpatches.Patch(color='#FFD700', label='Target (Total Load)'),
            mpatches.Patch(color='#90EE90', label='History Cause'),
            mpatches.Patch(color='#87CEFA', label='Time Cause'),
            mpatches.Patch(color='#FFB6C1', label='Trend Cause'),
        ]
        plt.legend(handles=legend_patches, loc='upper left', fontsize=12)
        plt.title("Heuristic Causal Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)  # 现在这里应该是安全的了

        # plt.show() # 如果不需要弹出窗口可注释