# -*- coding: utf-8 -*-
"""
dl_models/models.py
------------------
包含 4 个 Pro 版深度学习基准模型：
1. Bi-LSTM + Attention (时序捕捉王者)
2. Transformer (Sinusoidal PE + Deep Encoder, 并行计算王者)
3. DeepMLP (ResNet-MLP, 表格数据经典基线)
4. SimpleKAN (SiLU-based ResNet, 模拟 Kolmogorov-Arnold 网络特性)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 辅助模块: 正弦位置编码 (用于 Transformer) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


# ==========================================
# 模型 1: Bi-LSTM + Attention
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        # 双向 LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        # 注意力层 (因为是双向，所以输入维度是 hidden_dim * 2)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 输出头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]
        if x.dim() == 2: x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)  # [Batch, Seq, Hidden*2]

        # Attention Weights
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        # Context Vector
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(context_vector).squeeze()


# ==========================================
# 模型 2: Transformer (Pro)
# ==========================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :]).squeeze()


# ==========================================
# 模型 3: Deep MLP (Classic ResNet-MLP)
# ==========================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.1):
        super(DeepMLP, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 典型的残差 MLP 块: BN -> ReLU -> Dropout -> Linear
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # MLP 通常用 BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_blocks)
        ])

        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 处理时序输入，取平均作为特征
        if x.dim() == 3: x = x.mean(dim=1)

        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            out = block(x)
            x = out + residual  # ResNet 结构

        return self.output_head(x).squeeze()


# ==========================================
# 模型 4: Simple KAN (SiLU-based Network)
# ==========================================
class SimpleKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.7):
        super(SimpleKAN, self).__init__()

        # KAN 的特点：
        # 1. 激活函数都在边上 (这里用 SiLU 模拟)
        # 2. 通常不用 BatchNorm，而是用 LayerNorm
        # 3. 宽度比深度更重要 (虽然这里我们也堆叠了深度)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(), 
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_blocks)
        ])

        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 3: x = x.mean(dim=1)

        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            x = block(x) + residual

        return self.output_head(x).squeeze()