import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=24):
        """
        :param X: [N, F] 标准化后的特征矩阵
        :param y: [N] 标准化后的目标值
        :param seq_len: 回溯窗口长度 (例如过去24小时)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        # 如果总长度是 N，序列长度是 T，那么能生成的样本数是 N - T
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        # 输入: 从 i 到 i+seq_len (不包含) 的序列 -> [Seq_Len, Features]
        # 输出: 第 i+seq_len 个时刻的目标值 -> [1]
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len]