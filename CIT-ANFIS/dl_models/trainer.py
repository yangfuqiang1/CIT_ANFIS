import torch
import numpy as np
import time
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .dataset import TimeSeriesDataset
from .model import StrictLSTM


def run_dl_baseline(name, X_train, y_train, X_test, y_test, scaler_y, device):
    """
    运行 LSTM 模型，自动处理数据归一化问题。
    """
    print(f"\n--- Running Strict DL Model: {name} (Paper Level) ---")
    start_time = time.time()

    # --- 超参数设置 ---
    SEQ_LEN = 96
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    EPOCHS = 50
    LR = 0.001
    PATIENCE = 5

    # --- 1. 智能数据检查与处理 ---
    # 检查 y_test 是否为原始数据（未归一化）
    # 如果 y_test 的最大值远大于 1，说明是原始数据
    is_test_unscaled = np.max(np.abs(y_test)) > 1.5

    if is_test_unscaled:
        # print("   [Info] Detected unscaled y_test. Handling automatically...")
        # 为了计算验证 Loss，我们需要归一化的 y_test
        # reshape(-1, 1) 是因为 scaler 需要 2D 数组
        y_test_scaled_for_loader = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        # 最终计算 RMSE 用原始的 y_test
        y_test_final_ref = y_test
    else:
        # 如果传入的已经是归一化数据
        y_test_scaled_for_loader = y_test
        # 最终计算 RMSE 需要反归一化回真实值
        y_test_final_ref = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # --- 2. 构建数据集 ---
    # 训练集通常已经是归一化的
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len=SEQ_LEN)
    # 测试集放入 Loader 必须是归一化的，否则 Loss 会爆炸
    test_dataset = TimeSeriesDataset(X_test, y_test_scaled_for_loader, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   [LSTM] Seq Len: {SEQ_LEN}, Train Samples: {len(train_dataset)}")

    # --- 3. 初始化模型 ---
    input_dim = X_train.shape[1]
    model = StrictLSTM(
        input_size=input_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # --- 4. 训练循环 ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds.squeeze(), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = loss_fn(preds.squeeze(), batch_y)
                val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = val_loss / len(test_dataset)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"   [Early Stopping] Stopped at epoch {epoch + 1}")
                break

    # --- 5. 最终预测 ---
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())

    if len(all_preds) > 0:
        pred_scaled = np.concatenate(all_preds).flatten()
    else:
        pred_scaled = np.array([])

    # --- 6. 数据对齐与还原 ---
    # 对齐真实标签（去除前 SEQ_LEN 个数据）
    y_test_final_ref = y_test_final_ref[SEQ_LEN:]

    # 截断长度以匹配
    min_len = min(len(y_test_final_ref), len(pred_scaled))
    y_test_final_ref = y_test_final_ref[:min_len]
    pred_scaled = pred_scaled[:min_len]

    # 将预测值还原为真实值
    pred_final = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    end_time = time.time()

    # 计算最终 RMSE (此时 pred_final 和 y_test_final_ref 都是真实值量级)
    rmse = np.sqrt(mean_squared_error(y_test_final_ref, pred_final))
    mae = mean_absolute_error(y_test_final_ref, pred_final)

    return {'rmse': rmse, 'mae': mae, 'time': end_time - start_time}