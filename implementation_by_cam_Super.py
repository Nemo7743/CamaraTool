import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import os
import time

# =========================================================
# 【使用者設定區】
# =========================================================
CURRENT_MODE = "performance"
DEVICE_PREFERENCE = "cpu"
MODEL_PATH = r"C:\LT_Model\checkpoints\model_best_Nemo_8.pth"
ENABLE_AWB = True  # 是否開啟灰度世界自動白平衡

# =========================================================
# 0. 輔助函式：灰度世界白平衡
# =========================================================
def gray_world_awb(img):
    """
    實作灰度世界演算法 (Gray World Algorithm)
    假設影像的平均顏色為灰色，藉此修正色偏。
    """
    # 轉換為 float32 避免運算溢出
    img_float = img.astype(np.float32)
    
    # 計算各通道平均值
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    
    # 計算總平均值 (灰色基準)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # 防止除以零
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return img
    
    # 計算增益係數
    kb = avg_gray / avg_b
    kg = avg_gray / avg_g
    kr = avg_gray / avg_r
    
    # 應用增益並限制數值在 0~255
    img_float[:, :, 0] *= kb
    img_float[:, :, 1] *= kg
    img_float[:, :, 2] *= kr
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

# =========================================================
# 1. 環境與參數設定
# =========================================================
cam_index = 0
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

if CURRENT_MODE == "power_save":
    FRAME_SKIP = 3
    TARGET_FPS = 20
    WAIT_TIME = int(1000 / TARGET_FPS)
    print(f"啟動模式: [省電版]")
else:
    FRAME_SKIP = 0
    WAIT_TIME = 1
    print(f"啟動模式: [高效能版]")

# 裝置偵測
if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
    target_device = "cuda"
elif DEVICE_PREFERENCE == "cpu":
    target_device = "cpu"
else:
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(target_device)
print(f"運算裝置: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
else:
    torch.set_num_threads(4)

# =========================================================
# 2. 模型載入
# =========================================================
n_class = 4
classes = ['Block', 'Hand', 'SafeItem', 'Tool']
model = models.shufflenet_v2_x2_0(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_class)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("模型載入成功！")
except Exception as e:
    print(f"模型載入失敗: {e}")
    exit()

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =========================================================
# 3. 主迴圈
# =========================================================
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_label = "Initializing..."
last_conf = 0.0
last_color = (128, 128, 128)
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        should_infer = (frame_count % (FRAME_SKIP + 1) == 0)

        # --- 影像處理流程 ---
        # 1. 自動白平衡 (僅在推論幀執行，節省效能)
        processed_frame = frame.copy()
        if ENABLE_AWB:
            processed_frame = gray_world_awb(processed_frame)

        if should_infer:
            # 2. 縮放與格式轉換
            frame_small = cv2.resize(processed_frame, (128, 128))
            img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # 3. 正規化
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_norm = (img_norm - mean) / std
            img_norm = img_norm.transpose((2, 0, 1))
            
            # 4. 轉 Tensor 與推論
            if device.type == 'cuda':
                input_tensor = torch.from_numpy(img_norm).unsqueeze(0).pin_memory().to(device, non_blocking=True)
            else:
                input_tensor = torch.from_numpy(img_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                pred_index = pred.item()
                last_conf = conf.item()
            
            last_label = classes[pred_index] if last_conf > 0.5 else "Unknown"
            last_color = (0, 255, 0) if last_conf > 0.6 else (0, 0, 255)

        # --- 顯示 ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 繪製資訊 (在原始 frame 上繪製)
        cv2.rectangle(frame, (0, 0), (300, 95), (0, 0, 0), -1)
        text_label = f"{last_label}: {last_conf * 100:.1f}%"
        cv2.putText(frame, text_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
        
        awb_status = "AWB: ON" if ENABLE_AWB else "AWB: OFF"
        mode_str = "High Perf" if CURRENT_MODE == "performance" else "Power Save"
        cv2.putText(frame, f"FPS: {int(fps)} | {mode_str} | {awb_status}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 如果你想直接看白平衡後的畫面，可以改顯示 processed_frame
        cv2.imshow('Inference', frame)

        if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
