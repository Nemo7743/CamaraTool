import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import os
import time


# =========================================================
# 【使用者設定區】在此切換模式
# =========================================================
# 模式選項: "performance" (高效能) 或 "power_save" (省電)
CURRENT_MODE = "performance"

# 運算裝置選項: "auto" (自動偵測), "cuda" (強制GPU), "cpu" (強制CPU)
DEVICE_PREFERENCE = "cpu"

# 模型路徑 (請確認路徑正確)
MODEL_PATH = r"C:\LT_Model\checkpoints\3.0_project_gray.pth"

# =========================================================
# 0. 環境與參數設定
# =========================================================
cam_index = 0

# 防止 CPU 多執行緒搶佔資源
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

# 根據模式設定參數
if CURRENT_MODE == "power_save":
    FRAME_SKIP = 3        # 每 3 幀推論 1 次 (大幅降低負載)
    TARGET_FPS = 20       # 限制最高 FPS
    WAIT_TIME = int(1000 / TARGET_FPS)
    print(f"啟動模式: [省電版] (FPS限制: {TARGET_FPS}, 跳幀: {FRAME_SKIP})")
else:
    FRAME_SKIP = 0        # 不跳幀，每幀都推論
    WAIT_TIME = 1         # 最小延遲
    print(f"啟動模式: [高效能版] (FPS無限制, 全速運算)")

# 檢查 CUDA 狀態與處理使用者裝置偏好
if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
    target_device = "cuda"
elif DEVICE_PREFERENCE == "cpu":
    target_device = "cpu"
else:
    # Auto 模式，或是選了 cuda 但沒硬體支援時的 fallback
    if DEVICE_PREFERENCE == "cuda":
        print("警告: 您選擇了 GPU 但系統未偵測到 CUDA，已自動切換回 CPU。")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(target_device)
print(f"PyTorch 版本: {torch.__version__}")
print(f"運算裝置: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # 加速固定尺寸輸入的網路
    torch.set_num_threads(1) # GPU 模式下，CPU 設為 1 核負責傳輸即可
    print(f"GPU型號: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(4) # CPU 模式下，多開幾核加速

# =========================================================
# 1. 模型載入
# =========================================================
#n_class = 4
#classes = ['Block', 'Hand', 'SafeItem', 'Tool']

n_class = 2
classes = ['Hand', 'Hand']

# 建立模型結構 (需與權重檔匹配)
model = models.shufflenet_v2_x2_0(weights=None) # 注意這裡若報錯，請改回 x1_0
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_class)

# 載入權重
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("模型載入成功！")
except Exception as e:
    print(f"模型載入失敗: {e}")
    print("請檢查路徑或是 shufflenet 版本(x1_0 vs x2_0)")
    exit()

# =========================================================
# 2. 預處理常數定義 (Numpy 向量化加速)
# =========================================================
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =========================================================
# 3. 主迴圈
# =========================================================
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("無法開啟攝影機。")
    exit()

# 初始化變數
frame_count = 0
last_label = "Initializing..."
last_conf = 0.0
last_color = (128, 128, 128)
prev_time = time.time()

print("開始執行...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 決定是否執行推論 (跳幀機制)
        # 如果是高效能模式，FRAME_SKIP 為 0，條件永遠成立
        should_infer = (frame_count % (FRAME_SKIP + 1) == 0)

        if should_infer:
            # --- 預處理 ---
            frame_small = cv2.resize(frame, (128, 128))
            img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Normalize (Numpy 向量運算)
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std
            img = img.transpose((2, 0, 1)) # HWC -> CHW
            
            # 轉 Tensor
            if device.type == 'cuda':
                # 【優化】使用 pin_memory 和 non_blocking 加速 CPU->GPU 傳輸
                input_tensor = torch.from_numpy(img).unsqueeze(0).pin_memory().to(device, non_blocking=True)
            else:
                input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            # --- 推論 ---
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                # 取得結果
                pred_index = pred.item()
                last_conf = conf.item()
            
            last_label = classes[pred_index] if last_conf > 0.5 else "Unknown"
            last_color = (0, 255, 0) if last_conf > 0.6 else (0, 0, 255)

        # =========================================================
        # 繪圖與顯示 (使用當前或上一次的結果)
        # =========================================================
        
        # 顯示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.rectangle(frame, (0, 0), (250, 90), (0, 0, 0), -1) # 背景黑框方便閱讀
        
        # 顯示標籤
        text_label = f"{last_label}: {last_conf * 100:.1f}%"
        cv2.putText(frame, text_label, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
        
        # 顯示 FPS 與 模式
        mode_str = "High Perf" if CURRENT_MODE == "performance" else "Power Save"
        device_str = "GPU" if device.type == "cuda" else "CPU"
        cv2.putText(frame, f"FPS: {int(fps)} | {mode_str} | {device_str}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Inference', frame)

        # FPS 控制
        if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
