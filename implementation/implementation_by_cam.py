import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import os
import time
import sys # 用於安全退出

# =========================================================
# 【使用者設定區】在此切換模式
# =========================================================
# 信心域值
confidence = 0.80

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
n_class = 2
classes = ['Hand', 'noHand']

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

# 用於平滑數字的變數
smoothed_probs = np.zeros(n_class)
alpha = 0.2  # 平滑係數 (0.1~0.3 之間，愈小愈平滑但反應愈慢)

print("開始執行...")
print("【提示】若偵測到 Hand > 90%，畫面將自動暫停。請按【空白鍵】繼續。")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 決定是否執行推論 (跳幀機制)
        should_infer = (frame_count % (FRAME_SKIP + 1) == 0)

        if should_infer:
            # --- 預處理 ---
            frame_small = cv2.resize(frame, (128, 128))
            img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std
            img = img.transpose((2, 0, 1)) # HWC -> CHW
            
            if device.type == 'cuda':
                input_tensor = torch.from_numpy(img).unsqueeze(0).pin_memory().to(device, non_blocking=True)
            else:
                input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            # --- 推論 ---
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                
                # 取得所有類別的信心指數 (轉為 numpy)
                current_probs = probs.cpu().numpy()[0]
                
                # 平滑化處理：EMA 濾波器
                if frame_count == 1:
                    smoothed_probs = current_probs
                else:
                    smoothed_probs = alpha * current_probs + (1 - alpha) * smoothed_probs
                
                # 取得平滑後的最高者
                pred_index = np.argmax(smoothed_probs)
                last_conf = smoothed_probs[pred_index]
                last_label = classes[pred_index] if last_conf > 0.5 else "Unknown"
                last_color = (0, 255, 0) if last_conf > 0.6 else (0, 0, 255)

        # =========================================================
        # 繪圖與顯示
        # =========================================================
        
        # 顯示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 動態調整黑框高度以容納所有類別
        bg_height = 80 + (n_class * 25)
        cv2.rectangle(frame, (0, 0), (280, bg_height), (0, 0, 0), -1) 
        
        # 1. 顯示主結果 (最高信心度)
        text_label = f"Result: {last_label}"
        cv2.putText(frame, text_label, (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
        
        # 2. 顯示各類別詳細信心指數 (平滑後)
        for i, class_name in enumerate(classes):
            prob_text = f"- {class_name}: {smoothed_probs[i] * 100:.1f}%"
            y_pos = 70 + (i * 25)
            color = (255, 255, 255) if i == np.argmax(smoothed_probs) else (150, 150, 150)
            cv2.putText(frame, prob_text, (15, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # 3. 顯示 FPS 與 模式
        mode_str = "High Perf" if CURRENT_MODE == "performance" else "Power Save"
        device_str = "GPU" if device.type == "cuda" else "CPU"
        cv2.putText(frame, f"FPS: {int(fps)} | {mode_str} | {device_str}", (10, bg_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Inference', frame)

        # =========================================================
        # 新增功能：若 Hand > 90% 則暫停
        # =========================================================
        if last_label == 'Hand' and last_conf > confidence:
            # 在畫面上額外顯示暫停提示
            pause_h = 30
            cv2.putText(frame, f"PAUSED (Hand > {confidence*100}%)", (300, pause_h), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press SPACE to resume", (300, pause_h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            # 重新刷新畫面以顯示提示文字
            cv2.imshow('Inference', frame)
            
            print(f"偵測到 Hand ({last_conf*100:.1f}%)，暫停中... 請按空白鍵繼續")

            # 進入暫停迴圈，等待空白鍵
            while True:
                # waitKey(0) 表示無限等待
                key = cv2.waitKey(0) 
                
                if key == 32: # 32 是空白鍵 (Space) 的 ASCII 碼
                    print("繼續執行...")
                    # 恢復後稍微重置平滑值，避免一啟動因為舊的平均值太高馬上又暫停
                    # (選擇性，若不重置則會保留高信心值，可能瞬間又停)
                    smoothed_probs = np.zeros(n_class) 
                    frame_count = 0 # 確保下一幀會立即推論
                    break
                elif key == ord('q'): # 允許在暫停狀態下直接按 q 離開
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()

        # 一般的按鍵偵測 (沒暫停時)
        if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()