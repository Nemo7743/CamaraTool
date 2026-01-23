import torch
from torchvision import transforms
# from PIL import Image  <-- 移除 PIL，因為效率較差
import cv2
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import os

# =========================================================
# 【關鍵修正 1】限制 CPU 執行緒數
# 防止 OpenCV 和 PyTorch 搶佔所有核心導致 100% 負載
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
cv2.setNumThreads(0)  # 0 表示由 OpenCV 自動決定最少執行緒，或設為 1

# =========================================================
# 1. 設定與模型載入s
# =========================================================
model_path = r"C:/LT_Model/model/model_best_td2_7.pth"
n_class = 3
classes = ['Hand', 'SafeItem', 'Tool']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用裝置: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = models.shufflenet_v2_x2_0(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_class)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =========================================================
# 2. 預處理定義 (優化版：移除 PIL 轉換)
# =========================================================
# 手動定義 Normalize 參數，以便在 Numpy 中直接計算
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =========================================================
# 3. 開啟攝影機
# =========================================================
cap = cv2.VideoCapture(0)
# 設定攝影機解析度，降低 CPU 讀取負擔 (視需求調整)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("無法開啟攝影機。")
    exit()

print("開始推論... CPU 優化模式")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================================================
    # 【關鍵修正 2】加速預處理 (Numpy -> Tensor)
    # 避免 cv2 -> PIL -> Tensor 的來回轉換
    # =========================================================
    frame_small = cv2.resize(frame, (128, 128))
    
    # BGR 轉 RGB
    img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    
    # 歸一化 (0~1) 並標準化 (Normalize) - 使用 Numpy 向量運算加速
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    
    # HWC (高,寬,色) -> CHW (色,高,寬)
    img = img.transpose((2, 0, 1))
    
    # 轉為 Tensor 並送入 GPU
    input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # =========================================================
    # 4. 模型推論
    # =========================================================
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
        # 這裡 .item() 會強迫 CPU 等待 GPU 完成，產生同步
        pred_index = pred.item()
        confidence = conf.item()
        
    label = classes[pred_index] if confidence > 0.5 else "Unknown"

    # =========================================================
    # 5. 顯示結果
    # =========================================================
    display_text = f"{label}: {confidence * 100:.1f}%"
    color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)

    cv2.putText(frame, display_text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Main Feed', frame)
    # cv2.imshow('Input', frame_small) # 若不需要看輸入圖，註解掉可省一點 CPU

    # 控制 FPS：waitKey(1) 表示只等待 1ms
    # 如果還是覺得 CPU 高，可以改為 waitKey(10) 強制休息
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
