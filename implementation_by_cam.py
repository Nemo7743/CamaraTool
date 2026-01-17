import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from ShuffleNetV2 import shufflenetv2

# =========================================================
# 1. 設定與模型載入 (維持不變)
# =========================================================
model_path = r"model_old/model_keep/model_best_xs1_1.pth"
width_mult = 0.5
n_class = 2
classes = ['hand', 'no_hand']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = shufflenetv2(n_class, width_mult)
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# =========================================================
# 2. 預處理定義
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================================================
# 3. 開啟攝影機
# =========================================================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("無法開啟攝影機，請檢查索引號是否正確。")
    exit()

print("開始即時推論... 按下 'q' 鍵退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 新增：將原始影像縮放至 128x128 用於顯示與推論 ---
    frame_small = cv2.resize(frame, (128, 128))
    
    # 轉換色彩空間供模型使用 (BGR -> RGB)
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Tensor 轉換
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # =========================================================
    # 4. 模型推論
    # =========================================================
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        
    label = classes[pred]

    # =========================================================
    # 5. 顯示結果
    # =========================================================
    # 在原始大視窗顯示結果 (640x480)
    cv2.putText(frame, f"Pred: {label}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Main Feed (640x480)', frame)

    # --- 新增：顯示縮放後的視窗 (128x128) ---
    cv2.imshow('Model Input (128x128)', frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
