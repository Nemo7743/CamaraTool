import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import os
import time
import copy

# 引入必要的量化與融合工具
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, QConfig, FakeQuantize, ObserverBase, fuse_modules
from torch.nn.utils.fusion import fuse_conv_bn_eval

# =========================================================
# 【使用者設定區】在此切換模式與路徑
# =========================================================
CURRENT_MODE = "performance"  # "performance" or "power_save"

# ⚠️ 請指向 BN-Folding 後的 .pth 檔
MODEL_PATH = r"checkpoints_qat_balanced/best_student_fused_deploy.pth"

# 偵測框大小設定
ROI_SIZE = 1000
ROI_HALF = ROI_SIZE // 2

# QAT 設定 (必須與訓練時一致，用於重建結構)
Q_FRAC_WEIGHT = 8
Q_FRAC_ACT = 8
TARGET_CONV5_CHANNELS = 960
NUM_CLASSES = 4
CLASSES = ['Block', 'Hand', 'SafeItem', 'Tool'] # 請確認類別順序與訓練時一致

# =========================================================
# 1. 模型結構定義與工具函數 (必須包含才能載入結構)
# =========================================================

# --- A. Observer ---
class StaticFixedPointObserver(ObserverBase):
    def __init__(self, frac_bits, quant_min=-32768, quant_max=32767, dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, **kwargs):
        super().__init__(dtype=dtype)
        self.frac_bits = frac_bits
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        scale_val = 1.0 / (2 ** frac_bits)
        self.register_buffer('fixed_scale', torch.tensor([scale_val]))
        self.register_buffer('fixed_zp', torch.tensor([0], dtype=torch.int32))
    def forward(self, x): return x
    def calculate_qparams(self): return self.fixed_scale, self.fixed_zp

def get_fixed_point_qconfig(frac_weight, frac_act):
    weight_fq = FakeQuantize.with_args(observer=StaticFixedPointObserver, quant_min=-32768, quant_max=32767, dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, frac_bits=frac_weight)
    act_fq = FakeQuantize.with_args(observer=StaticFixedPointObserver, quant_min=-32768, quant_max=32767, dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, frac_bits=frac_act)
    return QConfig(activation=act_fq, weight=weight_fq)

# --- B. 模型 Wrapper ---
class DistillQATModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = original_model

    def forward(self, x):
        # 注意：雖然部署模型通常不需要 QuantStub，但為了結構 Key 匹配，我們保留它
        # 實際上對於 Float 輸入，QuantStub 不會有副作用 (除非我們啟用量化推論)
        x = self.quant(x)
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        x = x.mean([2, 3])
        x = self.model.fc(x)
        x = self.dequant(x)
        return x

# --- C. 剪枝與融合工具 ---
def prune_shufflenet_conv5(model, target_channels):
    conv5_block = model.conv5
    conv = conv5_block[0]
    # 重建結構即可，不需要排序權重
    new_conv = nn.Conv2d(conv.in_channels, target_channels, kernel_size=1, stride=1, padding=0, bias=False)
    new_bn = nn.BatchNorm2d(target_channels)
    model.conv5[0] = new_conv
    model.conv5[1] = new_bn
    model.fc = nn.Linear(target_channels, NUM_CLASSES)
    return model

def fuse_qat_model_for_deploy(model):
    """將 QAT 模型結構轉換為無 BN 的部署結構"""
    fused_model = copy.deepcopy(model)
    fused_model.eval()
    def _recursive_fuse(module):
        for name, child in module.named_children():
            if isinstance(child, (torch.ao.nn.intrinsic.qat.ConvBnReLU2d, torch.ao.nn.intrinsic.qat.ConvBn2d)):
                # 執行融合：這會產生一個標準的 nn.Conv2d
                fused_conv = fuse_conv_bn_eval(child, child.bn)
                # 繼承屬性以防萬一
                if hasattr(child, 'weight_fake_quant'): fused_conv.weight_fake_quant = child.weight_fake_quant
                setattr(module, name, fused_conv)
            else:
                _recursive_fuse(child)
    _recursive_fuse(fused_model)
    return fused_model

# =========================================================
# 2. 載入模型邏輯
# =========================================================
def load_my_fused_model(model_path, device):
    print(f"正在重建模型結構以載入: {model_path}")
    
    # 移除 try-except 以便看到其他潛在錯誤
    # 1. 基底模型 (X0.5)
    base_model = models.shufflenet_v2_x0_5(weights=None)
    
    # 2. 重現剪枝
    base_model = prune_shufflenet_conv5(base_model, target_channels=TARGET_CONV5_CHANNELS)
    
    # 3. 包裝 QAT
    qat_model = DistillQATModel(base_model)
    
    # 設定 QConfig
    qat_model.qconfig = get_fixed_point_qconfig(Q_FRAC_WEIGHT, Q_FRAC_ACT)
    
    # 4. 重現 PyTorch 融合
    # [步驟 A] fuse_modules 必須在 Eval 模式下進行
    qat_model.eval()
    fuse_modules(qat_model.model, [['conv1.0', 'conv1.1', 'conv1.2']], inplace=True)
    
    # [步驟 B] ✅【關鍵修正】prepare_qat 必須在 Train 模式下進行
    qat_model.train()
    prepare_qat(qat_model, inplace=True)
    
    # 5. 重現部署融合 (ConvBn -> Conv)
    # 這一步會產生與 pth 檔匹配的結構
    deploy_model = fuse_qat_model_for_deploy(qat_model)
    
    # 6. 載入權重
    state_dict = torch.load(model_path, map_location=device)
    deploy_model.load_state_dict(state_dict)
    
    deploy_model.to(device)
    deploy_model.eval() # 最後推論時切回 Eval
    
    return deploy_model

# =========================================================
# 3. 環境初始化
# =========================================================
cam_index = 1
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

if CURRENT_MODE == "power_save":
    FRAME_SKIP = 3
    TARGET_FPS = 20
    WAIT_TIME = int(1000 / TARGET_FPS)
else:
    FRAME_SKIP = 0
    WAIT_TIME = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"運算裝置: {device}")

# =========================================================
# 4. 初始化模型
# =========================================================
if not os.path.exists(MODEL_PATH):
    print(f"❌ 錯誤: 找不到模型檔案 {MODEL_PATH}")
    print("請確認路徑是否正確，或是否已執行 export_weights 腳本。")
    exit()

model = load_my_fused_model(MODEL_PATH, device)
if model is None:
    exit()
print("✅ Fused 模型載入成功！(Normalization 與 BN 已融合)")

# =========================================================
# 5. 主迴圈
# =========================================================
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("無法開啟攝影機。")
    exit()

frame_count = 0
last_label = "Initializing..."
last_conf = 0.0
last_color = (128, 128, 128)
prev_time = time.time()

print("開始執行...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # ROI 設定
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        x1 = max(0, center_x - ROI_HALF)
        y1 = max(0, center_y - ROI_HALF)
        x2 = min(w, center_x + ROI_HALF)
        y2 = min(h, center_y + ROI_HALF)

        frame_count += 1
        should_infer = (frame_count % (FRAME_SKIP + 1) == 0)

        if should_infer:
            roi_img = frame[y1:y2, x1:x2]
            
            if roi_img.size != 0:
                # 1. 縮放
                frame_small = cv2.resize(roi_img, (128, 128))
                
                # 2. 轉色域與正規化
                img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                
                # ⚠️【關鍵修改】Normalization 已融合，只需轉為 0~1
                # 這裡不需要減 mean 除 std 了！
                img = img.astype(np.float32) / 255.0
                
                img = img.transpose((2, 0, 1))
                
                # 3. 推論
                if device.type == 'cuda':
                    input_tensor = torch.from_numpy(img).unsqueeze(0).to(device, non_blocking=True)
                else:
                    input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                    pred_index = pred.item()
                    last_conf = conf.item()
                
                # 閾值判斷
                if last_conf > 0.5:
                    last_label = CLASSES[pred_index]
                    # 不同類別給不同顏色
                    if last_label == 'SafeItem': last_color = (255, 0, 0) # 藍色
                    elif last_label == 'Hand': last_color = (0, 0, 255)   # 紅色 (危險)
                    else: last_color = (0, 255, 0) # 綠色
                else:
                    last_label = "Unknown"
                    last_color = (128, 128, 128)
            else:
                last_label = "ROI Error"

        # =========================================================
        # 繪圖與顯示
        # =========================================================
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # 資訊面板
        cv2.rectangle(frame, (0, 0), (280, 100), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Class: {last_label}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_color, 2)
        cv2.putText(frame, f"Conf:  {last_conf * 100:.1f}%", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        mode_str = "High Perf" if CURRENT_MODE == "performance" else "Power Save"
        cv2.putText(frame, f"FPS: {int(fps)} | {mode_str}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 繪製 ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        cv2.imshow('Fused Model Inference', frame)

        if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
