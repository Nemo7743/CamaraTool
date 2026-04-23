import os
import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 引入 QAT 必要工具 (來自 Code B)
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, QConfig, FakeQuantize, ObserverBase, fuse_modules
from torch.nn.utils.fusion import fuse_conv_bn_eval

# =========================================================
# 【使用者設定區】
# =========================================================
# 模式選項: "performance" (高效能) 或 "power_save" (省電)
CURRENT_MODE = "performance"

# 模型路徑 (請替換為您的實際 QAT 模型路徑)
MODEL_PATH = r"model\sm_1.6.pth"

# 指定雙攝影機索引
CAM_INDEX_1 = 1
CAM_INDEX_2 = 2

# 信心指數閾值 (超過此數值即顯示為綠色，0.5 代表 50%)
# 針對四個標籤分別設定獨立閾值
CONFIDENCE_THRESHOLDS = {
    'Hand': 0.75,
    'Tool': 0.8,
    'Block': 0.6,
    'Safe_Operation': 0.6
}

# 模型與量化參數 (來自 Code B)
CONFIG = {
    'num_classes': 4,
    'class_names': ['Hand', 'Tool', 'Block', 'Safe_Operation'],
    'target_conv5_channels': 960,
    'norm_mean': [0.485, 0.456, 0.406],
    'norm_std':  [0.229, 0.224, 0.225],
    'q_frac_weight': 8,
    'q_frac_act': 8
}

# =========================================================
# 0. 環境與參數設定 (來自 Code A)
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

if CURRENT_MODE == "power_save":
    FRAME_SKIP = 3        # 每 3 幀推論 1 次
    TARGET_FPS = 20       # 限制最高 FPS
    WAIT_TIME = int(1000 / TARGET_FPS)
    print(f"啟動模式: [省電版] (FPS限制: {TARGET_FPS}, 跳幀: {FRAME_SKIP})")
else:
    FRAME_SKIP = 0        # 不跳幀
    WAIT_TIME = 1         # 最小延遲
    print(f"啟動模式: [高效能版] (FPS無限制, 全速運算)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch 版本: {torch.__version__}")
print(f"運算裝置: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"GPU型號: {torch.cuda.get_device_name(0)}")

# =========================================================
# 1. QAT 模型結構重建工具 (來自 Code B)
# =========================================================
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

class DistillQATModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = original_model

    def forward(self, x):
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

def prune_shufflenet_conv5(model, target_channels=960):
    conv5_block = model.conv5
    conv = conv5_block[0]
    new_conv = nn.Conv2d(conv.in_channels, target_channels, kernel_size=1, stride=1, padding=0, bias=False)
    new_bn = nn.BatchNorm2d(target_channels)
    model.conv5[0] = new_conv
    model.conv5[1] = new_bn
    model.fc = nn.Linear(target_channels, CONFIG['num_classes'])
    return model

def fuse_normalization_to_conv1(model, mean, std):
    print("🔨 Fusing Normalization (Mean/Std) into Conv1...")
    conv = model.conv1[0]
    mean_t = torch.tensor(mean).view(3, 1, 1).to(conv.weight.device)
    std_t = torch.tensor(std).view(3, 1, 1).to(conv.weight.device)
    with torch.no_grad():
        conv.weight.data.div_(std_t)
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
        weight_sum = conv.weight.data.sum(dim=(2, 3))
        bias_adjustment = (weight_sum * mean_t.squeeze()).sum(dim=1)
        conv.bias.data.sub_(bias_adjustment)
    return model

def create_deploy_model(model):
    print("🔨 Converting to deploy mode (FusedConv + ReLU)...")
    deploy_model = copy.deepcopy(model)
    deploy_model.eval()
    def _recursive_fuse(module):
        for name, child in module.named_children():
            if isinstance(child, torch.ao.nn.intrinsic.qat.ConvBnReLU2d):
                fused_conv = fuse_conv_bn_eval(child, child.bn)
                if hasattr(child, 'weight_fake_quant'): fused_conv.weight_fake_quant = child.weight_fake_quant
                replaced_module = nn.Sequential(fused_conv, nn.ReLU(inplace=True))
                setattr(module, name, replaced_module)
            elif isinstance(child, torch.ao.nn.intrinsic.qat.ConvBn2d):
                fused_conv = fuse_conv_bn_eval(child, child.bn)
                if hasattr(child, 'weight_fake_quant'): fused_conv.weight_fake_quant = child.weight_fake_quant
                setattr(module, name, fused_conv)
            else:
                _recursive_fuse(child)
    _recursive_fuse(deploy_model)
    return deploy_model

# =========================================================
# 2. 模型建立與載入 (融合 Code B 邏輯)
# =========================================================
print("🏗️ Reconstructing Student Model (X0.5)...")
base_model = models.shufflenet_v2_x0_5(weights=None)
base_model = prune_shufflenet_conv5(base_model, target_channels=CONFIG['target_conv5_channels'])
base_model = fuse_normalization_to_conv1(base_model, CONFIG['norm_mean'], CONFIG['norm_std'])

model = DistillQATModel(base_model)
model.qconfig = get_fixed_point_qconfig(CONFIG['q_frac_weight'], CONFIG['q_frac_act'])

# 執行融合 (重現 QAT 結構)
model.eval()
fuse_modules(model.model, [['conv1.0', 'conv1.1', 'conv1.2']], inplace=True)
fuse_modules(model.model, [['conv5.0', 'conv5.1']], inplace=True)

for name, module in model.model.named_modules():
    if isinstance(module, models.shufflenetv2.InvertedResidual):
        for i in range(len(module.branch1)):
            if isinstance(module.branch1[i], nn.Conv2d):
                fuse_modules(module.branch1, [str(i), str(i+1)], inplace=True) 
        for i in range(len(module.branch2)):
            if isinstance(module.branch2[i], nn.Conv2d):
                if i+1 < len(module.branch2) and isinstance(module.branch2[i+1], nn.BatchNorm2d):
                    if i+2 < len(module.branch2) and isinstance(module.branch2[i+2], nn.ReLU):
                        fuse_modules(module.branch2, [str(i), str(i+1), str(i+2)], inplace=True)
                    else:
                        fuse_modules(module.branch2, [str(i), str(i+1)], inplace=True)

model.train()
prepare_qat(model, inplace=True)

# 載入權重
print(f"🔄 Loading weights from {MODEL_PATH}...")
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ 模型載入失敗: {e}")
        exit()
else:
    print(f"找不到模型檔案: {MODEL_PATH}")
    exit()

# 轉換為部署模式並放至裝置
model = create_deploy_model(model)
model = model.to(device)
model.eval()

# =========================================================
# 3. 雙攝影機主迴圈
# =========================================================
cap1 = cv2.VideoCapture(CAM_INDEX_1, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap2 = cv2.VideoCapture(CAM_INDEX_2, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap1.isOpened():
    print(f"無法開啟攝影機 1 (Index {CAM_INDEX_1})。")
if not cap2.isOpened():
    print(f"無法開啟攝影機 2 (Index {CAM_INDEX_2})。")

if not cap1.isOpened() and not cap2.isOpened():
    print("雙鏡頭皆無法開啟，結束程式。")
    exit()

def draw_results(frame, probs, fps_val, title="Camera"):
    """輔助函式：將推論結果繪製在畫面上"""
    bg_height = (CONFIG['num_classes'] * 30) + 40
    cv2.rectangle(frame, (0, 0), (280, bg_height), (0, 0, 0), -1)
    
    mode_str = "High Perf" if CURRENT_MODE == "performance" else "Power Save"
    cv2.putText(frame, f"{title} | FPS: {int(fps_val)} | {mode_str}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if probs is not None:
        y_pos = 50 
        for i in range(CONFIG['num_classes']):
            class_name = CONFIG['class_names'][i]
            score = probs[i].item()
            
            if score >= CONFIDENCE_THRESHOLDS[class_name]:
                color = (0, 255, 0) # 綠色
                thickness = 2
            else:
                color = (0, 0, 255) # 紅色
                thickness = 1
            
            text_str = f"{class_name}: {score*100:.1f}%"
            cv2.putText(frame, text_str, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
            y_pos += 30
    else:
        cv2.putText(frame, "Initializing...", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return frame

frame_count = 0
prev_time = time.time()
last_probs1 = None 
last_probs2 = None

print("開始執行雙鏡頭即時推論... 按 'q' 離開")

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2:
            break
        
        frame_count += 1
        should_infer = (frame_count % (FRAME_SKIP + 1) == 0)

        if should_infer:
            frames_to_infer = []
            if ret1: frames_to_infer.append(frame1)
            if ret2: frames_to_infer.append(frame2)

            tensors = []
            for f in frames_to_infer:
                frame_small = cv2.resize(f, (128, 128))
                img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 256.0
                tensors.append(img_tensor)

            if len(tensors) > 0:
                batch_tensor = torch.stack(tensors)
                if device.type == 'cuda':
                    batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)
                else:
                    batch_tensor = batch_tensor.to(device)

                # 批次推論
                with torch.inference_mode():
                    outputs = model(batch_tensor)
                    probs = torch.sigmoid(outputs) 
                    
                idx = 0
                if ret1:
                    last_probs1 = probs[idx].cpu()
                    idx += 1
                if ret2:
                    last_probs2 = probs[idx].cpu()

        # 計算 FPS
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # 繪圖
        display_frames = []
        if ret1:
            frame1_drawn = draw_results(frame1, last_probs1, fps_val, "Cam 1")
            display_frames.append(frame1_drawn)
        if ret2:
            frame2_drawn = draw_results(frame2, last_probs2, fps_val, "Cam 2")
            display_frames.append(frame2_drawn)

        # 顯示
        if len(display_frames) == 2:
            # 確保高度一致，然後水平拼接
            h1, w1 = display_frames[0].shape[:2]
            h2, w2 = display_frames[1].shape[:2]
            if h1 != h2:
                # 若攝影機高度不同，配合第一台攝影機縮放
                display_frames[1] = cv2.resize(display_frames[1], (int(w2 * h1 / h2), h1))
            combined_frame = np.hstack((display_frames[0], display_frames[1]))
            cv2.imshow('Dual Camera QAT Live Inference', combined_frame)
        elif len(display_frames) == 1:
            cv2.imshow('Single Camera QAT Live Inference', display_frames[0])

        if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
