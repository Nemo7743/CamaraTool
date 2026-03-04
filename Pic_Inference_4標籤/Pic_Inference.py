import os
import copy
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models 
import numpy as np
from PIL import Image

# 引入 QAT 必要工具
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, QConfig, FakeQuantize, ObserverBase, fuse_modules
from torch.nn.utils.fusion import fuse_conv_bn_eval

# ==========================================
# 參數設定
# ==========================================
# 💡 在這裡手動選擇推論設備
# 'auto': 自動偵測 (有 GPU 用 GPU，否則 CPU)
# 'cpu' : 強制使用 CPU
# 'cuda': 強制使用 GPU
USE_DEVICE = 'GPU'  # <--- 請在這裡更改您想使用的硬體

MODEL_PATH = "model_best_td4_2_1.pth" 
IMAGE_PATH = "./test_image.jpg"  # 替換為你要測試的單張圖片路徑

CONFIG = {
    'num_classes': 4,
    'class_names': ['Hand', 'Tool', 'Block', 'Safe_Operation'],
    'target_conv5_channels': 960,
    # ImageNet Mean/Std (用於重建融合後的 Conv1)
    'norm_mean': [0.485, 0.456, 0.406],
    'norm_std':  [0.229, 0.224, 0.225],
    # Q8.8 格式設定
    'q_frac_weight': 8,
    'q_frac_act': 8
}

# ==========================================
# 1. Transform (模擬硬體)
# ==========================================
class HardwareSimulateTransform:
    """
    模擬 ZCU104 硬體輸入：
    1. 讀取圖片 RGB (0-255)
    2. 右移 8 bits (除以 256.0)
    3. 不做標準 Normalize (因為已經融合進模型權重)
    """
    def __call__(self, pic):
        img_tensor = transforms.functional.pil_to_tensor(pic).float()
        img_tensor = img_tensor / 256.0
        return img_tensor

# ==========================================
# 2. QAT 相關工具 (必須與訓練代碼一致)
# ==========================================
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

# ==========================================
# 3. 模型結構重建工具
# ==========================================
def prune_shufflenet_conv5(model, target_channels=960):
    conv5_block = model.conv5
    conv = conv5_block[0]
    bn = conv5_block[1]
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

# ==========================================
# 主程式
# ==========================================
def main():
    # 🌟 設備選擇邏輯
    if USE_DEVICE.lower() == 'cpu':
        device = torch.device("cpu")
    elif USE_DEVICE.lower() == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("⚠️ 警告: 找不到 CUDA 設備，強制降級使用 CPU。")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"🟢 Using device: {device}")
    
    # 1. 重建模型 (步驟必須與訓練完全一致！)
    print("🏗️ Reconstructing Student Model (X0.5)...")
    base_model = models.shufflenet_v2_x0_5(weights=None)
    
    base_model = prune_shufflenet_conv5(base_model, target_channels=CONFIG['target_conv5_channels'])
    base_model = fuse_normalization_to_conv1(base_model, CONFIG['norm_mean'], CONFIG['norm_std'])
    
    model = DistillQATModel(base_model)
    model.qconfig = get_fixed_point_qconfig(CONFIG['q_frac_weight'], CONFIG['q_frac_act'])
    
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
    
    # 2. 載入權重
    print(f"🔄 Loading weights from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    try:
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except RuntimeError as e:
        print(f"⚠️ Error loading state_dict: {e}")
        return
    
    # 3. 轉換為部署模式
    model = create_deploy_model(model)
    model = model.to(device)
    model.eval()

    # 4. 準備單張圖片
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file not found at {IMAGE_PATH}")
        return

    print(f"🖼️ Loading image from {IMAGE_PATH}...")
    image = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        HardwareSimulateTransform()
    ])
    
    # 加上 batch 維度 (1, C, H, W)
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 5. 執行推論與計算時間
    print("🚀 Starting Single Image Inference...")
    
    # 暖機 (Warm-up) 避免第一次推論時間包含初始化 GPU 資源的延遲
    with torch.inference_mode():
        for _ in range(3):
            _ = model(img_tensor)

    # 🌟 確保只有在實際使用 GPU 時才進行同步
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(img_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()

    # 6. 解析結果
    inference_time_ms = (end_time - start_time) * 1000
    preds = (outputs > 0).float().cpu().numpy()[0] # 取出 batch 中第一筆資料

    print("\n📊 Inference Results:")
    print("   ----------------------------")
    print(f"   ⏱️ 推論時間 (Inference Time): {inference_time_ms:.2f} ms")
    print("   ----------------------------")
    print("   🎯 預測標籤 (Predictions):")
    for i, class_name in enumerate(CONFIG['class_names']):
        status = "✅ 偵測到 (1)" if preds[i] == 1 else "❌ 未偵測 (0)"
        print(f"     - {class_name:<15}: {status}")
    print("   ----------------------------")

if __name__ == "__main__":
    main()