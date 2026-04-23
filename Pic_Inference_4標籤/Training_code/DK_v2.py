import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, QConfig, FakeQuantize, ObserverBase, fuse_modules
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from sklearn.metrics import f1_score, recall_score
import argparse

# ==========================================
# 1. 全局配置 (慢工出細活版)
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation + QAT (慢工出細活版 - CosineAnnealingWarmRestarts)")
    parser.add_argument('--data-root',          type=str,   default='./DowDwen_set_resized')
    parser.add_argument('--train-csv',          type=str,   default='train.csv')
    parser.add_argument('--val-csv',            type=str,   default='val.csv')
    parser.add_argument('--teacher-model-path', type=str,   default='./model/checkpoint/teacher_model/teacher_model_1.4/model_best.pth',
                                                            help='教師模型權重路徑（1.4版）')
    parser.add_argument('--batch-size',         type=int,   default=128)
    parser.add_argument('--learning-rate',      type=float, default=1e-3,  help='Learning rate (OneCycleLR max)')
    parser.add_argument('--epochs',             type=int,   default=60,    help='訓練 Epoch 數')
    parser.add_argument('--alpha',              type=float, default=2.0,   help='Soft Loss 權重（教師知識）')
    parser.add_argument('--cosine-t0',          type=int,   default=40,    help='CosineAnnealingWarmRestarts 重啟週期')
    parser.add_argument('--eta-min',            type=float, default=1e-7,  help='Cosine Annealing 最低學習率')
    parser.add_argument('--patience',           type=int,   default=30,    help='Early Stopping 耐心 (0=關閉)')
    parser.add_argument('--target-conv5-channels', type=int, default=960)
    parser.add_argument('--save-dir',           type=str,   default='checkpoints_distill_v2')
    parser.add_argument('--num-workers',        type=int,   default=4)
    parser.add_argument('--q-frac-weight',      type=int,   default=8)
    parser.add_argument('--q-frac-act',         type=int,   default=8)
    args = parser.parse_args()

    return {
        'data_root':             args.data_root,
        'train_csv':             args.train_csv,
        'val_csv':               args.val_csv,
        'teacher_model_path':    args.teacher_model_path,
        'num_classes':           4,
        'class_names':           ['Hand', 'Tool', 'Block', 'Safe_Operation'],
        'batch_size':            args.batch_size,
        'learning_rate':         args.learning_rate,
        'epochs':                args.epochs,
        'alpha':                 args.alpha,
        'cosine_t0':             args.cosine_t0,
        'eta_min':               args.eta_min,
        'patience':              args.patience,
        'target_conv5_channels': args.target_conv5_channels,
        'device':                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'save_dir':              args.save_dir,
        'norm_mean':             [0.485, 0.456, 0.406],
        'norm_std':              [0.229, 0.224, 0.225],
        'q_frac_weight':         args.q_frac_weight,
        'q_frac_act':            args.q_frac_act,
        'num_workers':           args.num_workers,
    }

CONFIG = get_args()

# ==========================================
# 2. Dataset & Transforms
# ==========================================
class HardwareSimulateTransform:
    def __call__(self, pic):
        img_tensor = transforms.functional.pil_to_tensor(pic).float()
        img_tensor = img_tensor / 256.0
        return img_tensor

class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = CONFIG['class_names']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        labels = row[self.label_cols].values.astype(np.float32)
        target = torch.tensor(labels)
        if self.transform:
            image = self.transform(image)
        return image, target

def get_pos_weights(dataset):
    """類別補償權重，限制上限 5.0 防止誤報爆炸。"""
    df = dataset.df
    labels = df[dataset.label_cols].values
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(df) - pos_counts
    pos_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)
    pos_weights = np.clip(pos_weights, 1.0, 10.0)  # ★ 學生模型需要較強的補償 (10.0)
    print(f"⚖️ Pos Weights (capped 5.0): {pos_weights}")
    return torch.tensor(pos_weights, dtype=torch.float32)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# ==========================================
# 3. QAT Observer
# ==========================================
class StaticFixedPointObserver(ObserverBase):
    def __init__(self, frac_bits, quant_min=-32768, quant_max=32767,
                 dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, **kwargs):
        super().__init__(dtype=dtype)
        self.frac_bits = frac_bits
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.register_buffer('fixed_scale', torch.tensor([1.0 / (2 ** frac_bits)]))
        self.register_buffer('fixed_zp', torch.tensor([0], dtype=torch.int32))

    def forward(self, x): return x
    def calculate_qparams(self): return self.fixed_scale, self.fixed_zp

def get_fixed_point_qconfig(frac_weight, frac_act):
    weight_fq = FakeQuantize.with_args(observer=StaticFixedPointObserver,
        quant_min=-32768, quant_max=32767, dtype=torch.qint32,
        qscheme=torch.per_tensor_symmetric, frac_bits=frac_weight)
    act_fq = FakeQuantize.with_args(observer=StaticFixedPointObserver,
        quant_min=-32768, quant_max=32767, dtype=torch.qint32,
        qscheme=torch.per_tensor_symmetric, frac_bits=frac_act)
    return QConfig(activation=act_fq, weight=weight_fq)

# ==========================================
# 4. 模型架構
# ==========================================
def prune_shufflenet_conv5(model, target_channels=960):
    conv5_block = model.conv5
    conv = conv5_block[0]
    bn   = conv5_block[1]
    raw_weight = conv.weight.data
    l1_norms = raw_weight.view(raw_weight.shape[0], -1).abs().sum(dim=1)
    _, indices = torch.sort(l1_norms, descending=True)
    keep_indices = indices[:target_channels]
    new_conv = nn.Conv2d(conv.in_channels, target_channels, kernel_size=1, stride=1, padding=0, bias=False)
    new_bn = nn.BatchNorm2d(target_channels)
    new_conv.weight.data = conv.weight.data.index_select(0, keep_indices)
    new_bn.weight.data   = bn.weight.data.index_select(0, keep_indices)
    new_bn.bias.data     = bn.bias.data.index_select(0, keep_indices)
    new_bn.running_mean  = bn.running_mean.index_select(0, keep_indices)
    new_bn.running_var   = bn.running_var.index_select(0, keep_indices)
    model.conv5[0] = new_conv
    model.conv5[1] = new_bn
    model.fc = nn.Linear(target_channels, CONFIG['num_classes'])
    return model

def fuse_normalization_to_conv1(model, mean, std):
    print("🔨 Fusing Normalization (Mean/Std) into Conv1...")
    conv = model.conv1[0]
    mean_t = torch.tensor(mean).view(3, 1, 1).to(conv.weight.device)
    std_t  = torch.tensor(std).view(3, 1, 1).to(conv.weight.device)
    with torch.no_grad():
        conv.weight.data.div_(std_t)
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
        weight_sum = conv.weight.data.sum(dim=(2, 3))
        bias_adjustment = (weight_sum * mean_t.squeeze()).sum(dim=1)
        conv.bias.data.sub_(bias_adjustment)
    return model

class DistillQATModel(nn.Module):
    def __init__(self, original_model, is_teacher=False):
        super().__init__()
        self.is_teacher = is_teacher
        if not is_teacher:
            self.quant   = QuantStub()
            self.dequant = DeQuantStub()
        self.model = original_model

    def forward(self, x):
        if not self.is_teacher: x = self.quant(x)
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        x = x.mean([2, 3])
        logits = self.model.fc(x)
        if not self.is_teacher: logits = self.dequant(logits)
        return logits

class MultiLabelDistillLoss(nn.Module):
    """蒸餾損失 = Hard Loss (標籤) + alpha * Soft Loss (教師邏輯)"""
    def __init__(self, pos_weights, alpha=3.0):
        super().__init__()
        self.alpha     = alpha
        self.hard_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.soft_loss = nn.BCEWithLogitsLoss()

    def forward(self, s_logits, t_logits, labels):
        loss_hard  = self.hard_loss(s_logits, labels)
        t_probs    = torch.sigmoid(t_logits).detach()
        loss_soft  = self.soft_loss(s_logits, t_probs)
        # 蒸餾平衡公式
        total_loss = loss_hard + (self.alpha * loss_soft)
        return total_loss, loss_hard, loss_soft

# ==========================================
# 5. 主程式
# ==========================================
def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"🚀 Training on {device}")
    print(f"📉 Scheduler: OneCycleLR (max_lr={CONFIG['learning_rate']:.0e})")
    print(f"🎯 Alpha (Soft Loss weight): {CONFIG['alpha']}")

    # --- A. Data ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        HardwareSimulateTransform()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        HardwareSimulateTransform()
    ])
    train_dataset = MultiLabelDataset(f"{CONFIG['data_root']}/{CONFIG['train_csv']}", CONFIG['data_root'], transform=train_transform)
    val_dataset   = MultiLabelDataset(f"{CONFIG['data_root']}/{CONFIG['val_csv']}",   CONFIG['data_root'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'], shuffle=False,
                              num_workers=CONFIG['num_workers'])

    pos_weights = get_pos_weights(train_dataset).to(device)

    # --- B. Teacher ---
    print("Load Teacher (X2.0)...")
    teacher_base = models.shufflenet_v2_x2_0(weights=None)
    in_features  = teacher_base.fc.in_features
    teacher_base.fc = nn.Linear(in_features, CONFIG['num_classes'])

    if not os.path.exists(CONFIG['teacher_model_path']):
        print(f"❌ 錯誤：無法找到教師權重：{CONFIG['teacher_model_path']}")
        print(f"   請確認路徑，或使用 --teacher-model-path 指定。")
        return
    try:
        ckpt = torch.load(CONFIG['teacher_model_path'], map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        teacher_base.load_state_dict(state_dict, strict=True)
        print("✅ 教師模型權重載入成功！")
    except Exception as e:
        print(f"❌ 載入教師權重時發生錯誤：{e}")
        return

    teacher_base = fuse_normalization_to_conv1(teacher_base, CONFIG['norm_mean'], CONFIG['norm_std'])
    teacher = DistillQATModel(teacher_base, is_teacher=True).to(device)
    teacher.eval()

    # --- C. Student ---
    print("Init Student (X0.5)...")
    student_base = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    student_base = prune_shufflenet_conv5(student_base, target_channels=CONFIG['target_conv5_channels'])
    student_base = fuse_normalization_to_conv1(student_base, CONFIG['norm_mean'], CONFIG['norm_std'])
    student = DistillQATModel(student_base, is_teacher=False).to(device)

    # --- D. QAT Setup ---
    print("Configuring QAT (Q8.8)...")
    student.eval()
    student.qconfig = get_fixed_point_qconfig(CONFIG['q_frac_weight'], CONFIG['q_frac_act'])
    fuse_modules(student.model, [['conv1.0', 'conv1.1', 'conv1.2']], inplace=True)
    fuse_modules(student.model, [['conv5.0', 'conv5.1']], inplace=True)
    for name, module in student.model.named_modules():
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
    student.train()
    prepare_qat(student, inplace=True)
    student.to(device)

    # --- E. Optimizer & Scheduler ---
    optimizer = optim.AdamW(student.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)

    # 學生模型使用 OneCycleLR 更能快速捕捉教師邊界
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG['learning_rate'], 
        epochs=CONFIG['epochs'], steps_per_epoch=len(train_loader)
    )
    ema_model = AveragedModel(student, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    criterion = MultiLabelDistillLoss(pos_weights, CONFIG['alpha']).to(device)

    # --- F. Main Loop ---
    best_f1 = 0.0
    no_improve_epochs = 0

    use_ema = True
    for epoch in range(CONFIG['epochs']):
        student.train()
        losses   = AverageMeter()
        losses_h = AverageMeter()
        losses_s = AverageMeter()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] | LR: {current_lr:.2e}")

        # 蒸餾時關閉 Mixup，避免干擾軟標籤學習
        current_mixup = None 
        current_ema   = ema_model if use_ema else None

        # 最後 5 個 Epoch 關閉 QAT Observer（穩定量化參數）
        if epoch >= CONFIG['epochs'] - 5:
            try:
                import torch.ao.quantization as tq
                student.apply(tq.disable_observer)
            except Exception:
                pass
            for m in student.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                t_logits = teacher(inputs)

            s_inputs = F.interpolate(inputs, size=(128, 128), mode='bilinear', align_corners=False)
            s_logits = student(s_inputs)

            loss, l_hard, l_soft = criterion(s_logits, t_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() # 恢復為 per-batch 更新
            ema_model.update_parameters(student)

            losses.update(loss.item(), inputs.size(0))
            losses_h.update(l_hard.item(), inputs.size(0))
            losses_s.update(l_soft.item(), inputs.size(0))

            if i % (len(train_loader) // 3) == 0 and i > 0:
                print(f"  Step [{i}/{len(train_loader)}] Loss: {losses.val:.4f} "
                      f"(H: {losses_h.val:.4f}, S: {losses_s.val:.4f})")

        # ★ Scheduler 每個 Epoch 才更新一次
        scheduler.step()

        print(f"  Avg Loss: {losses.avg:.4f} (Hard:{losses_h.avg:.3f}, Soft:{losses_s.avg:.3f})")

        # --- G. Validation ---
        eval_model = copy.deepcopy(student)
        eval_model.load_state_dict(ema_model.module.state_dict())
        eval_model.eval()

        all_preds, all_targets = [], []
        val_losses = AverageMeter()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                s_inputs = F.interpolate(inputs, size=(128, 128), mode='bilinear', align_corners=False)
                logits   = eval_model(s_inputs)
                loss_val = criterion.hard_loss(logits, targets)
                val_losses.update(loss_val.item(), inputs.size(0))
                preds = (logits > 0).float().cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())

        all_preds   = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        exact_match      = np.mean(np.all(all_preds == all_targets, axis=1)) * 100
        f1_macro         = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)

        print(f"\n🧩 Val Summary (Epoch {epoch+1}):")
        print(f"   Loss: {val_losses.avg:.4f}")
        print(f"   Exact Match Acc: {exact_match:.2f}%")
        print(f"   Macro F1-Score: {f1_macro:.4f}")
        print("   ----------------------------")
        print("   Recall per Class:")
        for i, name in enumerate(CONFIG['class_names']):
            print(f"     - {name:<15}: {recall_per_class[i]*100:.2f}%")
        print("   ----------------------------")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            no_improve_epochs = 0
            print(f"⭐ New Best F1: {best_f1:.4f} -> Saving model...")
            torch.save(eval_model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))
        else:
            no_improve_epochs += 1

        # Early Stopping
        if CONFIG['patience'] > 0 and no_improve_epochs >= CONFIG['patience']:
            print(f"\n🛑 Early Stopping: {CONFIG['patience']} 個 Epoch 內無改善，訓練結束。")
            break

    print(f"\n✅ Training Completed. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
