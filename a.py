import torch

path = "/home/nvidia-lab/ai4life/CTPET/ctvit_llavamed/checkpoints/align_region_minh/checkpoint-2786/mm_projector.bin"
state_dict = torch.load(path, map_location="cpu")

def check_bad_tensor(tensor, std_th=1e-3, zero_th=0.9):
    std = tensor.std().item()
    near_zero_ratio = (tensor.abs() < 1e-4).float().mean().item()
    
    is_bad = False
    reasons = []
    
    if std < std_th:
        is_bad = True
        reasons.append(f"std nhỏ ({std:.2e})")
        
    if near_zero_ratio > zero_th:
        is_bad = True
        reasons.append(f"{near_zero_ratio*100:.1f}% ~ 0")
        
    return is_bad, std, near_zero_ratio, reasons


def print_bad(name, tensor, std, near_zero, reasons):
    print(f"🚨 {name}")
    print(f"  shape = {tuple(tensor.shape)}")
    print(f"  std   = {std:.2e}")
    print(f"  near_zero = {near_zero*100:.2f}%")
    print(f"  reason: {', '.join(reasons)}")
    print()


# ===== MAIN =====
total = 0
bad = 0

for k, v in state_dict.items():
    if torch.is_tensor(v):
        total += 1
        is_bad, std, near_zero, reasons = check_bad_tensor(v)
        
        if is_bad:
            bad += 1
            print_bad(k, v, std, near_zero, reasons)

print("="*60)
print(f"Tổng tensor: {total}")
print(f"🚨 Tensor nghi vấn: {bad}")
print("="*60)