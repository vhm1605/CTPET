import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

from .fusion_module import PETCTFusion
# =========================================================================
# 1. KHỞI TẠO VÀ LOAD CỨNG VISION TOWER 
# =========================================================================
def build_vision_tower():
    from .fusion_module import PETCTFusion
    
    cvit = PETCTFusion()
    checkpoint_path = '/data/backup_sde1/PET-CT-report/ckpt/petct_emb/ctvit.89000.pt'
    print("-> Loading Vision Tower from:", checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
    
    cvit.load_state_dict(checkpoint) 
    
    cvit = cvit.to(torch.bfloat16)
    for param in cvit.parameters():
        param.requires_grad = False

    cvit.eval() 
    cvit.hidden_size = 512
    cvit.is_loaded = True
    return cvit

# =========================================================================
# 2. HÀM XỬ LÝ ẢNH CHUẨN GỐC CỦA BẠN (Chỉ unsqueeze thêm chiều Batch ở cuối)
# =========================================================================
def process_image(image: np.ndarray, is_pet=True, fix_depth=140):
    """
    Process the image from D x H x W to C x H x W x D 
    - Resize the depth dimension to fix_depth using interpolation
    - Ensure fix_depth is divisible by 4 (pad if necessary)
    - Normalize pixel values by dividing by 32767
    - Convert image to (1, H, W, D) format
    """
    D, H, W = image.shape

    # Convert to torch tensor and normalize to [0, 1]
    if is_pet:
        image_tensor = torch.tensor(image, dtype=torch.float32) / 32767.0
    else:
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # Min-max normalization
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        image_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)

    # Reshape to (1, 1, D, H, W) for interpolation (N, C, D, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    # Resize depth dimension using trilinear interpolation (D → fix_depth)
    image_tensor = F.interpolate(image_tensor, size=(fix_depth, 480, 480), mode='trilinear', align_corners=False)

    # Remove batch & channel dimensions → (fix_depth, H, W) -> Gốc của bạn ra (1, D, H, W)
    image_tensor = image_tensor.squeeze(0)

    # Thêm chiều Batch (1, C, D, H, W) để nhét vừa khít vào vision_tower
    return image_tensor.unsqueeze(0)

# =========================================================================
# 3. SCRIPT DUYỆT THƯ MỤC VÀ TIẾN HÀNH CACHE TOÀN BỘ
# =========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vision_tower = build_vision_tower().to(device)
    
    base_path = "/data/backup_sde1/PET-CT-report"
    years = ["PETCT_2017", "PETCT_2018", "PETCT_2019", "PETCT_2023"]
    regions = ["abdomen_pelvis", "chest", "head_neck"]
    
    for year in years:
        year_path = os.path.join(base_path, year)
        if not os.path.exists(year_path):
            continue
            
        months = [m for m in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, m))]
        
        for month in months:
            month_path = os.path.join(year_path, month)
            
            # Tạo sẵn cấu trúc thư mục cache tương ứng
            for region in regions:
                cache_dir = os.path.join(month_path, "cache", region)
                os.makedirs(cache_dir, exist_ok=True)
                
            pet_images_base_path = os.path.join(month_path, "images")
            if not os.path.exists(pet_images_base_path):
                continue
                
            search_pattern = os.path.join(pet_images_base_path, "*", "*.npy")
            pet_file_paths = glob.glob(search_pattern)
            
            print(f"Processing Year: {year} | Month: {month} | Found {len(pet_file_paths)} files.")
            
            for pet_file_path in pet_file_paths:
                parts = pet_file_path.split(os.sep)
                region = parts[-2]
                file_name = parts[-1]
                
                if region not in regions:
                    continue
                    
                ct_file_path = pet_file_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}ref_images{os.sep}")
                
                # Tên file cache giống hệt tên file pet/ct, chỉ đổi đuôi sang .pt để lưu dict của PyTorch
                cache_file_name = file_name.replace(".npy", ".pt")
                output_cache_path = os.path.join(month_path, "cache", region, cache_file_name)
                
                if os.path.exists(output_cache_path):
                    continue
                    
                if not os.path.exists(ct_file_path):
                    print(f"[Warning] Missing CT file for: {pet_file_path}")
                    continue
                    
                try:
                    pet_data = np.load(pet_file_path)
                    ct_data = np.load(ct_file_path)
                    pet_tensor = process_image(pet_data, is_pet=True).to(device).to(torch.bfloat16)
                    ct_tensor = process_image(ct_data, is_pet=False).to(device).to(torch.bfloat16)
                    
                    with torch.no_grad():
                        global_feat = vision_tower(pet_tensor, ct_tensor)
                        pet_cache = vision_tower.pet_enc.build_video_cache(pet_tensor, L=2)
                        ct_cache = vision_tower.ct_enc.build_video_cache(ct_tensor, L=2)
                    
                    # Đóng gói và đẩy về CPU giải phóng VRAM liên tục
                    unified_cache = {
                        "global_feat": global_feat.cpu(),
                        "pet_cache": [layer.cpu() for layer in pet_cache],
                        "ct_cache": [layer.cpu() for layer in ct_cache]
                    }
                    
                    torch.save(unified_cache, output_cache_path)
                    
                except Exception as e:
                    print(f"[Error] Lỗi khi xử lý file {file_name}: {e}")

if __name__ == "__main__":
    main()