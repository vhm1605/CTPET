
# import os
# import re
# import json
# import math
# import numpy as np
# import scipy.ndimage as ndimage
# import torch  # Tích hợp PyTorch

# # ============================================================
# # FULL CONFIGURATION FOR 3 REGIONS
# # ============================================================
# # ORGAN_THRESHOLDS = {
# #   "head_neck": {
# #     "11": {"name": "esophagus", "suv_threshold": 1.5},
# #     "22": {"name": "brain", "suv_threshold": 0.0},
# #     "57": {"name": "trachea", "suv_threshold": 1.5},
# #     "120": {"name": "skull", "suv_threshold": 1.5},
# #     "121": {"name": "spinal cord", "suv_threshold": 1.5},
# #     "126": {"name": "thyroid gland", "suv_threshold": 2.0}
# #   },

# #   "chest": {
# #     "1": {"name": "liver", "suv_threshold": 2.5},
# #     "6": {"name": "aorta", "suv_threshold": 1.5},
# #     "11": {"name": "esophagus", "suv_threshold": 1.5},
# #     "12": {"name": "stomach", "suv_threshold": 2.5},
# #     "28": {"name": "left lung upper lobe", "suv_threshold": 1.0},
# #     "29": {"name": "left lung lower lobe", "suv_threshold": 1.0},
# #     "30": {"name": "right lung upper lobe", "suv_threshold": 1.0},
# #     "31": {"name": "right lung middle lobe", "suv_threshold": 1.0},
# #     "32": {"name": "right lung lower lobe", "suv_threshold": 1.0},
# #     "115": {"name": "heart", "suv_threshold": 0.0},
# #     "121": {"name": "spinal cord", "suv_threshold": 1.5},
# #     "122": {"name": "sternum", "suv_threshold": 1.5}
# #   },

# #   "abdomen_pelvis": {
# #     "1": {"name": "liver", "suv_threshold": 2.5},
# #     "3": {"name": "spleen", "suv_threshold": 2.0},
# #     "4": {"name": "pancreas", "suv_threshold": 2.0},
# #     "5": {"name": "right kidney", "suv_threshold": 0.0},
# #     "14": {"name": "left kidney", "suv_threshold": 0.0},
# #     "6": {"name": "aorta", "suv_threshold": 1.5},
# #     "8": {"name": "right adrenal gland", "suv_threshold": 1.8},
# #     "9": {"name": "left adrenal gland", "suv_threshold": 1.8},
# #     "10": {"name": "gallbladder", "suv_threshold": 2.0},
# #     "11": {"name": "esophagus", "suv_threshold": 1.5},
# #     "12": {"name": "stomach", "suv_threshold": 2.5},
# #     "15": {"name": "bladder", "suv_threshold": 0.0},
# #     "62": {"name": "colon", "suv_threshold": 2.5},
# #     "118": {"name": "prostate", "suv_threshold": 2.0}
# #   }
# # }

# ORGAN_THRESHOLDS = {
#     "head_neck": {
#         "11":  {"name": "esophagus",     "suv_threshold": 2.5},
#         "22":  {"name": "brain",         "suv_threshold": 2.5},
#         "57":  {"name": "trachea",       "suv_threshold": 2.5},
#         "120": {"name": "skull",         "suv_threshold": 2.5},
#         "121": {"name": "spinal cord",   "suv_threshold": 2.5},
#         "126": {"name": "thyroid gland", "suv_threshold": 2.0},
#     },

#     "chest": {
#         "1":   {"name": "liver",                 "suv_threshold": 2.5},
#         "6":   {"name": "aorta",                 "suv_threshold": 2.5},
#         "11":  {"name": "esophagus",             "suv_threshold": 2.5},
#         "12":  {"name": "stomach",               "suv_threshold": 2.5},
#         "28":  {"name": "left lung upper lobe",  "suv_threshold": 1.0},
#         "29":  {"name": "left lung lower lobe",  "suv_threshold": 1.0},
#         "30":  {"name": "right lung upper lobe", "suv_threshold": 1.0},
#         "31":  {"name": "right lung middle lobe","suv_threshold": 1.0},
#         "32":  {"name": "right lung lower lobe", "suv_threshold": 1.0},
#         "115": {"name": "heart",                 "suv_threshold": 2.5},
#         "121": {"name": "spinal cord",           "suv_threshold": 2.5},
#         "122": {"name": "sternum",               "suv_threshold": 2.5},
#     },

#     "abdomen_pelvis": {
#         "1":   {"name": "liver",               "suv_threshold": 2.5},
#         "3":   {"name": "spleen",              "suv_threshold": 2.5},
#         "4":   {"name": "pancreas",             "suv_threshold": 2.5},
#         "5":   {"name": "right kidney",         "suv_threshold": 2.5},
#         "14":  {"name": "left kidney",          "suv_threshold": 2.5},
#         "6":   {"name": "aorta",                "suv_threshold": 2.5},
#         "8":   {"name": "right adrenal gland",  "suv_threshold": 2.5},
#         "9":   {"name": "left adrenal gland",   "suv_threshold": 2.5},
#         "10":  {"name": "gallbladder",          "suv_threshold": 2.5},
#         "11":  {"name": "esophagus",             "suv_threshold": 2.5},
#         "12":  {"name": "stomach",               "suv_threshold": 2.5},
#         "15":  {"name": "bladder",               "suv_threshold": 2.5},
#         "62":  {"name": "colon",                 "suv_threshold": 2.5},
#         "118": {"name": "prostate",              "suv_threshold": 2.5},
#     },
# }
# # ============================================================
# # HELPER FUNCTIONS
# # ============================================================
# def dicom_time_to_seconds(t):
#     t = str(t).split(".")[0].zfill(6)
#     return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])

# def get_seg_path(pet_path):
#     path_norm = os.path.normpath(pet_path)
#     parts = path_norm.split(os.sep)
#     valid_organs = set(ORGAN_THRESHOLDS.keys())
    
#     for i in range(len(parts) - 1):
#         if parts[i] == "images" and parts[i+1] in valid_organs:
#             parts[i] = "ref_images"
#             parts[i+1] = f"{parts[i+1]}_seg"
#             break
#     return os.sep.join(parts)

# def load_start_index_from_json(pet_path, current_region):
#     pet_path = os.path.normpath(pet_path)
#     parts = pet_path.split(os.sep)
#     try:
#         images_idx = parts.index("images")
#     except ValueError:
#         return None
        
#     filename_npy = parts[-1]
#     filename_json = filename_npy.replace(".npy", ".json")
    
#     month_dir_parts = parts[:images_idx]
#     region_index_file_path = os.path.join(os.sep.join(month_dir_parts), "region_index", filename_json)
    
#     if not os.path.exists(region_index_file_path):
#         return None
        
#     with open(region_index_file_path, "r", encoding="utf-8") as f:
#         mapping_data = json.load(f)
        
#     region_info = mapping_data.get("regions", {}).get(current_region)
#     if not region_info or isinstance(region_info, str):
#         return None
        
#     return region_info.get("start_index")

# def validate_suv_parameters(filename, weight_kg, total_dose, suv_volume=None):
#     issues = []
    
#     if weight_kg <= 0.0:
#         issues.append(f"Cân nặng bằng không hoặc rỗng ({weight_kg} kg)")
#     elif weight_kg < 3.0 or weight_kg > 250.0:
#         issues.append(f"Cân nặng ngoài dải sinh lý thực tế ({weight_kg} kg)")

#     if total_dose <= 0.0:
#         issues.append(f"Tổng liều tiêm vô lý ({total_dose} Bq)")
#     elif total_dose < 100000000.0: 
#         issues.append(f"Tổng liều tiêm thấp bất thường ({total_dose} Bq)")
#     elif total_dose > 600000000.0:
#         issues.append(f"Tổng liều tiêm cao vượt ngưỡng an toàn ({total_dose} Bq)")

#     if suv_volume is not None:
#         max_suv_in_vol = float(np.max(suv_volume))
#         min_suv_in_vol = float(np.min(suv_volume))
#         if max_suv_in_vol > 60.0: 
#             issues.append(f"SUV_Max toàn bộ khối ảnh cao vượt ngưỡng tự nhiên ({max_suv_in_vol:.2f})")
#         elif max_suv_in_vol == 0.0:
#             issues.append("Toàn bộ ma trận SUV bằng 0")
#         if min_suv_in_vol < -5.0:
#             issues.append(f"Khối ảnh xuất hiện nhiễu âm nặng bất thường ({min_suv_in_vol:.2f})")

#     return issues

# def detect_region_from_path(pet_path):
#     """
#     Tự động trích xuất current_region từ đường dẫn file ảnh PET.
#     """
#     path_norm = os.path.normpath(pet_path)
#     parts = path_norm.split(os.sep)
#     valid_regions = set(ORGAN_THRESHOLDS.keys())
    
#     # Duyệt ngược từ tên file lên để tìm khớp thư mục vùng
#     for part in reversed(parts):
#         if part in valid_regions:
#             return part
#     for part in parts:
#         for region in valid_regions:
#             if region in part:
#                 return region
#     raise ValueError(f"Không nhận diện được phân vùng cơ thể (current_region) từ đường dẫn: {pet_path}")


# # ============================================================
# # FINAL SINGLE PATH TO TORCH SUV TENSOR PIPELINE
# # ============================================================
# import os
# import re
# import json
# import math
# import numpy as np
# import torch

# def pipeline_pet_to_suv_torch(pet_path, device="cpu"):
#     """
#     Input:
#         - pet_path (str): Đường dẫn đến 1 file dữ liệu PET dạng .npy
#         - device (str): Thiết bị lưu trữ Tensor đầu ra ("cpu" hoặc "cuda")
#     Output:
#         - suv_tensor (torch.Tensor): Tensor 3D [Z, Y, X] chứa giá trị SUV, giữ nguyên cấu trúc shape ảnh gốc.
#                                      (Trả về Tensor toàn số 0 nếu có lỗi).
#         - all_issues (list): Danh sách lỗi/cảnh báo lâm sàng (nếu có).
#         - current_region (str): Phân vùng cơ thể tự động nhận diện được.
#     """
#     pet_path = os.path.normpath(pet_path)
#     parts = pet_path.split(os.sep)
    
#     all_issues = []
#     current_region = "Unknown"
    
#     # Ưu tiên load dữ liệu ảnh trước để lấy shape.
#     # Nếu bản thân file ảnh lỗi không đọc được, trả về tensor 0 kích thước (1,1,1).
#     try:
#         pet_volume = np.load(pet_path).astype(np.float32)
#         vol_shape = pet_volume.shape
#     except Exception as e:
#         all_issues.append(f"Lỗi đọc file NPY ban đầu: {e}")
#         return torch.zeros((1, 1, 1), dtype=torch.float32, device=device), all_issues, current_region

#     try:
#         # 1. Tự động suy ra vùng cơ thể và start_slice_idx tương ứng
#         # (Nếu hàm detect_region_from_path lỗi, nó cũng sẽ bay thẳng xuống except bên dưới)
#         current_region = detect_region_from_path(pet_path)
#         start_slice_idx = load_start_index_from_json(pet_path, current_region)
#         if start_slice_idx is None:
#             raise ValueError(f"Không lấy được start_index từ mapping JSON cho vùng {current_region}")

#         try:
#             petct_idx = next(i for i, p in enumerate(parts) if p.startswith("PETCT_"))
#         except StopIteration:
#             raise ValueError("Không tìm thấy thư mục gốc PETCT_ từ đường dẫn ảnh.")

#         year_folder = parts[petct_idx]      
#         month_folder = parts[petct_idx + 1] 
#         filename = os.path.basename(pet_path)

#         match = re.match(r"day_(\d+)_patient_(\d+)\.npy", filename)
#         if not match:
#             raise ValueError(f"Định dạng tên file không khớp mẫu regex day_XX_patient_XX: {filename}")

#         day_str_raw = match.group(1)       
#         day_str_int = str(int(day_str_raw)) 
#         patient = match.group(2)       

#         base_meta_dir = os.path.join(
#             "/data/backup_sde1/PET_metadata_full", year_folder, month_folder, day_str_int, f"patient_{patient}"
#         )
#         if not os.path.exists(base_meta_dir):
#             base_meta_dir = os.path.join(
#                 "/data/backup_sde1/PET_metadata_full", year_folder, month_folder, day_str_raw, f"patient_{patient}"
#             )

#         patient_json_path = os.path.join(base_meta_dir, "patient_metadata.json")
#         if not os.path.exists(patient_json_path):
#             raise ValueError(f"Thiếu file patient_metadata.json tại đường dẫn: {base_meta_dir}")
            
#         with open(patient_json_path, "r", encoding="utf-8") as f:
#             patient_data = json.load(f)
            
#         weight_kg_val = patient_data.get("patient_weight_kg", patient_data.get("weight_kg"))
#         if weight_kg_val is None:
#             raise ValueError("Thiếu trường dữ liệu cân nặng bệnh nhân trong JSON.")
            
#         weight_kg = float(weight_kg_val)
#         if weight_kg <= 0.0:
#             raise ValueError(f"Cân nặng bệnh nhân không hợp lệ: {weight_kg} kg")
#         weight_g = weight_kg * 1000.0

#         num_slices = vol_shape[0]

#         metadata_jsonl_path = os.path.join(base_meta_dir, "pet_metadata.jsonl")
#         if not os.path.exists(metadata_jsonl_path):
#             raise ValueError(f"Thiếu file dữ liệu lát cắt pet_metadata.jsonl tại đường dẫn: {base_meta_dir}")

#         slice_metadata_map = {}
#         with open(metadata_jsonl_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 if not line.strip(): continue
#                 meta = json.loads(line)
#                 if "slice_idx" in meta:
#                     slice_metadata_map[int(meta["slice_idx"])] = meta

#         suv_volume = np.zeros_like(pet_volume)
#         valid_slice_count = 0
#         runtime_issues = []
#         all_injected_doses = []

#         # XỬ LÝ LÁT CẮT & TÍNH PHÂN RÃ THEO THỜI GIAN
#         for idx in range(num_slices):
#             actual_wb_idx = start_slice_idx + idx
#             if actual_wb_idx not in slice_metadata_map:
#                 continue
                
#             metadata = slice_metadata_map[actual_wb_idx]
#             if metadata.get("units", "") != "BQML":
#                 raise ValueError(f"Đơn vị máy quét lỗi (Units = '{metadata.get('units', '')}'). Yêu cầu phải là BQML.")

#             required_fields = ["total_dose", "half_life", "injection_time"]
#             for field in required_fields:
#                 if field not in metadata:
#                     raise ValueError(f"Lát cắt số {actual_wb_idx} bị thiếu trường dữ liệu cốt lõi: {field}")

#             injected_dose = float(metadata["total_dose"])
#             remaining_dose = metadata.get("remaining_dose", 0.0)
#             if remaining_dose:
#                 injected_dose -= float(remaining_dose)
                
#             all_injected_doses.append(injected_dose)
#             half_life = float(metadata["half_life"])
#             decay_correction = metadata.get("decay_correction", "NONE")
            
#             injection_seconds = dicom_time_to_seconds(metadata["injection_time"])
#             scan_time_str = metadata.get("acquisition_time", metadata.get("series_time"))
#             if not scan_time_str:
#                 raise ValueError(f"Lát cắt số {actual_wb_idx} thiếu cả mốc acquisition_time và series_time")
#             scan_seconds = dicom_time_to_seconds(scan_time_str)
            
#             base_decay_seconds = scan_seconds - injection_seconds
#             if base_decay_seconds < 0:
#                 base_decay_seconds += 86400

#             frame_ref_time = float(metadata.get("frame_reference_time", 0.0))
#             decay_seconds = base_decay_seconds + (frame_ref_time / 1000.0) if frame_ref_time > 0.0 else base_decay_seconds
            
#             if half_life <= 0:
#                 raise ValueError(f"Thông số chu kỳ bán rã (half_life) lỗi: {half_life}")

#             if idx in [0, num_slices // 2, num_slices - 1]:
#                 if decay_seconds <= 0:
#                     runtime_issues.append(f"Mốc [Lát {actual_wb_idx}]: Thời gian quét lỗi ({decay_seconds}s)")
#                 elif decay_seconds < 2400: 
#                     runtime_issues.append(f"Mốc [Lát {actual_wb_idx}]: Thời gian chờ ngắn ({decay_seconds // 60} phút)")
#                 elif decay_seconds > 7200: 
#                     runtime_issues.append(f"Mốc [Lát {actual_wb_idx}]: Thời gian chờ dài ({decay_seconds // 3600} tiếng)")

#             decay_factor = math.exp(-math.log(2) * decay_seconds / half_life)

#             if decay_correction in ["START", "NONE"]:
#                 corrected_dose = injected_dose * decay_factor
#                 if corrected_dose <= 0:
#                     raise ValueError("Liều phóng xạ sau phân rã nhỏ hơn hoặc bằng 0.")
#                 suv_factor = weight_g / corrected_dose
#             elif decay_correction == "ADMIN":
#                 if injected_dose <= 0:
#                     raise ValueError("Liều tiêm ban đầu nhỏ hơn hoặc bằng 0.")
#                 suv_factor = weight_g / injected_dose

#             rescale_slope = float(metadata.get("rescale_slope", 1.0))
#             rescale_intercept = float(metadata.get("rescale_intercept", 0.0))

#             bqml_slice = (pet_volume[idx] * rescale_slope) + rescale_intercept
#             suv_volume[idx] = bqml_slice * suv_factor
#             valid_slice_count += 1

#         if valid_slice_count != num_slices:
#             raise ValueError(f"Số lượng lát cắt xử lý được ({valid_slice_count}) không khớp với tổng số lát cắt của ảnh ({num_slices}) của {pet_path}")

#         avg_total_dose = float(np.mean(all_injected_doses)) if all_injected_doses else 0.0
#         metadata_issues = validate_suv_parameters(filename, weight_kg, avg_total_dose, suv_volume)
#         all_issues.extend(metadata_issues + runtime_issues)

#         # 2. Thành công -> Đẩy ma trận SUV bình thường lên thiết bị
#         suv_tensor = torch.from_numpy(suv_volume).to(device)
#         return suv_tensor, all_issues, current_region

#     except Exception as e:
#         # 3. Gặp bất kỳ lỗi nào -> Bắt lỗi, ghi log và trả về Tensor toàn 0
#         all_issues.append(f"Xử lý thất bại (trả về Tensor 0): {str(e)}")
#         zero_tensor = torch.zeros(vol_shape, dtype=torch.float32, device=device)
#         return zero_tensor, all_issues, current_region

# # ============================================================
# # RUNTIME EXAMPLE - BATCH PROCESSING WITH REAL-TIME DIAGNOSTICS
# # ============================================================
# if __name__ == "__main__":
#     import os
#     import json
#     import glob
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     from tqdm import tqdm
#     from collections import Counter, defaultdict

#     # 1. Cấu hình các đường dẫn gốc và năm cần quét
#     base_root = "/data/backup_sde1/PET-CT-report"
#     target_years = ["PETCT_2017", "PETCT_2018", "PETCT_2019", "PETCT_2023"]
    
#     device_target = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🚀 Khởi tạo Pipeline. Device mục tiêu: {device_target.upper()}")

#     # 2. Thu thập tất cả các file ảnh .npy (Loại bỏ các ca whole_body)
#     all_pet_files = []
#     for year in target_years:
#         search_pattern = os.path.join(base_root, year, "**", "images", "**", "*.npy")
#         files = glob.glob(search_pattern, recursive=True)
#         filtered_files = [f for f in files if "whole_body" not in f.lower()]
#         all_pet_files.extend(filtered_files)

#     total_files = len(all_pet_files)
#     print(f"📂 Tìm thấy tổng cộng {total_files} file ảnh PET (.npy) cần xử lý.")

#     # 3. Khởi tạo các biến thống kê
#     success_count = 0
#     failure_count = 0
#     all_reports = {}
    
#     region_distribution = defaultdict(Counter)
#     processed_counter = 0

#     print("\n🎬 Bắt đầu tiến trình xử lý hàng loạt và phân tích vùng...")
#     for pet_path in tqdm(all_pet_files, desc="Processing PET files"):
#         filename = os.path.basename(pet_path)
#         high_suv_organs_in_sample = 0
        
#         try:
#             # Chạy pipeline trích xuất SUV tensor và nhận diện vùng
#             suv_tensor, issues, region = pipeline_pet_to_suv_torch(pet_path, device=device_target)
            
#             # 🚨 KIỂM TRA VÙNG HỢP LỆ
#             if region not in ORGAN_THRESHOLDS:
#                 tqdm.write(f"⚠️ [SAI PATH/VÙNG] Không khớp cấu hình vùng ('{region}') tại file: {pet_path}")
#                 if issues:
#                     tqdm.write(f"   └─> Cảnh báo từ pipeline: {issues}")
                
#                 all_reports[pet_path] = {
#                     "status": "SKIPPED_BAD_REGION",
#                     "region": region,
#                     "clinical_issues": issues
#                 }
#                 failure_count += 1
#                 processed_counter += 1
#                 continue

#             # Tìm file phân vùng tương ứng
#             seg_path = get_seg_path(pet_path)
#             if os.path.exists(seg_path):
#                 seg_volume = np.load(seg_path)
#                 seg_tensor = torch.from_numpy(seg_volume).to(device_target)
                
#                 # -----------------------------------------------------------------
#                 # 🔄 TIỀN XỬ LÝ & RESIZE ĐỒNG BỘ (SHAPE MỤC TIÊU: 140, 480, 480)
#                 # -----------------------------------------------------------------
                
#                 # A. Khớp kích thước khối SEG (Dùng nội suy 'nearest' để bảo toàn nhãn nguyên gốc)
#                 seg_tensor = seg_tensor.float().unsqueeze(0).unsqueeze(0)  # Thêm Batch & Channel thành 5D
#                 seg_tensor = F.interpolate(seg_tensor, size=(140, 480, 480), mode='nearest')
#                 seg_tensor = seg_tensor.squeeze(0).squeeze(0).long()       # Trả về 3D định dạng Long
                
#                 # B. Khớp kích thước khối PET SUV (Dùng nội suy 'trilinear' cho giá trị liên tục)
#                 suv_tensor = suv_tensor.unsqueeze(0).unsqueeze(0)          # Thêm Batch & Channel thành 5D
#                 suv_tensor = F.interpolate(suv_tensor, size=(140, 480, 480), mode='trilinear', align_corners=False)
#                 suv_tensor = suv_tensor.squeeze(0).squeeze(0).float()      # Trả về 3D định dạng Float
                
#                 # Double-check tính an toàn ma trận trước khi tính toán roi
#                 if suv_tensor.shape != seg_tensor.shape:
#                     raise ValueError(f"Lệch cấu trúc khối ảnh sau nội suy! SUV: {list(suv_tensor.shape)} != SEG: {list(seg_tensor.shape)}")

#                 # -----------------------------------------------------------------
#                 # ĐẾM SỐ LƯỢNG BỘ PHẬN VƯỢT NGƯỠNG SUV
#                 # -----------------------------------------------------------------
#                 config = ORGAN_THRESHOLDS[region]
#                 unique_labels = torch.unique(seg_tensor)
#                 unique_labels = unique_labels[unique_labels != 0]
                
#                 for label_id in unique_labels:
#                     label_str = str(int(label_id.item()))
#                     if label_str in config:
#                         threshold = config[label_str]["suv_threshold"]
                        
#                         organ_mask = (seg_tensor == label_id)
#                         organ_suvs = suv_tensor[organ_mask]
                        
#                         if organ_suvs.numel() > 0:
#                             suv_95th = torch.quantile(organ_suvs, 0.95).item()
#                             if suv_95th > threshold:
#                                 high_suv_organs_in_sample += 1
                                
#                 # Cập nhật thống kê phân phối vùng sau khi chạy thành công ca
#                 region_distribution[region][high_suv_organs_in_sample] += 1
                
#                 all_reports[pet_path] = {
#                     "status": "SUCCESS",
#                     "region": region,
#                     "shape": list(suv_tensor.shape),
#                     "device": str(suv_tensor.device),
#                     "high_suv_organs_count": high_suv_organs_in_sample,
#                     "clinical_issues": issues
#                 }
#                 success_count += 1
                
#             else:
#                 issues.append(f"Không tìm thấy file segmentation tương ứng tại: {seg_path}")
#                 all_reports[pet_path] = {
#                     "status": "MISSING_SEGMENTATION",
#                     "clinical_issues": issues
#                 }
#                 failure_count += 1

#         except Exception as e:
#             tqdm.write(f"❌ [CRASH BIẾN ĐỔI] Lỗi tại file [{filename}]: {str(e)}")
#             all_reports[pet_path] = {
#                 "status": "FAILED",
#                 "error_details": str(e)
#             }
#             failure_count += 1
        
#         processed_counter += 1
        
#         # --- THỐNG KÊ ĐỊNH KỲ MỖI 10 SAMPLE ---
#         if processed_counter % 10 == 0:
#             stats_output = []
#             stats_output.append(f"\n" + "="*60)
#             stats_output.append(f"📊 THỐNG KÊ PHÂN PHỐI ĐỊNH KỲ TẠI SAMPLE SỐ {processed_counter}")
#             stats_output.append("="*60)
            
#             for rgn in ORGAN_THRESHOLDS.keys():
#                 stats_output.append(f"📍 Vùng: {rgn.upper()}")
#                 if rgn in region_distribution and len(region_distribution[rgn]) > 0:
#                     for num_organs in sorted(region_distribution[rgn].keys()):
#                         count_images = region_distribution[rgn][num_organs]
#                         stats_output.append(f"  - Số ảnh có {num_organs} organ SUV cao: {count_images} ảnh")
#                 else:
#                     stats_output.append(f"  - (Chưa có dữ liệu ảnh được xử lý thành công)")
                    
#             stats_output.append("="*60 + "\n")
#             tqdm.write("\n".join(stats_output))

#     # 5. Xuất báo cáo tổng quan cuối cùng
#     print("\n" + "📊" * 15 + " THỐNG KÊ TỔNG KẾT TIẾN TRÌNH " + "📊" * 15)
#     print(f"• Tổng số file quét được: {total_files}")
#     print(f"• Xử lý THÀNH CÔNG:       {success_count} / {total_files} file")
#     print(f"• Xử lý THẤT BẠI/BỎ QUA:  {failure_count} / {total_files} file")

#     log_output_path = "./batch_processing_report.json"
#     with open(log_output_path, "w", encoding="utf-8") as f:
#         json.dump(all_reports, f, ensure_ascii=False, indent=4)
#     print(f"📝 Đã xuất báo cáo chi tiết từng file tại: {log_output_path}")





import os

import re

import json

import math

import numpy as np

import scipy.ndimage as ndimage

import torch  # Tích hợp PyTorch



# ============================================================

# FULL CONFIGURATION FOR 3 REGIONS

# ============================================================

# ORGAN_THRESHOLDS = {

#   "head_neck": {

#     "11": {"name": "esophagus", "suv_threshold": 1.5},

#     "22": {"name": "brain", "suv_threshold": 0.0},

#     "57": {"name": "trachea", "suv_threshold": 1.5},

#     "120": {"name": "skull", "suv_threshold": 1.5},

#     "121": {"name": "spinal cord", "suv_threshold": 1.5},

#     "126": {"name": "thyroid gland", "suv_threshold": 2.0}

#   },



#   "chest": {

#     "1": {"name": "liver", "suv_threshold": 2.5},

#     "6": {"name": "aorta", "suv_threshold": 1.5},

#     "11": {"name": "esophagus", "suv_threshold": 1.5},

#     "12": {"name": "stomach", "suv_threshold": 2.5},

#     "28": {"name": "left lung upper lobe", "suv_threshold": 1.0},

#     "29": {"name": "left lung lower lobe", "suv_threshold": 1.0},

#     "30": {"name": "right lung upper lobe", "suv_threshold": 1.0},

#     "31": {"name": "right lung middle lobe", "suv_threshold": 1.0},

#     "32": {"name": "right lung lower lobe", "suv_threshold": 1.0},

#     "115": {"name": "heart", "suv_threshold": 0.0},

#     "121": {"name": "spinal cord", "suv_threshold": 1.5},

#     "122": {"name": "sternum", "suv_threshold": 1.5}

#   },



#   "abdomen_pelvis": {

#     "1": {"name": "liver", "suv_threshold": 2.5},

#     "3": {"name": "spleen", "suv_threshold": 2.0},

#     "4": {"name": "pancreas", "suv_threshold": 2.0},

#     "5": {"name": "right kidney", "suv_threshold": 0.0},

#     "14": {"name": "left kidney", "suv_threshold": 0.0},

#     "6": {"name": "aorta", "suv_threshold": 1.5},

#     "8": {"name": "right adrenal gland", "suv_threshold": 1.8},

#     "9": {"name": "left adrenal gland", "suv_threshold": 1.8},

#     "10": {"name": "gallbladder", "suv_threshold": 2.0},

#     "11": {"name": "esophagus", "suv_threshold": 1.5},

#     "12": {"name": "stomach", "suv_threshold": 2.5},

#     "15": {"name": "bladder", "suv_threshold": 0.0},

#     "62": {"name": "colon", "suv_threshold": 2.5},

#     "118": {"name": "prostate", "suv_threshold": 2.0}

#   }

# }



ORGAN_THRESHOLDS = {

    "head_neck": {

        "11":  {"name": "esophagus",     "suv_threshold": 2.5},

        "22":  {"name": "brain",         "suv_threshold": 2.5},

        "57":  {"name": "trachea",       "suv_threshold": 2.5},

        "120": {"name": "skull",         "suv_threshold": 2.5},

        "121": {"name": "spinal cord",   "suv_threshold": 2.5},

        "126": {"name": "thyroid gland", "suv_threshold": 2.0},

    },



    "chest": {

        "1":   {"name": "liver",                 "suv_threshold": 2.5},

        "6":   {"name": "aorta",                 "suv_threshold": 2.5},

        "11":  {"name": "esophagus",             "suv_threshold": 2.5},

        "12":  {"name": "stomach",               "suv_threshold": 2.5},

        "28":  {"name": "left lung upper lobe",  "suv_threshold": 1.0},

        "29":  {"name": "left lung lower lobe",  "suv_threshold": 1.0},

        "30":  {"name": "right lung upper lobe", "suv_threshold": 1.0},

        "31":  {"name": "right lung middle lobe","suv_threshold": 1.0},

        "32":  {"name": "right lung lower lobe", "suv_threshold": 1.0},

        "115": {"name": "heart",                 "suv_threshold": 2.5},

        "121": {"name": "spinal cord",           "suv_threshold": 2.5},

        "122": {"name": "sternum",               "suv_threshold": 2.5},

    },



    "abdomen_pelvis": {

        "1":   {"name": "liver",               "suv_threshold": 2.5},

        "3":   {"name": "spleen",              "suv_threshold": 2.5},

        "4":   {"name": "pancreas",             "suv_threshold": 2.5},

        "5":   {"name": "right kidney",         "suv_threshold": 2.5},

        "14":  {"name": "left kidney",          "suv_threshold": 2.5},

        "6":   {"name": "aorta",                "suv_threshold": 2.5},

        "8":   {"name": "right adrenal gland",  "suv_threshold": 2.5},

        "9":   {"name": "left adrenal gland",   "suv_threshold": 2.5},

        "10":  {"name": "gallbladder",          "suv_threshold": 2.5},

        "11":  {"name": "esophagus",             "suv_threshold": 2.5},

        "12":  {"name": "stomach",               "suv_threshold": 2.5},

        "15":  {"name": "bladder",               "suv_threshold": 2.5},

        "62":  {"name": "colon",                 "suv_threshold": 2.5},

        "118": {"name": "prostate",              "suv_threshold": 2.5},

    },

}

# ============================================================

# HELPER FUNCTIONS

# ============================================================

def dicom_time_to_seconds(t):

    t = str(t).split(".")[0].zfill(6)

    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])



def get_seg_path(pet_path):

    path_norm = os.path.normpath(pet_path)

    parts = path_norm.split(os.sep)

    valid_organs = set(ORGAN_THRESHOLDS.keys())



    for i in range(len(parts) - 1):

        if parts[i] == "images" and parts[i+1] in valid_organs:

            parts[i] = "ref_images"

            parts[i+1] = f"{parts[i+1]}_seg"

            break

    return os.sep.join(parts)



def load_start_index_from_json(pet_path, current_region):

    pet_path = os.path.normpath(pet_path)

    parts = pet_path.split(os.sep)

    try:

        images_idx = parts.index("images")

    except ValueError:

        return None



    filename_npy = parts[-1]

    filename_json = filename_npy.replace(".npy", ".json")



    month_dir_parts = parts[:images_idx]

    region_index_file_path = os.path.join(os.sep.join(month_dir_parts), "region_index", filename_json)



    if not os.path.exists(region_index_file_path):

        return None



    with open(region_index_file_path, "r", encoding="utf-8") as f:

        mapping_data = json.load(f)



    region_info = mapping_data.get("regions", {}).get(current_region)

    if not region_info or isinstance(region_info, str):

        return None



    return region_info.get("start_index")



def validate_suv_parameters(filename, weight_kg, total_dose, suv_volume=None):

    issues = []



    if weight_kg <= 0.0:

        issues.append(f"Cân nặng bằng không hoặc rỗng ({weight_kg} kg)")

    elif weight_kg < 3.0 or weight_kg > 250.0:

        issues.append(f"Cân nặng ngoài dải sinh lý thực tế ({weight_kg} kg)")



    if total_dose <= 0.0:

        issues.append(f"Tổng liều tiêm vô lý ({total_dose} Bq)")

    elif total_dose < 100000000.0: 

        issues.append(f"Tổng liều tiêm thấp bất thường ({total_dose} Bq)")

    elif total_dose > 600000000.0:

        issues.append(f"Tổng liều tiêm cao vượt ngưỡng an toàn ({total_dose} Bq)")



    if suv_volume is not None:

        max_suv_in_vol = float(np.max(suv_volume))

        min_suv_in_vol = float(np.min(suv_volume))

        if max_suv_in_vol > 60.0: 

            issues.append(f"SUV_Max toàn bộ khối ảnh cao vượt ngưỡng tự nhiên ({max_suv_in_vol:.2f})")

        elif max_suv_in_vol == 0.0:

            issues.append("Toàn bộ ma trận SUV bằng 0")

        if min_suv_in_vol < -5.0:

            issues.append(f"Khối ảnh xuất hiện nhiễu âm nặng bất thường ({min_suv_in_vol:.2f})")



    return issues



def detect_region_from_path(pet_path):

    """

    Tự động trích xuất current_region từ đường dẫn file ảnh PET.

    """

    path_norm = os.path.normpath(pet_path)

    parts = path_norm.split(os.sep)

    valid_regions = set(ORGAN_THRESHOLDS.keys())



    # Duyệt ngược từ tên file lên để tìm khớp thư mục vùng

    for part in reversed(parts):

        if part in valid_regions:

            return part

    for part in parts:

        for region in valid_regions:

            if region in part:

                return region

    raise ValueError(f"Không nhận diện được phân vùng cơ thể (current_region) từ đường dẫn: {pet_path}")





# ============================================================

# FINAL SINGLE PATH TO TORCH SUV TENSOR PIPELINE

# ============================================================

import os

import re

import json

import math

import numpy as np

import torch



def pipeline_pet_to_suv_torch(pet_path, device="cpu"):

    """

    Input:

        - pet_path (str): Đường dẫn đến 1 file dữ liệu PET dạng .npy

        - device (str): Thiết bị lưu trữ Tensor đầu ra ("cpu" hoặc "cuda")

    Output:

        - suv_tensor (torch.Tensor): Tensor 3D [Z, Y, X] chứa giá trị SUV, giữ nguyên cấu trúc shape ảnh gốc.

                                     (Trả về Tensor toàn số 0 nếu có lỗi).

        - all_issues (list): Danh sách lỗi/cảnh báo lâm sàng (nếu có).

        - current_region (str): Phân vùng cơ thể tự động nhận diện được.

    """

    pet_path = os.path.normpath(pet_path)

    parts = pet_path.split(os.sep)



    all_issues = []

    current_region = "Unknown"



    # Ưu tiên load dữ liệu ảnh trước để lấy shape.

    # Nếu bản thân file ảnh lỗi không đọc được, trả về tensor 0 kích thước (1,1,1).

    try:

        pet_volume = np.load(pet_path).astype(np.float32)

        vol_shape = pet_volume.shape

    except Exception as e:

        all_issues.append(f"Lỗi đọc file NPY ban đầu: {e}")

        return torch.zeros((1, 1, 1), dtype=torch.float32, device=device), all_issues, current_region



    try:

        # 1. Tự động suy ra vùng cơ thể và start_slice_idx tương ứng

        # (Nếu hàm detect_region_from_path lỗi, nó cũng sẽ bay thẳng xuống except bên dưới)

        current_region = detect_region_from_path(pet_path)

        start_slice_idx = load_start_index_from_json(pet_path, current_region)

        if start_slice_idx is None:

            raise ValueError(f"Không lấy được start_index từ mapping JSON cho vùng {current_region}")



        try:

            petct_idx = next(i for i, p in enumerate(parts) if p.startswith("PETCT_"))

        except StopIteration:

            raise ValueError("Không tìm thấy thư mục gốc PETCT_ từ đường dẫn ảnh.")



        year_folder = parts[petct_idx]      

        month_folder = parts[petct_idx + 1] 

        filename = os.path.basename(pet_path)



        match = re.match(r"day_(\d+)_patient_(\d+)\.npy$", filename)

        if not match:

            raise ValueError(f"Định dạng tên file không khớp mẫu regex day_XX_patient_XX: {filename}")



        day_str_raw = match.group(1)       

        day_str_int = str(int(day_str_raw)) 

        patient = match.group(2)       



        base_meta_dir = os.path.join(

            "/data/backup_sde1/PET_metadata_full", year_folder, month_folder, day_str_int, f"patient_{patient}"

        )

        if not os.path.exists(base_meta_dir):

            base_meta_dir = os.path.join(

                "/data/backup_sde1/PET_metadata_full", year_folder, month_folder, day_str_raw, f"patient_{patient}"

            )



        patient_json_path = os.path.join(base_meta_dir, "patient_metadata.json")

        if not os.path.exists(patient_json_path):

            raise ValueError(f"Thiếu file patient_metadata.json tại đường dẫn: {base_meta_dir}")



        with open(patient_json_path, "r", encoding="utf-8") as f:

            patient_data = json.load(f)



        weight_kg_val = patient_data.get("patient_weight_kg", patient_data.get("weight_kg"))

        if weight_kg_val is None:

            raise ValueError("Thiếu trường dữ liệu cân nặng bệnh nhân trong JSON.")



        weight_kg = float(weight_kg_val)

        if weight_kg <= 0.0:

            raise ValueError(f"Cân nặng bệnh nhân không hợp lệ: {weight_kg} kg")

        weight_g = weight_kg * 1000.0



        num_slices = vol_shape[0]



        metadata_jsonl_path = os.path.join(base_meta_dir, "pet_metadata.jsonl")

        if not os.path.exists(metadata_jsonl_path):

            raise ValueError(f"Thiếu file dữ liệu lát cắt pet_metadata.jsonl tại đường dẫn: {base_meta_dir}")



        slice_metadata_map = {}

        with open(metadata_jsonl_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip(): continue

                meta = json.loads(line)

                if "slice_idx" in meta:

                    slice_metadata_map[int(meta["slice_idx"])] = meta



        suv_volume = np.zeros_like(pet_volume)

        valid_slice_count = 0

        runtime_issues = []

        all_injected_doses = []



        # XỬ LÝ LÁT CẮT & TÍNH PHÂN RÃ THEO THỜI GIAN

        for idx in range(num_slices):

            actual_wb_idx = start_slice_idx + idx

            if actual_wb_idx not in slice_metadata_map:

                continue



            metadata = slice_metadata_map[actual_wb_idx]

            if str(metadata.get("units", "")).upper() != "BQML":

                raise ValueError(
                    f"Đơn vị máy quét lỗi (Units = '{metadata.get('units', '')}'). "
                    f"Yêu cầu phải là BQML."
                )

            corrected_image = metadata.get("corrected_image")
            if corrected_image is not None:
                if isinstance(corrected_image, str):
                    corrected_tokens = {x.strip().upper() for x in corrected_image.split("\\")}
                else:
                    corrected_tokens = {str(x).upper() for x in corrected_image}

                if "DECY" not in corrected_tokens:
                    raise ValueError(
                        f"Lát {actual_wb_idx}: CorrectedImage không chứa DECY "
                        f"nhưng pipeline đang tính SUV từ ảnh decay-corrected."
                    )



            required_fields = ["total_dose", "half_life", "injection_time"]

            for field in required_fields:

                if field not in metadata:

                    raise ValueError(f"Lát cắt số {actual_wb_idx} bị thiếu trường dữ liệu cốt lõi: {field}")



            # DICOM RadionuclideTotalDose is already the administered activity
            # at RadiopharmaceuticalStartTime. Do NOT subtract residual activity
            # again unless the source field is explicitly known to be a pre-injection
            # syringe assay rather than the standard DICOM administered dose.
            injected_dose = float(metadata["total_dose"])

            all_injected_doses.append(injected_dose)

            half_life = float(metadata["half_life"])

            decay_correction = str(metadata.get("decay_correction", "NONE")).upper()

            injection_seconds = dicom_time_to_seconds(metadata["injection_time"])

            if half_life <= 0:
                raise ValueError(
                    f"Thông số chu kỳ bán rã (half_life) lỗi: {half_life}"
                )

            if decay_correction == "START":
                # -------------------------------------------------------------
                # IMPORTANT FOR THIS GE DISCOVERY STE DATASET
                # -------------------------------------------------------------
                # The 313-slice DICOM series has:
                #   Units = BQML
                #   DecayCorrection = START
                #   CorrectedImage contains DECY
                #
                # In this dataset, the stored DecayFactor values are internally
                # consistent with decay correction to SeriesTime, not with the
                # constant AcquisitionTime written in every slice.
                #
                # Therefore, for SUVbw normalization, decay the injected activity
                # from injection time to SeriesTime:
                #
                #   D_ref = D0 * exp(-ln(2) * (SeriesTime-InjectionTime) / half_life)
                #
                # Do NOT add FrameReferenceTime here. FrameReferenceTime belongs to
                # the individual bed/frame timing and has already been handled by
                # the scanner's decay correction of the image values.
                # -------------------------------------------------------------

                series_time_str = metadata.get("series_time")
                if not series_time_str:
                    raise ValueError(
                        f"Lát cắt số {actual_wb_idx} thiếu series_time "
                        f"cho DecayCorrection=START"
                    )

                reference_seconds = dicom_time_to_seconds(series_time_str)

                decay_seconds = reference_seconds - injection_seconds
                if decay_seconds < 0:
                    decay_seconds += 86400

                if idx in [0, num_slices // 2, num_slices - 1]:
                    if decay_seconds <= 0:
                        runtime_issues.append(
                            f"Mốc [Lát {actual_wb_idx}]: "
                            f"Thời gian từ tiêm đến SeriesTime lỗi ({decay_seconds}s)"
                        )
                    elif decay_seconds < 2400:
                        runtime_issues.append(
                            f"Mốc [Lát {actual_wb_idx}]: "
                            f"Thời gian chờ ngắn ({decay_seconds // 60} phút)"
                        )
                    elif decay_seconds > 7200:
                        runtime_issues.append(
                            f"Mốc [Lát {actual_wb_idx}]: "
                            f"Thời gian chờ dài ({decay_seconds // 3600} tiếng)"
                        )

                # QA check: verify that scanner DecayFactor is consistent with
                # SeriesTime-based correction for this GE dataset.
                #
                # Here the observed factors match:
                #   exp(ln(2) * (FrameReferenceTime + ActualFrameDuration/2) / T1/2)
                # to ~1e-5 relative error for the supplied case.
                stored_decay_factor = metadata.get("decay_factor")
                frame_ref_ms = metadata.get("frame_reference_time")
                actual_frame_duration_ms = metadata.get("actual_frame_duration")

                if (
                    stored_decay_factor is not None
                    and frame_ref_ms is not None
                    and actual_frame_duration_ms is not None
                ):
                    stored_decay_factor = float(stored_decay_factor)
                    frame_ref_s = float(frame_ref_ms) / 1000.0
                    half_frame_s = float(actual_frame_duration_ms) / 2000.0

                    expected_decay_factor = math.exp(
                        math.log(2.0)
                        * (frame_ref_s + half_frame_s)
                        / half_life
                    )

                    if stored_decay_factor > 0:
                        rel_err = abs(
                            stored_decay_factor - expected_decay_factor
                        ) / stored_decay_factor

                        # Be strict: if the dataset no longer follows the same GE
                        # timing convention, stop instead of silently computing SUV
                        # with a wrong reference time.
                        if rel_err > 0.01:
                            raise ValueError(
                                f"DecayFactor không khớp SeriesTime convention "
                                f"ở lát {actual_wb_idx}: stored={stored_decay_factor:.6f}, "
                                f"expected={expected_decay_factor:.6f}, "
                                f"relative_error={rel_err:.4%}"
                            )

                decay_factor_to_reference = math.exp(
                    -math.log(2.0) * decay_seconds / half_life
                )

                corrected_dose = injected_dose * decay_factor_to_reference

                if corrected_dose <= 0:
                    raise ValueError(
                        "Liều phóng xạ sau phân rã nhỏ hơn hoặc bằng 0."
                    )

                suv_factor = weight_g / corrected_dose

            elif decay_correction == "ADMIN":
                # Image is decay-corrected to administration/injection time.
                if injected_dose <= 0:
                    raise ValueError(
                        "Liều tiêm ban đầu nhỏ hơn hoặc bằng 0."
                    )

                suv_factor = weight_g / injected_dose

            elif decay_correction == "NONE":
                # Do not silently treat NONE as START.
                raise ValueError(
                    "DecayCorrection=NONE: ảnh chưa decay-correct; "
                    "pipeline hiện tại không xử lý trường hợp này."
                )

            else:
                raise ValueError(
                    f"Không hỗ trợ DecayCorrection='{decay_correction}'"
                )


            rescale_slope = float(metadata.get("rescale_slope", 1.0))

            rescale_intercept = float(metadata.get("rescale_intercept", 0.0))



            bqml_slice = (pet_volume[idx] * rescale_slope) + rescale_intercept

            suv_volume[idx] = bqml_slice * suv_factor

            valid_slice_count += 1



        if valid_slice_count != num_slices:

            raise ValueError(f"Số lượng lát cắt xử lý được ({valid_slice_count}) không khớp với tổng số lát cắt của ảnh ({num_slices}) của {pet_path}")



        avg_total_dose = float(np.mean(all_injected_doses)) if all_injected_doses else 0.0

        metadata_issues = validate_suv_parameters(filename, weight_kg, avg_total_dose, suv_volume)

        all_issues.extend(metadata_issues + runtime_issues)



        # 2. Thành công -> Đẩy ma trận SUV bình thường lên thiết bị

        suv_tensor = torch.from_numpy(suv_volume).to(device)

        return suv_tensor, all_issues, current_region



    except Exception as e:

        # 3. Gặp bất kỳ lỗi nào -> Bắt lỗi, ghi log và trả về Tensor toàn 0

        all_issues.append(f"Xử lý thất bại (trả về Tensor 0): {str(e)}")

        zero_tensor = torch.zeros(vol_shape, dtype=torch.float32, device=device)

        return zero_tensor, all_issues, current_region



# ============================================================

# RUNTIME EXAMPLE - BATCH PROCESSING WITH REAL-TIME DIAGNOSTICS

# ============================================================

if __name__ == "__main__":

    import os

    import json

    import glob

    import numpy as np

    import torch

    import torch.nn.functional as F

    from tqdm import tqdm

    from collections import Counter, defaultdict



    # 1. Cấu hình các đường dẫn gốc và năm cần quét

    base_root = "/data/backup_sde1/PET-CT-report"

    target_years = ["PETCT_2017", "PETCT_2018", "PETCT_2019", "PETCT_2023"]



    device_target = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Khởi tạo Pipeline. Device mục tiêu: {device_target.upper()}")



    # 2. Thu thập tất cả các file ảnh .npy (Loại bỏ các ca whole_body)

    all_pet_files = []

    for year in target_years:

        search_pattern = os.path.join(base_root, year, "**", "images", "**", "*.npy")

        files = glob.glob(search_pattern, recursive=True)

        filtered_files = [f for f in files if "whole_body" not in f.lower()]

        all_pet_files.extend(filtered_files)



    total_files = len(all_pet_files)

    print(f"📂 Tìm thấy tổng cộng {total_files} file ảnh PET (.npy) cần xử lý.")



    # 3. Khởi tạo các biến thống kê

    success_count = 0

    failure_count = 0

    all_reports = {}



    region_distribution = defaultdict(Counter)

    processed_counter = 0



    print("\n🎬 Bắt đầu tiến trình xử lý hàng loạt và phân tích vùng...")

    for pet_path in tqdm(all_pet_files, desc="Processing PET files"):

        filename = os.path.basename(pet_path)

        high_suv_organs_in_sample = 0



        try:

            # Chạy pipeline trích xuất SUV tensor và nhận diện vùng

            suv_tensor, issues, region = pipeline_pet_to_suv_torch(pet_path, device=device_target)



            # 🚨 KIỂM TRA VÙNG HỢP LỆ

            if region not in ORGAN_THRESHOLDS:

                tqdm.write(f"⚠️ [SAI PATH/VÙNG] Không khớp cấu hình vùng ('{region}') tại file: {pet_path}")

                if issues:

                    tqdm.write(f"   └─> Cảnh báo từ pipeline: {issues}")



                all_reports[pet_path] = {

                    "status": "SKIPPED_BAD_REGION",

                    "region": region,

                    "clinical_issues": issues

                }

                failure_count += 1

                processed_counter += 1

                continue



            # Tìm file phân vùng tương ứng

            seg_path = get_seg_path(pet_path)

            if os.path.exists(seg_path):

                seg_volume = np.load(seg_path)

                seg_tensor = torch.from_numpy(seg_volume).to(device_target)



                # -----------------------------------------------------------------

                # 🔄 TIỀN XỬ LÝ & RESIZE ĐỒNG BỘ (SHAPE MỤC TIÊU: 140, 480, 480)

                # -----------------------------------------------------------------



                # A. Khớp kích thước khối SEG (Dùng nội suy 'nearest' để bảo toàn nhãn nguyên gốc)

                seg_tensor = seg_tensor.float().unsqueeze(0).unsqueeze(0)  # Thêm Batch & Channel thành 5D

                seg_tensor = F.interpolate(seg_tensor, size=(140, 480, 480), mode='nearest')

                seg_tensor = seg_tensor.squeeze(0).squeeze(0).long()       # Trả về 3D định dạng Long



                # B. Khớp kích thước khối PET SUV (Dùng nội suy 'trilinear' cho giá trị liên tục)

                suv_tensor = suv_tensor.unsqueeze(0).unsqueeze(0)          # Thêm Batch & Channel thành 5D

                suv_tensor = F.interpolate(suv_tensor, size=(140, 480, 480), mode='trilinear', align_corners=False)

                suv_tensor = suv_tensor.squeeze(0).squeeze(0).float()      # Trả về 3D định dạng Float



                # Double-check tính an toàn ma trận trước khi tính toán roi

                if suv_tensor.shape != seg_tensor.shape:

                    raise ValueError(f"Lệch cấu trúc khối ảnh sau nội suy! SUV: {list(suv_tensor.shape)} != SEG: {list(seg_tensor.shape)}")



                # -----------------------------------------------------------------

                # ĐẾM SỐ LƯỢNG BỘ PHẬN VƯỢT NGƯỠNG SUV

                # -----------------------------------------------------------------

                config = ORGAN_THRESHOLDS[region]

                unique_labels = torch.unique(seg_tensor)

                unique_labels = unique_labels[unique_labels != 0]



                for label_id in unique_labels:

                    label_str = str(int(label_id.item()))

                    if label_str in config:

                        threshold = config[label_str]["suv_threshold"]



                        organ_mask = (seg_tensor == label_id)

                        organ_suvs = suv_tensor[organ_mask]



                        if organ_suvs.numel() > 0:

                            suv_95th = torch.quantile(organ_suvs, 0.95).item()

                            if suv_95th > threshold:

                                high_suv_organs_in_sample += 1



                # Cập nhật thống kê phân phối vùng sau khi chạy thành công ca

                region_distribution[region][high_suv_organs_in_sample] += 1



                all_reports[pet_path] = {

                    "status": "SUCCESS",

                    "region": region,

                    "shape": list(suv_tensor.shape),

                    "device": str(suv_tensor.device),

                    "high_suv_organs_count": high_suv_organs_in_sample,

                    "clinical_issues": issues

                }

                success_count += 1



            else:

                issues.append(f"Không tìm thấy file segmentation tương ứng tại: {seg_path}")

                all_reports[pet_path] = {

                    "status": "MISSING_SEGMENTATION",

                    "clinical_issues": issues

                }

                failure_count += 1



        except Exception as e:

            tqdm.write(f"❌ [CRASH BIẾN ĐỔI] Lỗi tại file [{filename}]: {str(e)}")

            all_reports[pet_path] = {

                "status": "FAILED",

                "error_details": str(e)

            }

            failure_count += 1



        processed_counter += 1



        # --- THỐNG KÊ ĐỊNH KỲ MỖI 10 SAMPLE ---

        if processed_counter % 10 == 0:

            stats_output = []

            stats_output.append(f"\n" + "="*60)

            stats_output.append(f"📊 THỐNG KÊ PHÂN PHỐI ĐỊNH KỲ TẠI SAMPLE SỐ {processed_counter}")

            stats_output.append("="*60)



            for rgn in ORGAN_THRESHOLDS.keys():

                stats_output.append(f"📍 Vùng: {rgn.upper()}")

                if rgn in region_distribution and len(region_distribution[rgn]) > 0:

                    for num_organs in sorted(region_distribution[rgn].keys()):

                        count_images = region_distribution[rgn][num_organs]

                        stats_output.append(f"  - Số ảnh có {num_organs} organ SUV cao: {count_images} ảnh")

                else:

                    stats_output.append(f"  - (Chưa có dữ liệu ảnh được xử lý thành công)")



            stats_output.append("="*60 + "\n")

            tqdm.write("\n".join(stats_output))



    # 5. Xuất báo cáo tổng quan cuối cùng

    print("\n" + "📊" * 15 + " THỐNG KÊ TỔNG KẾT TIẾN TRÌNH " + "📊" * 15)

    print(f"• Tổng số file quét được: {total_files}")

    print(f"• Xử lý THÀNH CÔNG:       {success_count} / {total_files} file")

    print(f"• Xử lý THẤT BẠI/BỎ QUA:  {failure_count} / {total_files} file")



    log_output_path = "./batch_processing_report.json"

    with open(log_output_path, "w", encoding="utf-8") as f:

        json.dump(all_reports, f, ensure_ascii=False, indent=4)

    print(f"📝 Đã xuất báo cáo chi tiết từng file tại: {log_output_path}")