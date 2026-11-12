#!/usr/bin/env python3
"""
PET2Rep-faithful adaptation for Vietnamese PET/CT .npy data.

Goal
----
Reproduce the public PET2Rep Qwen2.5-VL inference path as closely as possible,
while adapting only the unavoidable dataset/model differences:

1) Backbone: Qwen2.5-VL-32B-Instruct.
2) User data are 3D NumPy arrays with shape X x Y x Y. PET2Rep samples CORONAL
   planes, so this adaptation reformats the volume and samples along Y (axis 1
   by default), rather than treating axis 0 axial slices as model images.
3) 10 coronal locations are uniformly sampled across the full volume.
4) Each fused PET/CT image is downsampled by 4x along both spatial dimensions
   before being passed to Qwen (e.g. 256x256 -> 64x64).
5) PET path comes from align_test.json; paired CT is inferred by
   /images/ -> /ref_images/.
6) Output format remains the requested evaluator JSONL:
   question_id / prompt / text / metadata.
7) Only the selected question IDs are evaluated.

PET2Rep-faithful parts
----------------------
- Qwen2_5_VLForConditionalGeneration
- Qwen2.5-VL-32B-Instruct
- 8-bit bitsandbytes quantization for model weights
- attn_implementation="sdpa"
- device_map={"": 0} to keep the quantized model on GPU 0
- AutoProcessor + apply_chat_template + process_vision_info
- images first, then one text prompt in the user message
- GenerationConfig:
    temperature=0.7
    max_new_tokens=1024
    top_p=1
    top_k=50
    do_sample=True
    repetition_penalty=1
- Coronal sampling.
- Input mode is FUSED PET/CT (PET overlay on CT).

Important limitation
--------------------
This variant intentionally departs from the paper's three-key-slice protocol:
it uniformly samples 10 coronal slices across the full volume.

PET is pseudo-colored with a deterministic JET-style colormap; CT remains grayscale.
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


DEFAULT_ANNOTATION = (
    "/data/backup_sde1/PET-CT-report/"
    "pretrain_data/single_turn/align_test.json"
)

DEFAULT_IMAGE_FOLDER = "/data/backup_sde1/PET-CT-report"

DEFAULT_OUTPUT = (
    "/home/nvidia-lab/ai4life/CTPET/baseline/"
    "pet2rep_qwen25vl_32b_10_coronal_fused_down4_int8.jsonl"
)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

NUM_CORONAL_SLICES = 10
IMAGE_DOWNSAMPLE_FACTOR = 4


TARGET_IDS = {
    "00f8634df7c9acb37ad3d4f8d19043f745fd1d7b8acad3004a85713a85a74cf2",
    "020124357a5436865344db666fdf8609c424b7adfd02fd4b8bf86f083b7f6286",
    "04fecb361db4ce4894ab9239279b8e75b78fe7636e5261743c6a073eeec2be77",
    "071fa6b0e48f295dab9b1276ddbceb76c3c1242a6202b0198c51bc03a9a2508f",
    "08589ee0508b59cdfd7f9ff281bd11c3782f9865637d994a064128296cbaa9a6",
    "0ad538980bf59b58bce9d8841ef4bbe06a552cb688c34fdd6561093816b43e70",
    "0ec3e4b76c5977976aceb3590d5596ff9cae04342fb731f24449a24e29831042",
    "0f24c9d580e30115d3fd56deadcb7b7297ffde0abea5388c3e042d39e8807bb8",
    "143a5519ac9378cde070c126f8d65c3c5d5cc4e5cd675a2312ad3118e1096456",
    "14816cb5a6fd7bfbd22c492a3aa285e008292b6bab27d7cc49bf1140cf9b8708",
    "14a36d4ba479f115af477231754ba50df8e4b7e73b9c623463d74dfd1923b00e",
    "1a8e51b92cc523b50190c822b8b25dc412e615e2e5fd41e0827d45944bfcd309",
    "1e2bc8139554fc607bc65093b26d8e22140e8b7debd9b963bfaebd6ed93de660",
    "21e2739da67e98230bafc3bd39e9d451d15178a2b1fa07d838293fff3037c96c",
    "29da840177a53491acf65fdb939f1678ce139c6d9158816ed0188f9a05f0cf0d",
    "2d7bfc998a4d1b2faf3310b6744f4a134f19b11aa7fd52bc57fa7dd7b779b9ac",
    "33f51130b8f12bfd19c140c873bc13d1fd853882c76252cf9735a72db951ed54",
    "37c21d01da39b77166461a72a652380bee260402d0fea828f800ad2cc535b0ef",
    "38fad129c031b1049009144a46c2af28ce5ed3a2001c6a42065b48c495e2b371",
    "3b570f4bb4507a1a02b7333ab19a337f30a4f69497c2b6f5a6d6f1b84772a70e",
    "3c04bfcf6c0c598f81e1333032b6e6b92a7078f2a29994edbe8a2bff6fd7e7a9",
    "4216d2d19923f7a498c2e4cf9447178016d0d7a7325b2baf007c4303f9085708",
    "4d095c395a8e269057e138cfdcbf144a538ad74356641b6653e1d9bdcae29861",
    "4eaaf727ffccb7175314714213d82f8552c9e493604a97faa45b446b93614655",
    "524edcebc5eb5bd9f72579560520e8ccfcdae86d1acf7139a418f121319b91cb",
    "533ee24348a95707ccbdefd78aea0022914397874c130bcf8c74519599669e7c",
    "5ab0e56a5d9ee03eb98f40854bac711667bd22ec942646e81b95c6031cefceb3",
    "5acd51ce558544cea977e1f4b90cd82e58bf2925493142350b5ea7736645601a",
    "5ea805dc47b8d0adea132e41136c3b915f9dca953c48eff8c6b3cf88f8248d26",
    "60446bd852cea96e64aaa095d492df84c799471e48f1747d26efeba0c1eb510e",
    "64c376385a25399b7ad8b2268116f8df293d2858bd4342b779911981a73e3d10",
    "6983cd84d82f786b99be6adc16fb20637d312a792499f8e2d89e4ccfde12c8c5",
    "6a3683092109c02d78b7d346b940ab4b504258d40e0e24e30396f856f6fc2deb",
    "6a9a2435a790ad8067584be6076e0c2f67bf67ba66a82075c69616ab80345716",
    "6ab1fc3c7dcf1bf0a33f26d16205f3111a0f53be1621987d5659b99b53fb772e",
    "7b7e9ec1aa516b31551384f3c781f1cf4309a6c6752efd6eb20671c9e92fb379",
    "7d1b44d21e955794b8cc95643670cee801aa47a9144b37a4d2a6e039c138163b",
    "88c24589d1306418552ff357c4f73b89dcfeace629cd9cbc909d9c139a99bdfc",
    "8da4ab3c1d777d164c5a8eb05bc2bc661f16e4eadacfc7161b2f49e3ed5e35ca",
    "9135f3b5bb2eeaa772523bef9d912f60c3f0ea402d1af9dac2f171352e086f64",
    "9245eee65933f830451b7a3a7e22ed5a6ac957bc212b5aa6ee6cedf56bd5d5f8",
    "9523daa50bc17eeb49d566f8463cbbb86e42b81ac0b83a785a6e47d9a6afcbee",
    "9855a518457b454a4459df5cc57f597004c0250dbd657d84238897c36e546685",
    "99db219ee879257cdce1771cbb9e6003848b873be8bf706b41b9a427a82b5dbd",
    "9d87e3bd427e85edad4eb344420a0effa2cf065810c722e00724a274730488a3",
    "a0ca5f53d37db4b2c3431eb28d963e74e561b262a96cc6b316f19f3da6c44d27",
    "a4529d5641786ac8256e1afa05ff9e70e0dffb8bcdd9fbf0df4a62c7ea1659b1",
    "abab0dd7ae03d6610684f29d791992d14753e615f20cb10b6ac3c75d80396dbe",
    "abf921ecc1129da102b92284e2ee4d4cd18eeada1d5626401d32a586784201c0",
    "ac8f8ccf59e73dd1d79c9a40bf0e9813fa92af4f683a9c0d3c66bb2c3fbafe4d",
    "b38ae4ee3c102526e4788842525ef450c88004f38c46e63b43f705aad3b61227",
    "b8d2279d009f56d754e50246ee25243f2e81b9ef9c67cfb4e26586e08e425bf6",
    "ba387f5af4cd141634ef4bdd9ab705b088099e681292ccbd2102854fba330f9b",
    "bb0455ad67dfdf7fca8a5f090e08dc09ed8072a32ade06a20d5a717f59abb725",
    "bd4cb7f18298a058d609bc056076b6ef0623003c9983a89d62fbe7a1c395a8a5",
    "bda36d0f612399a953aec687c379ddb3758629afb3c4409c11cb7f6e51a2da51",
    "c757d1f0fae95e2be1c3f7aa0fc91a9ad24a3a6f58c5c54d05ef39333efba8f7",
    "c798c745aefbbbaf5a3c7214e880f66ebd112a01a25bddb136b2db4ec158b1ab",
    "c85fec95150d2c5e3bf5c9ff9ddeb62b53645bfc3b0ff404afedda077b32edb4",
    "ccd01a8f09c546c3a157f9db2ae338511486c8f3fa763b2cba00b6298218fbd9",
    "d68f10ebd9e2b506c40f7670c30cf8541fcb8f59aa5bf3942449cf541f29f3c9",
    "d94ca737010ff95eb0d3cba211918762e437f6527954acc4b499a200724a02fe",
    "d99aff7a75579b65f2615146b33d3874678fba9972dd4dda9aa734c28449a160",
    "db7d6a15500fcc165bfeec033ccbaf63fbc3177c80f9faea55feb8f875067ff4",
    "e184f11502f2c6ad4010d2c11a5bb16b95033e8a9bee06059e73f3737f94a9cd",
    "e51e02634a56bf663bdcb1768c15d8749ce18e28f9b995479f9a892c10843bf3",
    "e835e3dc2d2a2b315df2dc822176e787033ede955b9f831a1b226c5a6a48096d",
    "e8db532055b1877e6567f43f5103456f62414e27c7b60ee65a09c9f5f735a748",
    "e99272f4f4f99bdac579471cd8fb0d244733d4d77a14288dee98b05dd091c3d0",
    "eb34f004bff145c61ef20e632e58b5882d6ef6c6f5e074c44feaf89e192de5c4",
    "f6e9c722b9be72c593b72e6e2236b6ca02aa230acd058328e3ced3569e487c97",
}


SYSTEM_PROMPT = (
    "Bạn là một bác sĩ y học hạt nhân giàu kinh nghiệm, có khả năng kết hợp "
    "hình ảnh PET/CT để viết báo cáo chẩn đoán."
)


TASK_PROMPT = (
    "Dựa trên các hình ảnh FDG-PET/CT được cung cấp, hãy viết phần mô tả hình ảnh "
    "bằng tiếng Việt theo văn phong báo cáo PET/CT lâm sàng.\n\n"

    "YÊU CẦU FORMAT:\n"
    "- Chỉ trả về phần mô tả hình ảnh, không viết tiêu đề, kết luận, nhận xét, "
    "giải thích hay khuyến cáo.\n"
    "- Không sử dụng bullet point, đánh số hoặc Markdown.\n"
    "- Viết thành một đoạn văn liên tục gồm các câu ngắn theo phong cách báo cáo.\n"
    "- Ưu tiên bắt đầu các phát hiện bằng các cấu trúc như: "
    "'Hình ảnh ...', 'Tăng nhẹ chuyển hóa FDG ...', "
    "'Chưa phát hiện hình ảnh ...'.\n"
    "- Trước tiên mô tả các vùng bắt xạ sinh lý nếu quan sát được, sau đó lần lượt "
    "mô tả các phát hiện bất thường.\n"
    "- Với mỗi tổn thương, mô tả vị trí giải phẫu, kích thước hoặc đặc điểm hình thái "
    "trên CT nếu xác định được, sau đó mô tả đặc điểm chuyển hóa FDG.\n"
    "- Sử dụng các cụm từ theo văn phong lâm sàng như "
    "'theo đặc điểm sinh lý', 'tăng chuyển hóa FDG', "
    "'không tăng chuyển hóa FDG', 'khả năng do viêm hoặc quá sản' "
    "khi phù hợp với hình ảnh.\n"
    "- Không thêm phần 'Cần đối chiếu...', 'Khuyến cáo...', 'Kết luận...' hoặc "
    "các nội dung ngoài phần mô tả hình ảnh.\n"
    "- Không tự tạo tổn thương không quan sát được.\n"
    "- Không tự tạo giá trị SUVmax hoặc kích thước nếu không thể xác định chắc chắn "
    "từ dữ liệu đầu vào.\n\n"

    "PHONG CÁCH ĐẦU RA MONG MUỐN:\n"
    "Hình ảnh bắt xạ theo đặc điểm sinh lý ở não. "
    "Tăng nhẹ chuyển hóa FDG ở amydal, rìa lưỡi hai bên theo đặc điểm sinh lý. "
    "Hình ảnh ... tăng chuyển hóa FDG. "
    "Hình ảnh ... không tăng chuyển hóa FDG khả năng do viêm hoặc quá sản."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PET2Rep-Fused Qwen2.5-VL-32B inference on "
            "10 uniformly sampled coronal PET/CT slices."
        )
    )

    parser.add_argument("--annotation_file", default=DEFAULT_ANNOTATION)
    parser.add_argument("--image_folder", default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)

    parser.add_argument(
        "--coronal_axis",
        type=int,
        choices=[1, 2],
        default=1,
        help=(
            "For XxYxY volumes, default assumes array order (z, y, x), "
            "so coronal = volume[:, y, :] (axis 1). Use 2 only if the "
            "in-plane orientation is transposed."
        ),
    )

    parser.add_argument(
        "--num_slices",
        type=int,
        default=NUM_CORONAL_SLICES,
        help=(
            "Number of uniformly sampled coronal slices. "
            "Default: 10. If the volume has fewer slices, all slices are used."
        ),
    )

    parser.add_argument(
        "--append_dataset_question",
        action="store_true",
        help=(
            "Append the original generic align_test question to PET2Rep's task prompt. "
            "OFF by default for closer reproduction of the official PET2Rep prompt flow."
        ),
    )

    parser.add_argument(
        "--fuse_alpha",
        type=float,
        default=0.50,
        help=(
            "Base PET overlay strength for fused visualization. "
            "The effective alpha is fuse_alpha * PET_normalized at each pixel."
        ),
    )

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help=(
            "Validate paths, XxYxY shapes, and 10-slice coronal extraction "
            "without loading Qwen."
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("align_test.json must contain a JSON array.")

    return data


def get_human_prompt(item: dict) -> str:
    for turn in item.get("conversations", []):
        if turn.get("from") == "human":
            return str(turn.get("value", ""))

    return "<image>\nHãy mô tả hình ảnh này."


def infer_ct_relative_path(pet_relative_path: str) -> str:
    p = pet_relative_path.replace("\\", "/")

    if "/images/" not in p:
        raise ValueError(
            f"Cannot infer CT path because '/images/' is missing: "
            f"{pet_relative_path}"
        )

    return p.replace("/images/", "/ref_images/", 1)


def assert_xyy_volume(arr: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected X x Y x Y volume, got shape={arr.shape} at {path}"
        )

    if arr.shape[1] != arr.shape[2]:
        raise ValueError(
            f"Expected X x Y x Y volume, got shape={arr.shape} at {path}"
        )

    return arr


def choose_coronal_indices(
    num_slices: int,
    num_selected: int = 10,
) -> List[int]:
    """
    Select up to `num_selected` coronal slices uniformly over the full volume.

    Examples:
      num_slices = 256, num_selected = 64
      -> approximately [0, 4, 8, ..., 251, 255]

    If num_slices <= num_selected, every slice is retained.
    """
    if num_slices <= 0:
        raise ValueError("Volume has no coronal slices.")

    if num_selected <= 0:
        raise ValueError("num_selected must be > 0.")

    if num_slices <= num_selected:
        return list(range(num_slices))

    indices = np.linspace(
        0,
        num_slices - 1,
        num=num_selected,
        dtype=np.int64,
    )

    return indices.tolist()


def finite_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def robust_unit_interval(
    x: np.ndarray,
    low_pct=1.0,
    high_pct=99.0,
) -> np.ndarray:
    """
    Display normalization only.

    PET2Rep's released inference script consumes already-prepared JPEGs;
    the exact JPEG windowing is not public.
    """
    x = finite_float32(x)
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(
        finite,
        [low_pct, high_pct],
    )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):
        lo = float(finite.min())
        hi = float(finite.max())

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip(
        (x - lo) / (hi - lo),
        0.0,
        1.0,
    ).astype(np.float32)


def zscore_volume(ct: np.ndarray) -> np.ndarray:
    """
    PET2Rep paper: CT intensities are standardized using z-score normalization.
    """
    ct = finite_float32(ct)

    mean = float(ct.mean())
    std = float(ct.std())

    if std < 1e-6:
        return ct - mean

    return (ct - mean) / std


def gray_to_rgb(gray01: np.ndarray) -> np.ndarray:
    gray01 = np.clip(
        gray01,
        0.0,
        1.0,
    )

    return np.repeat(
        gray01[..., None],
        3,
        axis=-1,
    )


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """
    Dependency-free JET-like pseudocolor for PET.

    PET2Rep states PET is pseudo-colored;
    exact colormap is not released.
    """
    x = np.clip(
        x,
        0.0,
        1.0,
    )

    r = np.clip(
        1.5 - np.abs(4.0 * x - 3.0),
        0.0,
        1.0,
    )

    g = np.clip(
        1.5 - np.abs(4.0 * x - 2.0),
        0.0,
        1.0,
    )

    b = np.clip(
        1.5 - np.abs(4.0 * x - 1.0),
        0.0,
        1.0,
    )

    return np.stack(
        [r, g, b],
        axis=-1,
    ).astype(np.float32)


def to_pil_rgb(rgb01: np.ndarray) -> Image.Image:
    rgb = np.clip(
        rgb01,
        0.0,
        1.0,
    )

    return Image.fromarray(
        np.round(rgb * 255.0).astype(np.uint8),
        mode="RGB",
    )


def downsample_pil_image(
    img: Image.Image,
    factor: int = IMAGE_DOWNSAMPLE_FACTOR,
) -> Image.Image:
    """
    Reduce both image dimensions by `factor`.

    Example:
        256 x 256 -> 64 x 64 when factor=4.
        512 x 256 -> 128 x 64 when factor=4.
    """
    if factor <= 0:
        raise ValueError("Downsample factor must be > 0.")

    width, height = img.size
    new_width = max(1, width // factor)
    new_height = max(1, height // factor)

    return img.resize(
        (new_width, new_height),
        resample=Image.Resampling.BILINEAR,
    )


def extract_coronal(
    volume: np.ndarray,
    index: int,
    coronal_axis: int,
) -> np.ndarray:
    """
    X x Y x Y volume -> 2D coronal plane.

    Standard array order after stacking axial DICOM slices is commonly:
        (z, y, x)

    Therefore:
        axis 0 fixed -> axial
        axis 1 fixed -> coronal
        axis 2 fixed -> sagittal

    Default:
        coronal_axis = 1
    """
    if coronal_axis == 1:
        return np.asarray(
            volume[:, index, :]
        )

    if coronal_axis == 2:
        return np.asarray(
            volume[:, :, index]
        )

    raise ValueError(
        f"coronal_axis must be 1 or 2, got {coronal_axis}"
    )


def map_index_between_grids(
    index: int,
    src_length: int,
    dst_length: int,
) -> int:
    """
    Map normalized PET coronal position to the corresponding CT position.
    """
    if src_length <= 1 or dst_length <= 1:
        return 0

    relative = index / float(
        src_length - 1
    )

    mapped = int(
        round(
            relative * (dst_length - 1)
        )
    )

    return max(
        0,
        min(
            dst_length - 1,
            mapped,
        ),
    )


def resize_float_plane(
    plane: np.ndarray,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    x = robust_unit_interval(
        plane,
        0.5,
        99.5,
    )

    img = Image.fromarray(
        np.round(x * 255.0).astype(np.uint8),
        mode="L",
    )

    img = img.resize(
        (out_hw[1], out_hw[0]),
        resample=Image.Resampling.BILINEAR,
    )

    return (
        np.asarray(
            img,
            dtype=np.float32,
        )
        / 255.0
    )


def fuse_pet_ct_rgb(
    pet_rgb: np.ndarray,
    ct_rgb: np.ndarray,
    pet01: np.ndarray,
    base_alpha: float,
) -> np.ndarray:
    """
    Create fused PET/CT visualization by superimposing pseudo-colored PET
    onto grayscale CT.
    """
    pet_rgb = np.clip(
        pet_rgb,
        0.0,
        1.0,
    )

    ct_rgb = np.clip(
        ct_rgb,
        0.0,
        1.0,
    )

    alpha = (
        np.clip(
            base_alpha,
            0.0,
            1.0,
        )
        * np.clip(
            pet01,
            0.0,
            1.0,
        )[..., None]
    )

    fused = (
        (1.0 - alpha) * ct_rgb
        + alpha * pet_rgb
    )

    return np.clip(
        fused,
        0.0,
        1.0,
    ).astype(np.float32)


def prepare_pet_ct_key_images(
    pet_path: Path,
    ct_path: Path,
    coronal_axis: int,
    fuse_alpha: float,
    num_slices: int,
) -> Tuple[
    List[Image.Image],
    List[int],
    List[int],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    """
    User data shape:
        X x Y x Y

    Select up to `num_slices` uniformly distributed coronal planes
    over the complete PET volume.

    Input to Qwen:
        FUSED PET/CT image at every selected coronal position.

    With the default setting:
        10 fused PET/CT images per study.
        Each image is reduced by 4x in width and 4x in height before Qwen.
    """
    pet = assert_xyy_volume(
        np.load(
            pet_path,
            allow_pickle=False,
        ),
        pet_path,
    )

    ct = assert_xyy_volume(
        np.load(
            ct_path,
            allow_pickle=False,
        ),
        ct_path,
    )

    pet_coronal_length = pet.shape[
        coronal_axis
    ]

    ct_coronal_length = ct.shape[
        coronal_axis
    ]

    pet_indices = choose_coronal_indices(
        num_slices=pet_coronal_length,
        num_selected=num_slices,
    )

    ct_indices = [
        map_index_between_grids(
            idx,
            src_length=pet_coronal_length,
            dst_length=ct_coronal_length,
        )
        for idx in pet_indices
    ]

    # PET2Rep:
    # CT standardized with z-score normalization.
    ct_z = zscore_volume(ct)

    images: List[Image.Image] = []

    for pet_idx, ct_idx in zip(
        pet_indices,
        ct_indices,
    ):
        pet_plane = finite_float32(
            extract_coronal(
                pet,
                pet_idx,
                coronal_axis,
            )
        )

        ct_plane = finite_float32(
            extract_coronal(
                ct_z,
                ct_idx,
                coronal_axis,
            )
        )

        # PET pseudo-color
        pet01 = robust_unit_interval(
            pet_plane,
            1.0,
            99.5,
        )

        pet_rgb = jet_colormap(
            pet01
        )

        # CT resized to PET coronal plane resolution
        ct01 = resize_float_plane(
            ct_plane,
            pet_plane.shape,
        )

        ct_rgb = gray_to_rgb(
            ct01
        )

        # PET/CT fused image
        fused_rgb = fuse_pet_ct_rgb(
            pet_rgb=pet_rgb,
            ct_rgb=ct_rgb,
            pet01=pet01,
            base_alpha=fuse_alpha,
        )

        fused_img = to_pil_rgb(
            fused_rgb
        )

        # Reduce each spatial dimension by 4x before sending the image to Qwen.
        # Example: 256x256 -> 64x64.
        fused_img = downsample_pil_image(
            fused_img,
            factor=IMAGE_DOWNSAMPLE_FACTOR,
        )

        images.append(
            fused_img
        )

    return (
        images,
        pet_indices,
        ct_indices,
        tuple(pet.shape),
        tuple(ct.shape),
    )


def build_pet2rep_messages(
    images: Sequence[Image.Image],
    original_prompt: str,
    append_dataset_question: bool = False,
) -> List[dict]:
    """
    Match public PET2Rep message structure:

        system
        user:
            image
            image
            ...
            image
            text prompt

    With default settings this means:
        10 image entries
        followed by one textual instruction.
    """
    dataset_question = (
        original_prompt
        .replace("<image>", "")
        .strip()
    )

    prompt_user_text = TASK_PROMPT

    if (
        append_dataset_question
        and dataset_question
    ):
        prompt_user_text += (
            f"\n\nYêu cầu của bộ dữ liệu: "
            f"{dataset_question}"
        )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": img,
                    }
                    for img in images
                ],
                {
                    "type": "text",
                    "text": prompt_user_text,
                },
            ],
        },
    ]


def load_done_ids(
    output_file: Path,
) -> set:
    done = set()

    if not output_file.exists():
        return done

    with output_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = row.get(
                "question_id"
            )

            if qid:
                done.add(
                    str(qid)
                )

    return done


def append_jsonl(
    path: Path,
    row: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def filter_selected(
    data: Sequence[dict],
) -> Tuple[List[dict], set]:
    all_ids = {
        str(item.get("id"))
        for item in data
    }

    missing = (
        TARGET_IDS
        - all_ids
    )

    selected = [
        item
        for item in data
        if str(item.get("id"))
        in TARGET_IDS
    ]

    # Preserve align_test.json order.
    return selected, missing


@torch.inference_mode()
def infer_one(
    model,
    processor,
    messages,
    generation_config,
) -> str:
    """
    Public PET2Rep / Qwen2.5-VL inference path.
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = inputs.to(
            "cuda"
        )
    else:
        inputs = inputs.to(
            next(
                model.parameters()
            ).device
        )

    generated_ids = model.generate(
        **inputs,
        generation_config=generation_config,
    )

    generated_ids_trimmed = [
        out_ids[
            len(in_ids):
        ]
        for in_ids, out_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = (
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )

    return (
        output_text[0]
        .strip()
    )


def validate_samples(
    data: Sequence[dict],
    image_folder: Path,
    coronal_axis: int,
    fuse_alpha: float,
    num_slices: int,
    limit: int,
):
    for item in data[
        : min(
            limit,
            len(data),
        )
    ]:
        qid = str(
            item["id"]
        )

        pet_rel = (
            str(item["image"])
            .replace("\\", "/")
        )

        ct_rel = (
            infer_ct_relative_path(
                pet_rel
            )
        )

        pet_path = (
            image_folder
            / pet_rel
        )

        ct_path = (
            image_folder
            / ct_rel
        )

        if not pet_path.exists():
            raise FileNotFoundError(
                f"Missing PET: {pet_path}"
            )

        if not ct_path.exists():
            raise FileNotFoundError(
                f"Missing CT: {ct_path}"
            )

        (
            images,
            pet_indices,
            ct_indices,
            pet_shape,
            ct_shape,
        ) = prepare_pet_ct_key_images(
            pet_path=pet_path,
            ct_path=ct_path,
            coronal_axis=coronal_axis,
            fuse_alpha=fuse_alpha,
            num_slices=num_slices,
        )

        print(
            f"[VALID] {qid}\n"
            f"  PET shape: {pet_shape}\n"
            f"  CT  shape: {ct_shape}\n"
            f"  coronal axis: {coronal_axis}\n"
            f"  requested slices: {num_slices}\n"
            f"  selected PET slices: {len(pet_indices)}\n"
            f"  PET indices: {pet_indices}\n"
            f"  CT indices : {ct_indices}\n"
            f"  input_mode: fused PET/CT\n"
            f"  images sent to VLM: {len(images)}\n"
            f"  image downsample factor: {IMAGE_DOWNSAMPLE_FACTOR}x per dimension\n"
            f"  image size sent to VLM: {images[0].size if images else None}\n"
            f"  PET: {pet_path}\n"
            f"  CT : {ct_path}"
        )


def main():
    args = parse_args()

    annotation_file = Path(
        args.annotation_file
    )

    image_folder = Path(
        args.image_folder
    )

    output_file = Path(
        args.output_file
    )

    all_data = load_json(
        annotation_file
    )

    data, missing_ids = (
        filter_selected(
            all_data
        )
    )

    print(
        "[INFO] PET2Rep-faithful adaptation"
    )

    print(
        f"[INFO] annotation_file = "
        f"{annotation_file}"
    )

    print(
        f"[INFO] image_folder    = "
        f"{image_folder}"
    )

    print(
        f"[INFO] output_file     = "
        f"{output_file}"
    )

    print(
        f"[INFO] model           = "
        f"{args.model_path}"
    )

    print(
        "[INFO] input_mode      = "
        "fused PET/CT"
    )

    print(
        f"[INFO] fuse_alpha      = "
        f"{args.fuse_alpha}"
    )

    print(
        f"[INFO] target IDs      = "
        f"{len(TARGET_IDS)}"
    )

    print(
        f"[INFO] matched samples = "
        f"{len(data)}"
    )

    print(
        "[INFO] volume shape    = "
        "X x Y x Y"
    )

    print(
        f"[INFO] coronal axis    = "
        f"{args.coronal_axis}"
    )

    print(
        f"[INFO] slice selector  = "
        f"{args.num_slices} uniformly sampled coronal slices"
    )

    print(
        f"[INFO] image resize    = "
        f"1/{IMAGE_DOWNSAMPLE_FACTOR} width x "
        f"1/{IMAGE_DOWNSAMPLE_FACTOR} height"
    )

    if missing_ids:
        print(
            f"[WARNING] "
            f"{len(missing_ids)} selected IDs "
            f"not found in align_test.json:"
        )

        for qid in sorted(
            missing_ids
        ):
            print(
                f"          {qid}"
            )

    if not data:
        raise RuntimeError(
            "No selected IDs were found "
            "in align_test.json."
        )

    if args.validate_only:
        validate_samples(
            data=data,
            image_folder=image_folder,
            coronal_axis=args.coronal_axis,
            fuse_alpha=args.fuse_alpha,
            num_slices=args.num_slices,
            limit=args.limit or 3,
        )
        return

    done_ids = (
        load_done_ids(
            output_file
        )
        if args.resume
        else set()
    )

    if done_ids:
        print(
            f"[INFO] resume: "
            f"{len(done_ids)} IDs "
            f"already in output"
        )

    print(
        f"[INFO] loading model: "
        f"{args.model_path}"
    )

    print(
        "[INFO] quantization    = bitsandbytes INT8"
    )

    # 8-bit quantization reduces Qwen2.5-VL-32B GPU weight memory from
    # roughly ~64 GB in BF16 to roughly ~32-40 GB total in practice,
    # depending on CUDA/vision-module overhead and the exact environment.
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_path
        )
    )

    # PET2Rep generation configuration.
    generation_config = (
        GenerationConfig(
            temperature=0.7,
            max_new_tokens=args.max_new_tokens,
            top_p=1,
            top_k=50,
            do_sample=True,
            repetition_penalty=1,
        )
    )

    processed = 0

    for item in data:
        if (
            args.limit is not None
            and processed >= args.limit
        ):
            break

        question_id = str(
            item["id"]
        )

        if question_id in done_ids:
            continue

        original_prompt = (
            get_human_prompt(
                item
            )
        )

        pet_rel = (
            str(item["image"])
            .replace("\\", "/")
        )

        ct_rel = (
            infer_ct_relative_path(
                pet_rel
            )
        )

        pet_path = (
            image_folder
            / pet_rel
        )

        ct_path = (
            image_folder
            / ct_rel
        )

        if not pet_path.exists():
            raise FileNotFoundError(
                f"Missing PET: "
                f"{pet_path}"
            )

        if not ct_path.exists():
            raise FileNotFoundError(
                f"Missing CT: "
                f"{ct_path}"
            )

        (
            images,
            pet_indices,
            ct_indices,
            pet_shape,
            ct_shape,
        ) = prepare_pet_ct_key_images(
            pet_path=pet_path,
            ct_path=ct_path,
            coronal_axis=args.coronal_axis,
            fuse_alpha=args.fuse_alpha,
            num_slices=args.num_slices,
        )

        messages = (
            build_pet2rep_messages(
                images=images,
                original_prompt=original_prompt,
                append_dataset_question=(
                    args.append_dataset_question
                ),
            )
        )

        print(
            f"[INFER] {question_id} | "
            f"images={len(images)} | "
            f"image_size={images[0].size if images else None}",
            flush=True,
        )

        prediction = infer_one(
            model=model,
            processor=processor,
            messages=messages,
            generation_config=(
                generation_config
            ),
        )

        result = {
            "question_id": question_id,
            "prompt": original_prompt,
            "text": prediction,
            "metadata": {},
        }

        append_jsonl(
            output_file,
            result,
        )

        processed += 1

        print(
            f"[{processed}] "
            f"{question_id} | "
            f"PET_shape={pet_shape} | "
            f"CT_shape={ct_shape} | "
            f"PET_coronal={len(pet_indices)} slices | "
            f"CT_coronal={len(ct_indices)} slices | "
            f"fused_images={len(images)} | "
            f"chars={len(prediction)}"
        )

        del images
        del messages

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[DONE] wrote "
        f"{processed} predictions "
        f"to {output_file}"
    )


if __name__ == "__main__":
    main()