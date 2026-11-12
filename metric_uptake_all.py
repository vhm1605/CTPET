import os
import json
import re
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import nltk
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score


warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
GT_PATH = "/data/backup_sde1/PET-CT-report/pretrain_data/single_turn/align_test.json"

OUT_DIR = "/home/nvidia-lab/ai4life/CTPET/metric_out"
os.makedirs(OUT_DIR, exist_ok=True)

BERT_MODEL = "/home/nvidia-lab/ai4life/bert-multilingual"

MODEL_PATHS = {
#     # -------- MISTRAL --------
#     "mistral_only_pet": "/data/ychieu/results/noise_ctvit_med.jsonl",

#     "mistral_suv_ratio": "/home/nvidia-lab/ai4life/CTPET/ctvit_llavamed/infer/lora_region_resume/checkpoint-11108/test/answer.jsonl",

#     "mistral_suv_threshold": "/home/nvidia-lab/ai4life/CTPET_new_filter/ctvit_llavamed/infer/test_region_new_suv_selection_final/checkpoint-22216/answer.jsonl",

#     # Nếu muốn chạy thêm PETCT pure trong code cũ thì mở dòng này
#     "mistral_petct_pure": "/data/ychieu1/checkpoint-19502/test/answers.jsonl",

    # -------- QWEN --------
    #"qwen_only_pet": "/home/nvidia-lab/ai4life/kienpt/PET-CT/ctvit_qwen_instruct/checkpoints/onlypet/infer/test/answer.jsonl",
    "qwen_suv_threshold_old": "/home/nvidia-lab/ai4life/PETCT_qwen/checkpoints/petct_region_new_suv/infer/checkpoint-10830/answer.jsonl",

    "qwen_suv_threshold": "/home/nvidia-lab/ai4life/PETCT_qwen/checkpoints/petct_region_new_suv_low_lr/infer/checkpoint-5415/answer.jsonl",

    "qwen_suv_ratio": "/home/nvidia-lab/ai4life/PETCT_qwen/checkpoints/petct_region_fix/infer/checkpoint-10830/answer.jsonl",

    "qwen_pet_ct": "/home/nvidia-lab/ai4life/kienpt/PET-CT/ctvit_qwen_instruct/checkpoints/petct_fix_prompt_final_v5/infer1_512/test/answer.jsonl",

    # qwen_suv_threshold: đang tune, chưa có path

    # # -------- M3D --------
    # "m3d_only_pet": "/data/ychieu/results/noise_ctvit_m3d_projector.jsonl",

    # # m3d_ctpet: đang tune, chưa có path

    # "m3d_suv_ratio": "/home/nvidia-lab/ai4life/CTPET_new_filter/ctvit_m3d/infer/lora_resume/checkpoint-16662/answer.jsonl",

    # "m3d_suv_threshold": "/home/nvidia-lab/ai4life/CTPET_new_filter/ctvit_m3d/infer/lora_new_suv/checkpoint-16662/answer.jsonl",
}


# ======================================================
# NLTK
# ======================================================
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

smooth = nltk.translate.bleu_score.SmoothingFunction().method4


# ======================================================
# CLEAN CHAT MARKER
# ======================================================
def clean_chat_markers(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"(?m)^\s*(assistant|user)\s*\n?",
        "",
        text
    )

    text = re.sub(
        r"\n[ \t]*\n+",
        "\n",
        text
    )

    return text.strip()


# ======================================================
# PET UPTAKE FILTER
# ======================================================
POSITIVE_PATTERNS = [
    r"tăng\s*(nhẹ\s*)?chuyển\s*h[oó]a",
    r"không\s*tăng\s*chuyển\s*h[oó]a",
    r"\bfdg\b",
    r"suv\s*max",
    r"suvmax",
    r"bắt\s*xạ",
]

NEGATIVE_PATTERNS = [
    r"đặc điểm sinh lý",
    r"bắt xạ sinh lý",
    r"sinh lý ở tim",
    r"sinh lý ở gan",
    r"sinh lý ở não",
    r"phân bố sinh lý",
    r"bài tiết sinh lý",
]

positive_re = re.compile(
    "|".join(POSITIVE_PATTERNS),
    flags=re.IGNORECASE
)

negative_re = re.compile(
    "|".join(NEGATIVE_PATTERNS),
    flags=re.IGNORECASE
)


def split_sentences(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    sents = re.split(
        r"(?<=[.!?;])\s+",
        text
    )

    return [s.strip() for s in sents if s.strip()]


def extract_pet_uptake_sentences(text):
    sentences = split_sentences(text)

    outputs = []

    for sent in sentences:
        sent_lower = sent.lower()

        if not positive_re.search(sent_lower):
            continue

        if negative_re.search(sent_lower):
            continue

        outputs.append(sent)

    return outputs


def get_filtered_pet_text(text):
    filtered = extract_pet_uptake_sentences(text)

    if len(filtered) == 0:
        return "Không có uptake bất thường"

    return " ".join(filtered)


# ======================================================
# LOAD PREDICTION
# ======================================================
def load_jsonl_prediction(path):
    ans_dict = {}
    bad_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines.append((line_idx, str(e)))
                continue

            qid = item.get("question_id", None)
            txt = item.get("text", "")

            if qid is None:
                bad_lines.append((line_idx, "missing question_id"))
                continue

            ans_dict[qid] = txt

    return ans_dict, bad_lines


# ======================================================
# LOAD GT
# ======================================================
def load_gt(path):
    with open(path, "r", encoding="utf-8") as f:
        data_desc = json.load(f)

    gt_dict = {}

    for item in data_desc:
        qid = item.get("id", None)
        if qid is None:
            continue

        gt_text = ""

        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                gt_text = conv.get("value", "")
                break

        if gt_text:
            gt_dict[qid] = gt_text

    return gt_dict


# ======================================================
# TOKENIZE
# ======================================================
def safe_word_tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return text.split()


# ======================================================
# BLEU + ROUGE
# ======================================================
def compute_bleu_rouge(pred_text_filtered, gt_text_filtered):
    pred_tokens = safe_word_tokenize(pred_text_filtered)
    gt_tokens = safe_word_tokenize(gt_text_filtered)

    if len(pred_tokens) == 0:
        pred_tokens = [""]

    if len(gt_tokens) == 0:
        gt_tokens = [""]

    bleu = sentence_bleu(
        [gt_tokens],
        pred_tokens,
        smoothing_function=smooth
    )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=False
    )

    rouge_scores = scorer.score(
        gt_text_filtered,
        pred_text_filtered
    )

    rouge1_f = rouge_scores["rouge1"].fmeasure
    rougeL_f = rouge_scores["rougeL"].fmeasure

    return bleu, rouge1_f, rougeL_f


# ======================================================
# RUN ONE MODEL
# ======================================================
def run_one_model(model_name, pred_path, gt_dict):
    print("\n" + "=" * 80)
    print("Running:", model_name)
    print("Prediction:", pred_path)

    if not os.path.exists(pred_path):
        print("SKIP, file not found:", pred_path)
        return None

    ans_dict, bad_lines = load_jsonl_prediction(pred_path)

    output_path = os.path.join(
        OUT_DIR,
        f"{model_name}_pet_uptake_metric.txt"
    )

    matched_items = []

    for qid, gt_text in tqdm(gt_dict.items(), desc=f"Eval {model_name}"):
        if qid not in ans_dict:
            continue

        pred_text = clean_chat_markers(ans_dict[qid])

        gt_text_filtered = get_filtered_pet_text(gt_text)
        pred_text_filtered = get_filtered_pet_text(pred_text)

        bleu, rouge1_f, rougeL_f = compute_bleu_rouge(
            pred_text_filtered,
            gt_text_filtered
        )

        # Tính BERTScore từng sample một, giống code cũ:
        # bert_score.score([pred_text_filtered], [gt_text_filtered], ...)
        P, R, F1 = bert_score.score(
            [pred_text_filtered],
            [gt_text_filtered],
            model_type=BERT_MODEL,
            num_layers=9,
            verbose=False
        )

        bert_f = F1.mean().item()

        matched_items.append({
            "id": qid,
            "gt_filtered": gt_text_filtered,
            "pred_filtered": pred_text_filtered,
            "BLEU": bleu,
            "ROUGE-1 F1": rouge1_f,
            "ROUGE-L F1": rougeL_f,
            "BERTScore F1": bert_f,
        })

    if len(matched_items) == 0:
        print("No matched IDs for:", model_name)
        return None

    metric_keys = [
        "BLEU",
        "ROUGE-1 F1",
        "ROUGE-L F1",
        "BERTScore F1"
    ]

    avg_metrics = {
        key: float(np.mean([x[key] for x in matched_items]))
        for key in metric_keys
    }

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(f"=== {model_name} ===\n")
        out_file.write(f"Prediction path: {pred_path}\n")
        out_file.write(f"GT path: {GT_PATH}\n")
        out_file.write(f"Total predictions loaded: {len(ans_dict)}\n")
        out_file.write(f"Total GT loaded: {len(gt_dict)}\n")
        out_file.write(f"Matched samples: {len(matched_items)}\n")

        if len(bad_lines) > 0:
            out_file.write(f"Bad json lines: {len(bad_lines)}\n")
            for line_idx, err in bad_lines[:50]:
                out_file.write(f"  Line {line_idx}: {err}\n")

            if len(bad_lines) > 50:
                out_file.write(f"  ... and {len(bad_lines) - 50} more\n")

        out_file.write("\n" + "=" * 70 + "\n\n")

        for item in matched_items:
            out_file.write(f"ID: {item['id']}\n")

            out_file.write(
                "Filtered Ground Truth:\n"
                f"{item['gt_filtered']}\n\n"
            )

            out_file.write(
                "Filtered Prediction:\n"
                f"{item['pred_filtered']}\n\n"
            )

            out_file.write("Metrics:\n")

            for key in metric_keys:
                out_file.write(f"  {key}: {item[key]:.4f}\n")

            out_file.write("\n" + "-" * 60 + "\n\n")

        out_file.write("\n=== Average Metrics ===\n")

        for key, value in avg_metrics.items():
            out_file.write(f"{key}: {value:.4f}\n")

        out_file.write("\n" + "=" * 70 + "\n")

    print("Saved:", output_path)
    print("Average:", avg_metrics)

    return {
        "model": model_name,
        "path": pred_path,
        "output": output_path,
        "num_pred": len(ans_dict),
        "num_gt": len(gt_dict),
        "num_matched": len(matched_items),
        "num_bad_lines": len(bad_lines),
        **avg_metrics
    }


# ======================================================
# MAIN
# ======================================================
def main():
    print("Loading GT...")
    gt_dict = load_gt(GT_PATH)
    print("GT loaded:", len(gt_dict))

    all_results = []

    for model_name, pred_path in MODEL_PATHS.items():
        result = run_one_model(
            model_name=model_name,
            pred_path=pred_path,
            gt_dict=gt_dict
        )

        if result is not None:
            all_results.append(result)

    summary_txt = os.path.join(
        OUT_DIR,
        "summary_pet_uptake_metrics.txt"
    )

    summary_json = os.path.join(
        OUT_DIR,
        "summary_pet_uptake_metrics_qwen.json"
    )

    metric_keys = [
        "BLEU",
        "ROUGE-1 F1",
        "ROUGE-L F1",
        "BERTScore F1"
    ]

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Summary PET Uptake Metrics ===\n\n")

        for res in all_results:
            f.write(f"Model: {res['model']}\n")
            f.write(f"Path: {res['path']}\n")
            f.write(f"Output: {res['output']}\n")
            f.write(f"Matched: {res['num_matched']}\n")
            f.write(f"Bad lines: {res['num_bad_lines']}\n")

            for key in metric_keys:
                f.write(f"{key}: {res[key]:.4f}\n")

            f.write("\n" + "-" * 60 + "\n\n")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nDone.")
    print("Summary txt:", summary_txt)
    print("Summary json:", summary_json)


if __name__ == "__main__":
    main()