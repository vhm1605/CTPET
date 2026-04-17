import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import warnings
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def remove_assistant_marker_loose(text: str) -> str:
#     # thay mọi dòng chỉ có 'assistant' (có thể có spaces) thành 1 newline
#     return re.sub(r"\n\s*assistant\s*\n", "", text)

def clean_chat_markers(text: str) -> str:
    text = re.sub(r'(?m)^\s*(assistant|user)\s*\n?', '', text)
    text = re.sub(r'\n[ \t]*\n+', '', text)

    # 3) Xóa newline/space thừa đầu-cuối
    return text.strip()

# Tắt cảnh báo beta/bias nếu muốn
# warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`")
nltk.download('punkt_tab')
smooth = SmoothingFunction().method4

# HÀM TÍNH METRIC
def compute_metrics(pred_text, gt_text):
    pred_tokens = nltk.word_tokenize(pred_text)
    gt_tokens = nltk.word_tokenize(gt_text)

    bleu = sentence_bleu([gt_tokens], pred_tokens)
    # bleu = sentence_bleu([gt_tokens], pred_tokens,  weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    rouge_scores = scorer.score(gt_text, pred_text)
    rouge1_f = rouge_scores['rouge1'].fmeasure
    rougeL_f = rouge_scores['rougeL'].fmeasure

    P, R, F1 = bert_score.score([pred_text], [gt_text], model_type='/home/nvidia-lab/ai4life/kienpt/bert-multilingual', num_layers=9, verbose=False)
    bert_f = F1.mean().item()

    return {
        "BLEU": bleu,
        "ROUGE-1 F1": rouge1_f,
        "ROUGE-L F1": rougeL_f,
        "BERTScore F1": bert_f
    }

# Đọc file ans_conv_val_filtered.jsonl

ans_dict = {}
with open('/home/nvidia-lab/ai4life/CTPET/ctvit_llavamed/infer/lora_region_resume/checkpoint-11108/test/answer.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        ans_dict[item['question_id']] = item['text']

# Đọc file data_desc_conv_val_filtered.json
with open('/data/backup_sde1/PET-CT-report/pretrain_data/single_turn/align_test.json', 'r', encoding='utf-8') as f:
# with open('/data/backup_sde1/PET-CT-report/pretrain_data/single_turn/petct/ct/align_test.json', 'r', encoding='utf-8') as f:

# with open('/workdir/radish/PET-CT/PET-CT-report/pretrain_data/single_turn/petct/ct/align_test.json', 'r', encoding='utf-8') as f:
    data_desc = json.load(f)

# Biến lưu các metric
all_metrics = {
    "BLEU": [],
    "ROUGE-1 F1": [],
    "ROUGE-L F1": [],
    "BERTScore F1": []
}

output_path = "petct_ctvit_llavamed_region_resume.txt"
# output_path = "ct_ctvit_qwen_instruct_lora_ver2_ckpt_13435_new_compare.txt"
print('save to', output_path)
with open(output_path, "a", encoding="utf-8") as out_file:
    out_file.write("\n=== New Run ===\n")
    
    for item in tqdm(data_desc):
        id_ = item['id']
        if id_ in ans_dict:
            pred_text = ans_dict[id_]
            pred_text = clean_chat_markers(pred_text)
            gt_text = ""
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    gt_text = conv['value']
                    break
            if gt_text:
                metrics = compute_metrics(pred_text, gt_text)
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                # Ghi từng cặp vào file (append)
                out_file.write(f"ID: {id_}\n")
                out_file.write(f"Ground Truth: {gt_text}\n")
                out_file.write(f"Prediction: {pred_text}\n")
                out_file.write(f"Metrics:\n")
                for key, value in metrics.items():
                    out_file.write(f"  {key}: {value:.4f}\n")
                out_file.write("\n" + "-"*50 + "\n\n")

    # Tính và ghi trung bình các metric
    avg_metrics = {key: np.mean(all_metrics[key]) for key in all_metrics}

    out_file.write("\n=== Average Metrics for This Run ===\n")
    for key, value in avg_metrics.items():
        out_file.write(f"{key}: {value:.4f}\n")
    out_file.write("\n" + "="*70 + "\n")
