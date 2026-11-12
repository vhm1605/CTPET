import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score
import warnings
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

nltk.download('punkt')
smooth = nltk.translate.bleu_score.SmoothingFunction().method4


# ======================================================
# CLEAN CHAT MARKER
# ======================================================
def clean_chat_markers(text: str) -> str:
    text = re.sub(
        r'(?m)^\s*(assistant|user)\s*\n?',
        '',
        text
    )

    text = re.sub(
        r'\n[ \t]*\n+',
        '\n',
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
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    sents = re.split(
        r'(?<=[.!?;])\s+',
        text
    )

    return [s.strip() for s in sents if s.strip()]


def extract_pet_uptake_sentences(text):
    sentences = split_sentences(text)

    outputs = []

    for sent in sentences:
        sent_lower = sent.lower()

        # phải có uptake signal
        if not positive_re.search(sent_lower):
            continue

        # remove physiologic uptake
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
# METRIC
# ======================================================
def compute_metrics(pred_text, gt_text):

    # -------- filter PET uptake --------
    pred_text_filtered = get_filtered_pet_text(pred_text)
    gt_text_filtered = get_filtered_pet_text(gt_text)

    # -------- tokenization --------
    pred_tokens = nltk.word_tokenize(pred_text_filtered)
    gt_tokens = nltk.word_tokenize(gt_text_filtered)

    # -------- BLEU --------
    bleu = sentence_bleu(
        [gt_tokens],
        pred_tokens,
        smoothing_function=smooth
    )

    # -------- ROUGE --------
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rougeL'],
        use_stemmer=False
    )

    rouge_scores = scorer.score(
        gt_text_filtered,
        pred_text_filtered
    )

    rouge1_f = rouge_scores['rouge1'].fmeasure
    rougeL_f = rouge_scores['rougeL'].fmeasure

    # -------- BERTScore --------
    P, R, F1 = bert_score.score(
        [pred_text_filtered],
        [gt_text_filtered],
        model_type='/home/nvidia-lab/ai4life/bert-multilingual',
        num_layers=9,
        verbose=False
    )

    bert_f = F1.mean().item()

    return {
        "BLEU": bleu,
        "ROUGE-1 F1": rouge1_f,
        "ROUGE-L F1": rougeL_f,
        "BERTScore F1": bert_f,
        "Filtered GT": gt_text_filtered,
        "Filtered Pred": pred_text_filtered
    }


# ======================================================
# LOAD PREDICTION
# ======================================================
ans_dict = {}

with open(
    '/data/ychieu/results/adaptive_cosmos_lmed.jsonl',
    'r',
    encoding='utf-8'
) as f:

    for line in f:
        item = json.loads(line)
        ans_dict[item['question_id']] = item['text']


# ======================================================
# LOAD GT
# ======================================================
with open(
    '/data/backup_sde1/PET-CT-report/pretrain_data/single_turn/align_test.json',
    'r',
    encoding='utf-8'
) as f:

    data_desc = json.load(f)


# ======================================================
# SAVE METRIC
# ======================================================
all_metrics = {
    "BLEU": [],
    "ROUGE-1 F1": [],
    "ROUGE-L F1": [],
    "BERTScore F1": []
}

output_path = "cosmos_mistral_uptake_metric_oldest.txt"

print("save to", output_path)

with open(output_path, "a", encoding="utf-8") as out_file:

    out_file.write("\n=== New Run ===\n")

    for item in tqdm(data_desc):

        id_ = item['id']

        if id_ not in ans_dict:
            continue

        pred_text = ans_dict[id_]
        pred_text = clean_chat_markers(pred_text)

        gt_text = ""

        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                gt_text = conv['value']
                break

        if not gt_text:
            continue

        metrics = compute_metrics(
            pred_text,
            gt_text
        )

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        # =====================================
        # SAVE DETAIL
        # =====================================
        out_file.write(f"ID: {id_}\n")

        out_file.write(
            f"Filtered Ground Truth:\n"
            f"{metrics['Filtered GT']}\n\n"
        )

        out_file.write(
            f"Filtered Prediction:\n"
            f"{metrics['Filtered Pred']}\n\n"
        )

        out_file.write("Metrics:\n")

        for key in all_metrics:
            out_file.write(
                f"  {key}: {metrics[key]:.4f}\n"
            )

        out_file.write(
            "\n" + "-" * 60 + "\n\n"
        )

    # ==================================================
    # AVERAGE
    # ==================================================
    avg_metrics = {
        key: np.mean(all_metrics[key])
        for key in all_metrics
    }

    out_file.write(
        "\n=== Average Metrics ===\n"
    )

    for key, value in avg_metrics.items():
        out_file.write(
            f"{key}: {value:.4f}\n"
        )

    out_file.write(
        "\n" + "=" * 70 + "\n"
    )

print(avg_metrics)