import os
import re
import json
import csv
import warnings

import nltk
import numpy as np

from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score
from scipy.stats import wilcoxon


# ============================================================
# CONFIG
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

GT_PATH = (
    "/data/backup_sde1/PET-CT-report/"
    "pretrain_data/single_turn/align_test.json"
)

ANSWER_PATHS = {
    # Proposed / SUV threshold
    "Mistral_SUV": (
        "/home/nvidia-lab/ai4life/CTPET_new_filter/"
        "ctvit_llavamed/infer/test_region_new_suv_selection_final/"
        "checkpoint-22216/answer.jsonl"
    ),

    # Mistral only PET
    "Mistral_OnlyPET": (
        "/data/ychieu/results/noise_ctvit_med.jsonl"
    ),

    # Original Mistral
    "Mistral": (
        "/data/ychieu1/checkpoint-19502/test/answers.jsonl"
    ),
}

BERT_MODEL = "/home/nvidia-lab/ai4life/bert-multilingual"
BERT_NUM_LAYERS = 9
BERT_BATCH_SIZE = 16

OUTPUT_TXT = "mistral_suv_statistical_comparison.txt"
OUTPUT_CSV = "mistral_suv_per_sample_metrics.csv"

# One-sided paired Wilcoxon test for the directional hypothesis:
# H1: AnatoPET_SUV > baseline.
# Since x = proposed method and y = baseline below, use "greater".
ALTERNATIVE = "greater"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_chat_markers(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    # Remove lines such as:
    # assistant
    # user
    # **assistant**
    # **user**
    text = re.sub(
        r"(?im)^\s*\**\s*(assistant|user)\s*\**\s*$",
        "",
        text,
    )

    # Collapse excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()


# ============================================================
# LOAD ANSWERS
# ============================================================

def get_answer_id(item):
    """
    Support common JSONL formats.
    """
    for key in ["question_id", "id", "sample_id"]:
        if key in item:
            return str(item[key])

    raise KeyError(
        f"Cannot find ID key. Available keys: {list(item.keys())}"
    )


def get_answer_text(item):
    """
    Support common answer keys.
    """
    for key in [
        "text",
        "answer",
        "prediction",
        "pred",
        "response",
        "output",
    ]:
        if key in item:
            return clean_chat_markers(item[key])

    raise KeyError(
        f"Cannot find answer text key. Available keys: {list(item.keys())}"
    )


def load_answers(path):
    answers = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                id_ = get_answer_id(item)
                text = get_answer_text(item)

                answers[id_] = text

            except Exception as e:
                print(
                    f"[WARNING] {path}, line {line_num}: {e}"
                )

    return answers


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_dict = {}

    for item in data:
        id_ = str(item["id"])

        gt_text = None

        for conv in item["conversations"]:
            if conv["from"] == "gpt":
                gt_text = clean_chat_markers(conv["value"])
                break

        if gt_text:
            gt_dict[id_] = gt_text

    return gt_dict


# ============================================================
# BLEU + ROUGE
# ============================================================

rouge_evaluator = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"],
    use_stemmer=False,
)


def compute_lexical_metrics(pred_text, gt_text):
    pred_tokens = nltk.word_tokenize(pred_text)
    gt_tokens = nltk.word_tokenize(gt_text)

    # Keep same BLEU definition as your original code
    bleu = sentence_bleu(
        [gt_tokens],
        pred_tokens,
    )

    rouge_scores = rouge_evaluator.score(
        gt_text,
        pred_text,
    )

    return {
        "BLEU": bleu,
        "ROUGE-1 F1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-L F1": rouge_scores["rougeL"].fmeasure,
    }


# ============================================================
# STATISTICAL TEST
# ============================================================

def paired_wilcoxon(x, y, alternative="two-sided"):
    """
    x = proposed method
    y = baseline

    Returns:
        statistic
        p-value
        mean difference
        median difference
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y

    # scipy wilcoxon can fail if every difference == 0
    if np.allclose(diff, 0):
        statistic = 0.0
        pvalue = 1.0
    else:
        result = wilcoxon(
            x,
            y,
            alternative=alternative,
            zero_method="wilcox",
        )

        statistic = float(result.statistic)
        pvalue = float(result.pvalue)

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def significance_symbol(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_correction(pvalues):
    """
    Holm-Bonferroni correction without statsmodels.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    adjusted_sorted = np.zeros(n)

    running_max = 0.0

    for i, p in enumerate(sorted_p):
        adjusted = (n - i) * p
        adjusted = min(adjusted, 1.0)

        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = running_max

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return adjusted


# ============================================================
# MAIN
# ============================================================

def main():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("=" * 80)
    print("Loading ground truth...")
    gt_dict = load_ground_truth(GT_PATH)

    print(f"Ground truth samples: {len(gt_dict)}")

    # --------------------------------------------------------
    # Load all model outputs
    # --------------------------------------------------------

    answer_dicts = {}

    for model_name, path in ANSWER_PATHS.items():
        print(f"\nLoading {model_name}")
        print(f"  {path}")

        answer_dicts[model_name] = load_answers(path)

        print(
            f"  Number of predictions: "
            f"{len(answer_dicts[model_name])}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only compare IDs available for EVERY method.
    # --------------------------------------------------------

    common_ids = set(gt_dict.keys())

    for model_name in ANSWER_PATHS:
        common_ids &= set(answer_dicts[model_name].keys())

    common_ids = sorted(common_ids)

    print("\n" + "=" * 80)
    print(f"COMMON PAIRED SAMPLES = {len(common_ids)}")
    print("=" * 80)

    if len(common_ids) == 0:
        raise RuntimeError(
            "No common IDs found between GT and the 3 answer files."
        )

    # --------------------------------------------------------
    # Initialize metrics
    # --------------------------------------------------------

    metric_names = [
        "BLEU",
        "ROUGE-1 F1",
        "ROUGE-L F1",
        "BERTScore F1",
    ]

    all_metrics = {
        model: {
            metric: []
            for metric in metric_names
        }
        for model in ANSWER_PATHS
    }

    # --------------------------------------------------------
    # BLEU / ROUGE
    # --------------------------------------------------------

    print("\nComputing BLEU / ROUGE...")

    for id_ in tqdm(common_ids):

        gt = gt_dict[id_]

        for model_name in ANSWER_PATHS:
            pred = answer_dicts[model_name][id_]

            result = compute_lexical_metrics(
                pred,
                gt,
            )

            for key, value in result.items():
                all_metrics[model_name][key].append(value)

    # --------------------------------------------------------
    # BERTScore
    #
    # MUCH faster than calling bert_score.score once per sample
    # --------------------------------------------------------

    print("\nComputing BERTScore...")

    refs = [
        gt_dict[id_]
        for id_ in common_ids
    ]

    for model_name in ANSWER_PATHS:

        print(f"\nBERTScore: {model_name}")

        preds = [
            answer_dicts[model_name][id_]
            for id_ in common_ids
        ]

        P, R, F1 = bert_score.score(
            preds,
            refs,
            model_type=BERT_MODEL,
            num_layers=BERT_NUM_LAYERS,
            batch_size=BERT_BATCH_SIZE,
            verbose=True,
        )

        all_metrics[model_name]["BERTScore F1"] = (
            F1.cpu().numpy().tolist()
        )

    # --------------------------------------------------------
    # Average metrics
    # --------------------------------------------------------

    averages = {}

    print("\n")
    print("=" * 80)
    print("AVERAGE RESULTS")
    print("=" * 80)

    for model_name in ANSWER_PATHS:

        averages[model_name] = {}

        print(f"\n{model_name}")

        for metric in metric_names:

            values = np.asarray(
                all_metrics[model_name][metric]
            )

            mean = np.mean(values)
            std = np.std(values, ddof=1)

            averages[model_name][metric] = mean

            print(
                f"  {metric:20s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

    # --------------------------------------------------------
    # Statistical comparisons
    #
    # SUV vs OnlyPET
    # SUV vs original Mistral
    # --------------------------------------------------------

    comparisons = [
        ("Mistral_SUV", "Mistral_OnlyPET"),
        ("Mistral_SUV", "Mistral"),
    ]

    statistical_results = []

    print("\n")
    print("=" * 80)
    print(f"PAIRED WILCOXON TEST ({ALTERNATIVE})")
    print("=" * 80)

    for proposed, baseline in comparisons:

        print(
            f"\n{proposed} vs {baseline}"
        )

        for metric in metric_names:

            x = all_metrics[proposed][metric]
            y = all_metrics[baseline][metric]

            stat = paired_wilcoxon(
                x,
                y,
                alternative=ALTERNATIVE,
            )

            row = {
                "proposed": proposed,
                "baseline": baseline,
                "metric": metric,
                **stat,
            }

            statistical_results.append(row)

            print(
                f"  {metric:20s} | "
                f"Δmean = {stat['mean_diff']:+.6f} | "
                f"p = {stat['pvalue']:.6g} "
                f"{significance_symbol(stat['pvalue'])}"
            )

    # --------------------------------------------------------
    # Multiple comparison correction
    # 2 baselines x 4 metrics = 8 hypothesis tests
    # --------------------------------------------------------

    raw_pvalues = [
        x["pvalue"]
        for x in statistical_results
    ]

    corrected_pvalues = holm_correction(
        raw_pvalues
    )

    for row, corrected in zip(
        statistical_results,
        corrected_pvalues,
    ):
        row["holm_pvalue"] = float(corrected)

    print("\n")
    print("=" * 80)
    print("HOLM-CORRECTED P-VALUES")
    print("=" * 80)

    for row in statistical_results:
        print(
            f"{row['proposed']} vs {row['baseline']} | "
            f"{row['metric']:20s} | "
            f"raw p={row['pvalue']:.6g} | "
            f"Holm p={row['holm_pvalue']:.6g} "
            f"{significance_symbol(row['holm_pvalue'])}"
        )

    # --------------------------------------------------------
    # Save per-sample CSV
    # --------------------------------------------------------

    print(f"\nSaving per-sample metrics to {OUTPUT_CSV}")

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        header = ["id"]

        for model_name in ANSWER_PATHS:
            for metric in metric_names:
                header.append(
                    f"{model_name}_{metric}"
                )

        writer.writerow(header)

        for i, id_ in enumerate(common_ids):

            row = [id_]

            for model_name in ANSWER_PATHS:
                for metric in metric_names:
                    row.append(
                        all_metrics[model_name][metric][i]
                    )

            writer.writerow(row)

    # --------------------------------------------------------
    # Save summary TXT
    # --------------------------------------------------------

    print(f"Saving summary to {OUTPUT_TXT}")

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Statistical comparison of Mistral SUV threshold\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Number of paired samples: {len(common_ids)}\n"
        )
        f.write(
            f"Statistical test: paired Wilcoxon signed-rank "
            f"({ALTERNATIVE})\n\n"
        )

        # Means
        f.write("AVERAGE METRICS\n")
        f.write("-" * 80 + "\n")

        for model_name in ANSWER_PATHS:

            f.write(f"\n{model_name}\n")

            for metric in metric_names:

                values = np.asarray(
                    all_metrics[model_name][metric]
                )

                mean = np.mean(values)
                std = np.std(values, ddof=1)

                f.write(
                    f"{metric}: "
                    f"{mean:.6f} ± {std:.6f}\n"
                )

        # P-values
        f.write("\n\n")
        f.write("PAIRWISE STATISTICAL TESTS\n")
        f.write("-" * 80 + "\n")

        for row in statistical_results:

            f.write(
                f"\n{row['proposed']} vs "
                f"{row['baseline']}\n"
            )

            f.write(
                f"Metric: {row['metric']}\n"
            )

            f.write(
                f"Mean difference: "
                f"{row['mean_diff']:+.8f}\n"
            )

            f.write(
                f"Median difference: "
                f"{row['median_diff']:+.8f}\n"
            )

            f.write(
                f"Wilcoxon statistic: "
                f"{row['statistic']:.8f}\n"
            )

            f.write(
                f"Raw p-value: "
                f"{row['pvalue']:.10g}\n"
            )

            f.write(
                f"Holm-corrected p-value: "
                f"{row['holm_pvalue']:.10g}\n"
            )

            f.write(
                f"Significance (raw): "
                f"{significance_symbol(row['pvalue'])}\n"
            )

            f.write(
                f"Significance (Holm): "
                f"{significance_symbol(row['holm_pvalue'])}\n"
            )

    print("\nDone.")
    print(f"Summary : {OUTPUT_TXT}")
    print(f"Details : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()