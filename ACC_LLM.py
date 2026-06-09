import os
import random
import re
import numpy as np
from collections import Counter
from difflib import SequenceMatcher
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ============================================================
# ⭐ 1. 核心配置与预处理 (参考论文去噪标准)
# ============================================================
SPECIAL_TOKENS = "□◇"


def remove_title_line(text):
    """删除卷志类噪声行"""
    lines = text.splitlines()
    pattern = re.compile(r".{0,10}(志|誌|记|記|稿).{0,10}(卷).{0,10}")
    new_lines = []
    removed = False
    for line in lines:
        clean_line = line.strip()
        if not removed and 5 < len(clean_line) < 40 and pattern.search(clean_line):
            removed = True
            continue
        new_lines.append(line)
    return "".join(new_lines)


def clean_text(text):
    """学术标准清洗：去除空白符，保留汉字、数字及占位符"""
    if not text: return ""
    text = re.sub(r"\s+", "", text)
    text = re.sub(rf"[^\u3400-\u9FFF\uF900-\uFAFF0-9{SPECIAL_TOKENS}]", "", text)
    return text


def align_and_filter(pred, gt):
    """对齐并过滤特殊占位符以统一评测基准"""
    matcher = SequenceMatcher(None, pred, gt)
    new_pred, new_gt = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for p, g in zip(pred[i1:i2], gt[j1:j2]):
                if g not in SPECIAL_TOKENS:
                    new_pred.append(p)
                    new_gt.append(g)
        elif tag == "delete":
            for p in pred[i1:i2]:
                new_pred.append(p)
                new_gt.append("")
        elif tag == "insert":
            for g in gt[j1:j2]:
                if g not in SPECIAL_TOKENS:
                    new_pred.append("")
                    new_gt.append(g)
    return "".join(new_pred), "".join(new_gt)


# ============================================================
# ⭐ 2. 距离计算逻辑 (Levenshtein)
# ============================================================
def edit_distance(s1, s2):
    """计算编辑距离"""
    n, m = len(s1), len(s2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m]


# ============================================================
# ⭐ 3. 指标设计 (移除 WER，增加 P, R, BLEU)
# ============================================================
def calculate_metrics(pred_raw, gt_raw):
    p_text = remove_title_line(pred_raw)
    p_clean = clean_text(p_text)
    g_clean = clean_text(gt_raw)

    # 字符级对齐过滤
    cp, ct = align_and_filter(p_clean, g_clean)
    if not ct: return None

    # --- CER (Character Error Rate) ---
    char_ed = edit_distance(cp, ct)
    cer = char_ed / len(ct)

    # --- NED (Normalized Edit Distance) ---
    ned = 1.0 - char_ed / max(len(cp), len(ct), 1)

    # --- Precision / Recall / F1-Score ---
    matcher = SequenceMatcher(None, cp, ct)
    tp = sum(block.size for block in matcher.get_matching_blocks())
    precision = tp / len(cp) if len(cp) > 0 else 0
    recall = tp / len(ct) if len(ct) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- BLEU-4 (参考论文衡量序列准确度) ---
    reference = [list(ct)]
    candidate = list(cp)
    smooth = SmoothingFunction().method1
    # 权重设为 0.25 对应 BLEU-4
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "cer": cer, "ned": ned, "f1": f1,
        "precision": precision, "recall": recall, "bleu4": bleu4
    }


# ============================================================
# ⭐ 4. 主评测流程 (格式化为 0.4567)
# ============================================================
def evaluate_ocr_paper_standard(pred_dir, gt_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    metrics_list = []

    print(f">>> 启动学术级评测 (指标: P, R, F1, CER, NED, BLEU-4)")

    for file_name in tqdm(pred_files, desc="Processing"):
        p_path = os.path.join(pred_dir, file_name)
        g_path = os.path.join(gt_dir, file_name)
        if not os.path.exists(g_path): continue

        with open(p_path, "r", encoding="utf-8") as f:
            pred_text = f.read()
        with open(g_path, "r", encoding="utf-8") as f:
            gt_text = f.read()

        res = calculate_metrics(pred_text, gt_text)
        if res: metrics_list.append(res)

    if not metrics_list: return

    # 计算均值
    avg_cer = np.mean([m["cer"] for m in metrics_list])
    avg_ned = np.mean([m["ned"] for m in metrics_list])
    avg_f1 = np.mean([m["f1"] for m in metrics_list])
    avg_p = np.mean([m["precision"] for m in metrics_list])
    avg_r = np.mean([m["recall"] for m in metrics_list])
    avg_bleu = np.mean([m["bleu4"] for m in metrics_list])

    print("\n" + "=" * 85)
    print(f"{'Metric':<25} | {'Value (0.xxxx)':<15} | {'Description'}")
    print("-" * 85)
    # 按照 0.4567 格式输出 (.4f)
    print(f"{'CER ↓':<25} | {avg_cer:<15.4f} | 字错率")
    print(f"{'NED ↑':<25} | {avg_ned:<15.4f} | 归一化相似度")
    print(f"{'Precision (P) ↑':<25} | {avg_p:<15.4f} | 精确率 (查准率)")
    print(f"{'Recall (R) ↑':<25} | {avg_r:<15.4f} | 召回率 (查全率)")
    print(f"{'F1-Score ↑':<25} | {avg_f1:<15.4f} | 综合 F1 指标")
    print(f"{'BLEU-4 ↑':<25} | {avg_bleu:<15.4f} | 序列一致性 (BLEU)")
    print("-" * 85)


if __name__ == "__main__":
    PRED_RESULT_DIR = "/homes/hwj/code/OCR/test_300/qwen_beam_consultation"
    GT_TEXT_DIR = "/homes/hwj/code/OCR/testdata_300/text"
    evaluate_ocr_paper_standard(PRED_RESULT_DIR, GT_TEXT_DIR)


