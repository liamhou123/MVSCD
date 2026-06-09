import os


def remove_explanations(text):
    """
    规则：
    - 按行处理
    - 只要某一行包含关键词
      → 该行及其后所有行全部删除
    """
    triggers = [
        "注意:",
        "注意：",
        "——以上",
        "按閱讀順序",
        "按阅读顺序",
        "最終輸出",
        "最终输出",
        "*注意*",
        "此處僅提供",
        "此处仅提供",
        "---",
        "*註"
    ]

    lines = text.splitlines()
    kept_lines = []

    for line in lines:
        if any(t in line for t in triggers):
            break
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def batch_clean_txt(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".txt"):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        with open(in_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        clean_text = remove_explanations(raw_text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"✔ 已处理: {fname}")


if __name__ == "__main__":
    INPUT_DIR = "/homes/hwj/code/OCR/test_350_song/gemini"
    OUTPUT_DIR = "/homes/hwj/code/OCR/test_350_song/gemini_clean"

    batch_clean_txt(INPUT_DIR, OUTPUT_DIR)

