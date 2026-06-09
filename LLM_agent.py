import json
import os
import pickle
import re
import gc
import math
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from Bio import pairwise2
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
)
from qwen_vl_utils import process_vision_info


# ================================
# 全局配置
# ================================

class PipelineConfig:
    # 路径
    INPUT_FOLDER = "/homes/hwj/code/OCR/testdata_300/image_patch"
    CHAR_IMG_FOLDER = "/homes/hwj/code/OCR/testdata_300/image_patch_char"
    OUTPUT_ROOT = "/homes/hwj/code/OCR/test_300/qwen_beam_consultation"

    # 模型
    VLM_PATH = "/homes/hwj/model/Qwen3-VL-8B-Instruct"
    LLM_PATH = "/homes/hwj/model/Xunzi-Qwen3-8B/Xunzi-Qwen3-8B"
    CLIP_MODEL_PATH = "/homes/hwj/code/OCR/chinese-clip-vit-huge-patch14"

    # 解码参数
    BEAM_WIDTH = 5
    LM_TEMPERATURE = 2.0
    #W_TRANS = 0.1                     # 转移得分权重

    # 自适应采样
    UNCERTAINTY_SAMPLES = 1
    UNCERTAINTY_TEMP = 0.9
    N_MIN = 10
    N_MAX = 15
    T_MIN = 0.8
    T_MAX = 1.2
    UNCERTAINTY_SCALE = 6.0

    # 自适应权重（基于熵）
    ENTROPY_HIGH_THRESH = 1.0
    ENTROPY_LOW_THRESH = 0.5
    W_CLIP_HIGH_ENTROPY = 0.3
    W_LM_HIGH_ENTROPY = 0.5
    W_CLIP_LOW_ENTROPY = 0.5
    W_LM_LOW_ENTROPY = 0.3
    W_SELF_FIXED = 0.2               # 自洽性投票固定权重

    # 跨模态一致性惩罚
    CONSISTENCY_PENALTY = 0.5

    # 消融开关（启用分数：'self', 'clip', 'lm', 'trans'）
    ENABLED_SCORES = ['self','clip', 'lm']
    USE_ADAPTIVE_WEIGHTS = True
    USE_CONSISTENCY_PENALTY = True

    # 固定权重（当自适应关闭时使用）
    FIXED_WEIGHTS = {
        'self': 0.2,
        'clip': 0.4,
        'lm': 0.4,
        #'trans': 0.1,
    }

    STEP2_CACHE_DIR = os.path.join(OUTPUT_ROOT, "step2_cache")

    # 提示词
    PROMPTS = [{
        "text": "你是一位专门研究明清地方志的史学家，熟悉古籍文献。请精准辨识图中县志页面的文字，并按原样转录。仅输出识别出的汉字序列，严禁任何解释或分析。",
        "temp": 0.9,
        "samples": 10
    }]

    @classmethod
    def get_subfolder(cls, name):
        path = os.path.join(cls.OUTPUT_ROOT, name)
        os.makedirs(path, exist_ok=True)
        return path


# ================================
# 工具函数
# ================================

def clean_text(text):
    return re.sub(r"[^\u4e00-\u9fa5]", "", text)

def clear_memory(model=None, tokenizer=None):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

def parse_lattice_file(lattice_path):
    """从格点文件恢复 col_candidates 列表"""
    with open(lattice_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    col_candidates = []
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == '{':
            end = content.find('}', i)
            block = content[i+1:end]
            cand_dict = {}
            for item in block.split('|'):
                if ':' in item:
                    c, prob = item.split(':')
                    cand_dict[c] = float(prob)
                else:
                    cand_dict[item] = 1.0
            col_candidates.append(cand_dict)
            i = end + 1
        else:
            col_candidates.append({ch: 1.0})
            i += 1
    return col_candidates

# ================================
# Step 1: 候选生成（自适应采样 + 对齐 + Lattice）
# ================================

def run_step1_generate_candidates(vlm, processor, seq_folder, lattice_folder):
    print(">>> Step 1: 候选生成 (自适应动态采样 - 支持断点续跑)")

    entropy_prob_folder = os.path.join(PipelineConfig.OUTPUT_ROOT, "entropy_probs")
    os.makedirs(entropy_prob_folder, exist_ok=True)

    PROB_CERTAIN_THRESH = 0.999

    def compute_entropies_and_certainty(logits_list):
        entropies = []
        certain_mask = []
        for step_logits in logits_list:
            logits = step_logits[0]
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            top1_prob = torch.max(probs).item()
            is_certain = top1_prob > PROB_CERTAIN_THRESH
            entropies.append(entropy)
            certain_mask.append(is_certain)
        return entropies, certain_mask

    lattice_candidates = {}
    alignment_matrices = {}

    all_images = sorted(os.listdir(PipelineConfig.INPUT_FOLDER))
    for img in tqdm(all_images, desc="OCR Sampling"):
        name = os.path.splitext(img)[0]
        seq_path = os.path.join(seq_folder, f"{name}.txt")
        lat_path = os.path.join(lattice_folder, f"{name}.txt")
        matrix_path = os.path.join(lattice_folder, f"{name}_matrix.pkl")

        # ========== 断点续跑：如果已有序列文件和格点文件，则跳过VLM生成 ==========
        if os.path.exists(seq_path) and os.path.exists(lat_path):
            # 读取已有的序列和格点
            with open(seq_path, "r", encoding="utf-8") as f:
                all_seqs = [line.strip() for line in f if line.strip()]
            col_candidates = parse_lattice_file(lat_path)   # 需要实现 parse_lattice_file
            # 重新构建对齐矩阵（无需重新生成候选）
            if all_seqs and col_candidates:
                # 选择最常见的长度作为参考
                lens = Counter(len(s) for s in all_seqs)
                l_best = lens.most_common(1)[0][0]
                ref_seq = next(s for s in all_seqs if len(s) == l_best)

                matrix = []
                for s in all_seqs:
                    if not s:
                        continue
                    aln = pairwise2.align.globalms(ref_seq, s, 2, -1, -2, -1)
                    if aln:
                        a_ref, a_s = aln[0][0], aln[0][1]
                        mapped = [sc for rc, sc in zip(a_ref, a_s) if rc != '-']
                        if len(mapped) == l_best:
                            matrix.append(mapped)

                if matrix:
                    # 保存矩阵
                    with open(matrix_path, 'wb') as f:
                        pickle.dump(matrix, f)
                    alignment_matrices[name] = matrix
                    lattice_candidates[name] = col_candidates
                    continue   # 跳过后续的 VLM 生成步骤
                else:
                    print(f"警告: {name} 对齐失败，将重新生成")
            else:
                print(f"警告: {name} 序列或格点为空，将重新生成")

        # ========== 完整生成流程（VLM调用） ==========
        path = os.path.join(PipelineConfig.INPUT_FOLDER, img)
        messages = [{"role": "user", "content": [{"type": "image", "image": path},
                                                 {"type": "text", "text": PipelineConfig.PROMPTS[0]["text"]}]}]
        image_inputs, _ = process_vision_info(messages)
        prompt_tmpl = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_tmpl], images=image_inputs, return_tensors="pt").to(vlm.device)

        # 不确定性估计
        with torch.no_grad():
            outputs_unc = vlm.generate(
                **inputs,
                do_sample=True,
                temperature=PipelineConfig.UNCERTAINTY_TEMP,
                num_return_sequences=PipelineConfig.UNCERTAINTY_SAMPLES,
                max_new_tokens=128,
                output_scores=True,
                return_dict_in_generate=True
            )
        first_seq = clean_text(processor.decode(outputs_unc.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

        # 保存概率分布（可选）
        step_probs = []
        generated_ids = outputs_unc.sequences[0][inputs.input_ids.shape[1]:]
        scores = outputs_unc.scores
        for step_idx, step_logits in enumerate(scores):
            logits = step_logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            top5_indices = np.argsort(probs)[-5:][::-1]
            step_info = {
                "step": step_idx,
                "generated_token_id": generated_ids[step_idx].item(),
                "generated_char": processor.decode([generated_ids[step_idx]], skip_special_tokens=True),
                "top5": []
            }
            for idx in top5_indices:
                char = processor.decode([idx], skip_special_tokens=True)
                if char and char[0] not in ['<', ' '] and re.search(r'[\u4e00-\u9fa5]', char):
                    step_info["top5"].append({"char": char, "prob": float(probs[idx])})
            step_probs.append(step_info)
        with open(os.path.join(entropy_prob_folder, f"{name}_entropy_probs.json"), "w", encoding="utf-8") as f:
            json.dump(step_probs, f, ensure_ascii=False, indent=2)

        # 计算平均熵（仅不确定步骤）
        entropies, certain_mask = compute_entropies_and_certainty(outputs_unc.scores)
        uncertain_entropies = [e for e, is_certain in zip(entropies, certain_mask) if not is_certain]
        if uncertain_entropies:
            avg_entropy = np.mean(uncertain_entropies)
            MAX_ENTROPY = 2
            norm_entropy = min(1.0, avg_entropy / MAX_ENTROPY)
        else:
            norm_entropy = 0.0

        N = int(PipelineConfig.N_MIN + norm_entropy * (PipelineConfig.N_MAX - PipelineConfig.N_MIN))
        T = PipelineConfig.T_MIN + norm_entropy * (PipelineConfig.T_MAX - PipelineConfig.T_MIN)
        N = max(1, N)

        # 正式生成
        with torch.no_grad():
            outputs_final = vlm.generate(
                **inputs,
                do_sample=True,
                temperature=T,
                num_return_sequences=N,
                max_new_tokens=128,
                output_scores=False
            )
        all_seqs = []
        for out in outputs_final:
            decoded = processor.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            cleaned = clean_text(decoded)
            if cleaned:
                all_seqs.append(cleaned)
        if first_seq and first_seq not in all_seqs:
            all_seqs.append(first_seq)

        # 保存序列文件
        with open(seq_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_seqs))

        # 对齐并构建 Lattice
        if not all_seqs:
            continue
        lens = Counter(len(s) for s in all_seqs)
        l_best = lens.most_common(1)[0][0]
        ref_seq = next(s for s in all_seqs if len(s) == l_best)

        matrix = []
        for s in all_seqs:
            if not s:
                continue
            aln = pairwise2.align.globalms(ref_seq, s, 2, -1, -2, -1)
            if aln:
                a_ref, a_s = aln[0][0], aln[0][1]
                mapped = [sc for rc, sc in zip(a_ref, a_s) if rc != '-']
                if len(mapped) == l_best:
                    matrix.append(mapped)

        if not matrix:
            continue

        lattice_data = []
        col_candidates = []
        for i, col in enumerate(zip(*matrix)):
            chars = [c for c in col if c != '-']
            if not chars:
                continue
            cnt = Counter(chars)
            total = sum(cnt.values())
            candidates = dict(cnt.most_common())
            col_candidates.append(candidates)
            if len(candidates) <= 1:
                lattice_data.append(list(candidates.keys())[0])
            else:
                c_str = "|".join([f"{c}:{cnt/total:.4f}" for c, cnt in candidates.items()])
                lattice_data.append("{" + c_str + "}")

        with open(lat_path, "w") as f:
            f.write("".join(lattice_data))

        # 保存矩阵
        with open(matrix_path, 'wb') as f:
            pickle.dump(matrix, f)

        lattice_candidates[name] = col_candidates
        alignment_matrices[name] = matrix

    return lattice_candidates, alignment_matrices

# ================================
# Step 2: 多源评分（CLIP + 自洽性投票 + LLM）
# ================================

def run_step2_scoring(vlm, processor, lattice_candidates, alignment_matrices, cache_dir):
    print(">>> Step 2: 多源评分 (CLIP + 自洽性投票 + LLM)")

    os.makedirs(cache_dir, exist_ok=True)

    # # ---------- CLIP 模型（全局加载一次）----------
    # clip_model = CLIPModel.from_pretrained(PipelineConfig.CLIP_MODEL_PATH)
    # clip_processor = CLIPProcessor.from_pretrained(PipelineConfig.CLIP_MODEL_PATH)
    # clip_model = clip_model.to(vlm.device)


    # ✅ 修改后
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    clip_model = ChineseCLIPModel.from_pretrained(PipelineConfig.CLIP_MODEL_PATH)
    clip_processor = ChineseCLIPProcessor.from_pretrained(PipelineConfig.CLIP_MODEL_PATH)

    clip_model = clip_model.to(vlm.device)
    clip_model.eval()

    # ---------- LLM 模型（全局加载一次）----------
    llm = AutoModelForCausalLM.from_pretrained(
        PipelineConfig.LLM_PATH, device_map="auto",
        quantization_config=get_quant_config(), trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(PipelineConfig.LLM_PATH, trust_remote_code=True)

    # 定义辅助函数（保持不变）
    def raw_log_prob(text):
        inputs = tokenizer(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            outputs = llm(**inputs, labels=inputs.input_ids)
        log_probs = torch.log_softmax(outputs.logits[:, :-1], dim=-1)
        labels = inputs.input_ids[:, 1:]
        token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return token_lp.sum().item()

    def trans_prob(prev, curr):
        text = f"{prev}{curr}"
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(llm.device)
        if inputs.input_ids.shape[1] < 2:
            return -5.0
        with torch.no_grad():
            outputs = llm(**inputs, labels=inputs.input_ids)
        log_probs = torch.log_softmax(outputs.logits[:, :-1], dim=-1)
        curr_id = inputs.input_ids[0, 1]
        return log_probs[0, 0, curr_id].item()

    # 结果容器
    clip_scores_db = {}
    self_consistency_db = {}
    lm_unary_db = {}
    lm_trans_db = {}

    # 遍历所有图片
    for name, cols in tqdm(lattice_candidates.items(), desc="Step2 Processing"):
        cache_file = os.path.join(cache_dir, f"{name}.pkl")
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                if os.path.getsize(cache_file) == 0:
                    raise EOFError("Empty cache file")

                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)

                clip_scores_db[name] = cached['clip']
                self_consistency_db[name] = cached['self']
                lm_unary_db[name] = cached['unary']
                lm_trans_db[name] = cached['trans']
                continue

            except Exception as e:
                print(f"[Cache Error] {name} cache损坏，重新计算: {e}")
                os.remove(cache_file)

        # ---------- 处理该图片 ----------
        # 1. CLIP 评分
        file_clip_scores = []
        for i, cand_map in enumerate(cols):
            candidates = list(cand_map.keys())
            if len(candidates) <= 1:
                file_clip_scores.append({candidates[0]: 0.0})
                continue
            img_paths = []
            for offset in (-1, 0, 1):
                idx = i + offset
                if idx < 0:
                    continue
                p = os.path.join(PipelineConfig.CHAR_IMG_FOLDER, f"{name}_{idx:02d}.png")
                if os.path.exists(p):
                    img_paths.append(p)
            if not img_paths:
                scores = {c: math.log(1.0/len(candidates)) for c in candidates}
                file_clip_scores.append(scores)
                continue
            images = [Image.open(p).convert("RGB") for p in img_paths]
            texts = [f"汉字：{c}" for c in candidates]
            img_inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(clip_model.device)
            txt_inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(clip_model.device)
            with torch.no_grad():
                # img_feat = clip_model.get_image_features(img_inputs.pixel_values)
                # txt_feat = clip_model.get_text_features(txt_inputs.input_ids, txt_inputs.attention_mask)
                outputs = clip_model(
                    pixel_values=img_inputs["pixel_values"],
                    input_ids=txt_inputs["input_ids"],
                    attention_mask=txt_inputs["attention_mask"]
                )

                img_feat = outputs.image_embeds
                txt_feat = outputs.text_embeds

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = torch.matmul(img_feat, txt_feat.T).cpu().numpy()
            best_sims = np.max(sim, axis=0)
            logits = best_sims / 1.0
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            scores = {c: math.log(p+1e-8) for c, p in zip(candidates, probs)}
            file_clip_scores.append(scores)

        # 2. 自洽性投票
        matrix = alignment_matrices.get(name, [])
        file_self_scores = []
        if matrix:
            n_pos = len(matrix[0])
            for i in range(n_pos):
                chars = [seq[i] for seq in matrix if i < len(seq)]
                cnt = Counter(chars)
                total = sum(cnt.values())
                scores = {c: math.log(cnt/total + 1e-8) for c, cnt in cnt.items()}
                file_self_scores.append(scores)
        else:
            file_self_scores = []

        # 3. LLM 一元 + 二元得分
        file_unary = []
        file_trans = []
        if cols:
            anchor = [list(c.keys())[0] for c in cols]
            for i, cand in enumerate(cols):
                cand_list = list(cand.keys())
                # 一元
                raw = {}
                for c in cand_list:
                    start = max(0, i-5)
                    end = min(len(cols), i+4)
                    ctx = "".join(anchor[start:i]) + c + "".join(anchor[i+1:end])
                    raw[c] = raw_log_prob(ctx)
                arr = np.array(list(raw.values()))
                probs = np.exp((arr - np.max(arr)) / PipelineConfig.LM_TEMPERATURE)
                file_unary.append(dict(zip(cand_list, np.log(probs/np.sum(probs)+1e-8))))
                # 二元
                trans = {}
                if i > 0:
                    prev_cands = list(cols[i-1].keys())
                    for pc in prev_cands:
                        for cc in cand_list:
                            trans[(pc, cc)] = trans_prob(pc, cc)
                file_trans.append(trans)
        else:
            file_unary = []
            file_trans = []

        # 保存到缓存
        cached = {
            'clip': file_clip_scores,
            'self': file_self_scores,
            'unary': file_unary,
            'trans': file_trans
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached, f)

        # 添加到结果字典
        clip_scores_db[name] = file_clip_scores
        self_consistency_db[name] = file_self_scores
        lm_unary_db[name] = file_unary
        lm_trans_db[name] = file_trans

    # 释放 CLIP 和 VLM（VLM 已在外层释放）
    clear_memory(clip_model)
    # 注意：LLM 不释放，因为 Step3 可能还需要（虽然当前未使用重排序，但保留）
    return clip_scores_db, self_consistency_db, lm_unary_db, lm_trans_db, llm, tokenizer

# ================================
# Step 3: 可配置的 CRF 融合解码
# ================================

def run_step3_fusion(lattice_candidates, clip_scores_db, self_consistency_db,
                     lm_unary_db, lm_trans_db, final_folder):
    print(">>> Step 3: 多视角自洽性增强解码 (可配置评分)")

    enabled = PipelineConfig.ENABLED_SCORES          # ['self','clip','lm','trans']
    use_adaptive = PipelineConfig.USE_ADAPTIVE_WEIGHTS
    use_penalty = PipelineConfig.USE_CONSISTENCY_PENALTY
    fixed_weights = PipelineConfig.FIXED_WEIGHTS

    for name, cols in tqdm(lattice_candidates.items(), desc="Fusion"):
        clip_scores = clip_scores_db.get(name, [])
        self_scores = self_consistency_db.get(name, [])
        unary_scores = lm_unary_db.get(name, [])
        trans_scores = lm_trans_db.get(name, [])

        if not cols or not unary_scores:
            with open(os.path.join(final_folder, f"{name}.txt"), "w") as f:
                f.write("")
            continue

        # 计算每个位置的熵（基于候选频率）
        entropy_list = []
        for cands in cols:
            probs = list(cands.values())
            total = sum(probs)
            if total > 0:
                p_norm = [p/total for p in probs]
                entropy = -sum(p * math.log(p+1e-8) for p in p_norm)
            else:
                entropy = 0.0
            entropy_list.append(entropy)

        def adaptive_weights(entropy):
            if entropy > PipelineConfig.ENTROPY_HIGH_THRESH:
                return {
                    'self': PipelineConfig.W_SELF_FIXED,
                    'clip': PipelineConfig.W_CLIP_HIGH_ENTROPY,
                    'lm': PipelineConfig.W_LM_HIGH_ENTROPY,
                }
            elif entropy < PipelineConfig.ENTROPY_LOW_THRESH:
                return {
                    'self': PipelineConfig.W_SELF_FIXED,
                    'clip': PipelineConfig.W_CLIP_LOW_ENTROPY,
                    'lm': PipelineConfig.W_LM_LOW_ENTROPY,
                }
            else:
                return {'self': 0.2, 'clip': 0.4, 'lm': 0.4}

        beams = [("", 0.0)]
        for i, cands in enumerate(cols):
            if use_adaptive:
                weights = adaptive_weights(entropy_list[i])
            else:
                weights = {k: fixed_weights.get(k, 0.0) for k in enabled if k != 'trans'}

            next_beams = []
            for char, freq in cands.items():
                # 各项得分
                score_self = self_scores[i].get(char, -10.0) if i < len(self_scores) and 'self' in enabled else 0.0
                score_clip = clip_scores[i].get(char, -10.0) if i < len(clip_scores) and 'clip' in enabled else 0.0
                score_lm   = unary_scores[i].get(char, -10.0) if i < len(unary_scores) and 'lm' in enabled else 0.0

                unary_total = 0.0
                if 'self' in enabled:
                    unary_total += weights.get('self', 0.0) * score_self
                if 'clip' in enabled:
                    unary_total += weights.get('clip', 0.0) * score_clip
                if 'lm' in enabled:
                    unary_total += weights.get('lm', 0.0) * score_lm

                # 一致性惩罚（clip 与 lm 差异）
                if use_penalty and 'clip' in enabled and 'lm' in enabled:
                    s_clip = clip_scores[i].get(char, -10.0) if i < len(clip_scores) else -10.0
                    s_lm   = unary_scores[i].get(char, -10.0) if i < len(unary_scores) else -10.0
                    unary_total -= PipelineConfig.CONSISTENCY_PENALTY * abs(s_clip - s_lm)

                for path, score in beams:
                    trans = 0.0
                    if 'trans' in enabled and path:
                        prev = path[-1]
                        trans = trans_scores[i].get((prev, char), -15.0) if i < len(trans_scores) else 0.0
                    new_score = score + unary_total + (PipelineConfig.W_TRANS * trans if 'trans' in enabled else 0.0)
                    next_beams.append((path + char, new_score))

            beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:PipelineConfig.BEAM_WIDTH]

        best = beams[0][0] if beams else ""
        with open(os.path.join(final_folder, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(best)

# ================================
# 主程序
# ================================

if __name__ == "__main__":
    s1_p = PipelineConfig.get_subfolder("step1_sequences")
    s2_p = PipelineConfig.get_subfolder("step2_lattice")
    s3_p = PipelineConfig.get_subfolder("step3_final")
    step2_cache_dir = PipelineConfig.STEP2_CACHE_DIR
    os.makedirs(step2_cache_dir, exist_ok=True)

    # Step1（内部已支持断点续跑）
    v_model = AutoModelForVision2Seq.from_pretrained(
        PipelineConfig.VLM_PATH, device_map="auto",
        quantization_config=get_quant_config(), trust_remote_code=True
    )
    v_proc = AutoProcessor.from_pretrained(PipelineConfig.VLM_PATH, trust_remote_code=True)
    lattice_candidates, alignment_matrices = run_step1_generate_candidates(v_model, v_proc, s1_p, s2_p)
    clear_memory(v_model, v_proc)

    # Step2（支持断点续跑，通过图片级缓存）
    clip_scores, self_scores, l_unary, l_trans, llm, tokenizer = run_step2_scoring(
        v_model, v_proc, lattice_candidates, alignment_matrices, step2_cache_dir
    )

    # Step3
    run_step3_fusion(lattice_candidates, clip_scores, self_scores, l_unary, l_trans, s3_p)

    clear_memory(llm, tokenizer)
    print(">>> MVSCD OCR 系统运行完毕。")