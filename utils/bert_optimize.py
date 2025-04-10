import json
from typing import List, Dict
from transformers import BertForMaskedLM, BertTokenizer
import torch
import argparse
import time

# 自动检测可用设备
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# 加载BERT模型和分词器
model = BertForMaskedLM.from_pretrained("SIKU-BERT/sikubert")
tokenizer = BertTokenizer.from_pretrained("SIKU-BERT/sikubert")

# 仅在 CUDA 设备上使用 float16（MPS 不支持）
if DEVICE == "cuda":
    model = model.half()

model.to(DEVICE)
model.eval()


def get_initial_sequence(
    predictions: List[Dict], confidence_threshold: float = 0.8
) -> List[str]:
    """生成初始序列，低置信度用[UNK]占位"""
    sequence = []
    for pred in predictions:
        top_candidate, top_prob = pred["top_k_predictions"][0]
        if top_prob < confidence_threshold:
            sequence.append("[UNK]")
        else:
            sequence.append(top_candidate)
    return sequence


def bert_score_cnn_candidates(
    sequence: List[str],
    position: int,
    cnn_candidates: List[tuple],
    context_window: int = None,
) -> Dict[str, float]:
    """用BERT为CNN的top-k候选项评分"""
    if context_window is not None:
        start = max(0, position - context_window)
        end = min(len(sequence), position + context_window + 1)
        context_sequence = sequence[start:end]
        mask_pos_in_context = position - start
    else:
        context_sequence = sequence.copy()
        mask_pos_in_context = position

    context_sequence[mask_pos_in_context] = "[MASK]"
    text = "".join(context_sequence)

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    mask_token_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, mask_token_index]
    probs = torch.softmax(logits, dim=-1)

    bert_scores = {}
    for char, _ in cnn_candidates:
        token_id = tokenizer.convert_tokens_to_ids(char)
        bert_scores[char] = probs[token_id].item()
    return bert_scores


def process_predictions(
    predictions: List[Dict],
    confidence_threshold: float = 0.8,
    context_window: int = 5,
    lambda_weight: float = 0.5,
    verbose: bool = True,
) -> List[str]:
    """处理CNN预测结果，低置信度位置用BERT优化，verbose控制输出"""
    sequence = get_initial_sequence(predictions, confidence_threshold)
    if verbose:
        print(f"初始序列（低置信度用[UNK]）: {''.join(sequence)}")

    start_time = time.time()
    changes = []  # 记录变动
    for pos, pred in enumerate(predictions):
        top_candidate, top_prob = pred["top_k_predictions"][0]
        if top_prob < confidence_threshold:
            # if verbose:
            #     print(
            #         f"\n位置 {pos} 置信度 {top_prob:.4f} 低于阈值，用BERT评分CNN候选项"
            #     )
            cnn_candidates = pred["top_k_predictions"]
            bert_scores = bert_score_cnn_candidates(
                sequence, pos, cnn_candidates, context_window
            )

            cnn_dict = dict(cnn_candidates)
            best_candidate, best_score = None, -1
            # if verbose:
            #     print(
            #         f"{'候选词':<8} {'CNN概率':<10} {'BERT概率':<10} {'综合得分':<10}"
            #     )
            #     print("-" * 40)
            for char in cnn_dict:
                cnn_prob = cnn_dict[char]
                bert_prob = bert_scores.get(char, 1e-10)
                score = lambda_weight * cnn_prob + (1 - lambda_weight) * bert_prob
                # if verbose:
                #     print(
                #         f"{char:<8} {cnn_prob:<10.4f} {bert_prob:<10.4f} {score:<10.4f}"
                #     )
                if score > best_score:
                    best_candidate = char
                    best_score = score

            if best_candidate:
                old_char = sequence[pos]
                sequence[pos] = best_candidate
                if best_candidate != top_candidate:
                    # if verbose:
                    #     print(
                    #         f"位置 {pos}: {top_candidate} -> {best_candidate} (综合得分: {best_score:.4f})"
                    #     )
                    changes.append(
                        {
                            "position": pos,
                            "old_char": top_candidate,
                            "new_char": best_candidate,
                            "score": best_score,
                        }
                    )
                # elif verbose:
                #     print(
                #         f"位置 {pos}: 未变化，仍为 {best_candidate} (综合得分: {best_score:.4f})"
                #     )

    end_time = time.time()
    if verbose:
        print(f"\n总处理耗时: {end_time - start_time:.2f}秒")
        print(f"最终序列: {''.join(sequence)}")
        if changes:
            print("\n发生变化的字符位置:")
            for change in changes:
                print(
                    f"位置 {change['position']}: {change['old_char']} -> {change['new_char']} (综合得分: {change['score']:.4f})"
                )
        else:
            print("\n没有发生变化。")

    return sequence


def process_json(
    input_file: str,
    output_file: str = None,
    confidence_threshold: float = 0.8,
    context_window: int = 5,
    lambda_weight: float = 0.5,
    verbose: bool = True,
):
    """处理JSON文件并保存结果"""
    with open(input_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    predictions.sort(key=lambda x: x["order"])
    if verbose:
        print(f"开始处理，总字符数: {len(predictions)}")

    final_sequence = process_predictions(
        predictions, confidence_threshold, context_window, lambda_weight, verbose
    )

    for i, pred in enumerate(predictions):
        pred["final_char"] = final_sequence[i]

    output_file = output_file or input_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    if verbose:
        print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR with BERT scoring CNN top-k")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    parser.add_argument("--threshold", type=float, default=0.9, help="置信度阈值")
    parser.add_argument(
        "--context_window",
        type=int,
        default=-1,
        help="上下文窗口大小（-1表示完整序列）",
    )
    parser.add_argument(
        "--lambda_weight", type=float, default=0.5, help="CNN和BERT权重"
    )
    args = parser.parse_args()

    context_window = args.context_window if args.context_window != -1 else None
    process_json(
        args.input,
        args.output,
        args.threshold,
        context_window,
        args.lambda_weight,
        verbose=True,
    )
