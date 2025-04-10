import os
import json
import cv2
import torch
import tensorflow as tf
from utils.binarize_image import binarize_image
from utils.hrcenternet import detect_characters
from utils.optimize_bboxes import optimize_bboxes
from utils.sort_bboxes import sort_characters
from utils.predict_character import build_model, predict_character, get_w2i_dict
from utils.bert_optimize import process_json
import traceback


# 默认路径常量
DEFAULT_CNN_WEIGHTS_PATH = "models/cnn/best.weights.h5"
DEFAULT_CHAR_DICT_PATH = "models/cnn/w2i.json"
DEFAULT_HRCENTERNET_WEIGHTS_PATH = "models/hrcenternet/HRCenterNet.pth.tar"
DEFAULT_OUTPUT_JSON_PATH = "static/results/processed_ocr_result.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERMEDIATE_DIR = "static/results"  # 中间结果保存目录


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2, label=None):
    """在图像上绘制边界框，可选添加标签"""
    img_copy = image.copy()
    if len(img_copy.shape) == 2:  # 灰度图转RGB
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    for bbox in bboxes:
        top, left, bottom, right = (
            int(bbox["top"]),
            int(bbox["left"]),
            int(bbox["bottom"]),
            int(bbox["right"]),
        )
        cv2.rectangle(img_copy, (left, top), (right, bottom), color, thickness)
        if label and "order" in bbox:
            cv2.putText(
                img_copy,
                str(bbox["order"]),
                (left - 30, top + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
    return img_copy


def process_image_to_ocr(
    input_image_path: str,
    output_json_path: str = DEFAULT_OUTPUT_JSON_PATH,
    cnn_weights_path: str = DEFAULT_CNN_WEIGHTS_PATH,
    char_dict_path: str = DEFAULT_CHAR_DICT_PATH,
    hrcenternet_weights_path: str = DEFAULT_HRCENTERNET_WEIGHTS_PATH,
    nms_score_threshold: float = 0.3,
    iou_threshold: float = 0.1,
    top_k_predictions: int = 5,
):
    # 创建中间结果目录
    ensure_dir(INTERMEDIATE_DIR)
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # 定义统一的 JSON 文件名
    uniform_json_name = f"4.{base_name}_ocr_result.json"
    output_json_path = os.path.join(INTERMEDIATE_DIR, uniform_json_name)

    # 1. 验证输入文件是否存在
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"输入图像 {input_image_path} 不存在")

    # 2. 加载图像并进行二值化
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"无法读取图像 {input_image_path}")
    binarization_threshold, binary_image = binarize_image(original_image)
    binary_image_path = os.path.join(INTERMEDIATE_DIR, f"0.{base_name}_binary.png")
    cv2.imwrite(binary_image_path, binary_image)
    print(f"二值化完成，阈值: {binarization_threshold}，保存至: {binary_image_path}")

    # 3. 使用HRCenterNet检测边界框
    initial_bboxes = detect_characters(
        binary_image_path,
        model_path=hrcenternet_weights_path,
        nms_score=nms_score_threshold,
        iou_threshold=iou_threshold,
    )
    print(f"HRCenterNet检测到 {len(initial_bboxes)} 个初始边界框")
    # 保存初始边界框图像
    initial_bbox_image = draw_bboxes(binary_image, initial_bboxes, color=(235, 206, 87))
    initial_bbox_path = os.path.join(
        INTERMEDIATE_DIR, f"1.{base_name}_initialBboxes.png"
    )
    cv2.imwrite(initial_bbox_path, initial_bbox_image)
    print(f"初始边界框图像保存至: {initial_bbox_path}")

    # 4. 优化边界框
    initial_bboxes_list = [
        [b["top"], b["left"], b["bottom"], b["right"]] for b in initial_bboxes
    ]
    refined_bboxes_list = optimize_bboxes(
        cv2.bitwise_not(binary_image),
        initial_bboxes_list,
        0.15,
    )
    refined_bboxes = []
    for i, bbox in enumerate(refined_bboxes_list):
        bbox_dict = initial_bboxes[i].copy()
        bbox_dict["top"], bbox_dict["left"], bbox_dict["bottom"], bbox_dict["right"] = (
            map(int, bbox)
        )
        refined_bboxes.append(bbox_dict)
    print(f"边界框优化完成，优化后数量: {len(refined_bboxes)}")
    # 保存优化后的边界框图像
    refined_bbox_image = draw_bboxes(
        binary_image, refined_bboxes, color=(193, 182, 255)
    )
    refined_bbox_path = os.path.join(
        INTERMEDIATE_DIR, f"2.{base_name}_refinedBboxes.png"
    )
    cv2.imwrite(refined_bbox_path, refined_bbox_image)
    print(f"优化后边界框图像保存至: {refined_bbox_path}")

    # 5. 对边界框进行排序
    sorted_bboxes = sort_characters(refined_bboxes)
    print(f"边界框排序完成，总数: {len(sorted_bboxes)}")
    # 保存排序后的边界框图像（带编号）
    sorted_bbox_image = draw_bboxes(
        # original_image,
        binary_image,
        sorted_bboxes,
        color=(140, 206, 32),
        label=True,
    )
    sorted_bbox_path = os.path.join(INTERMEDIATE_DIR, f"3.{base_name}_sortedBboxes.png")
    cv2.imwrite(sorted_bbox_path, sorted_bbox_image)
    print(f"排序后边界框图像保存至: {sorted_bbox_path}")

    # 6. 加载字符预测所需的模型和字典
    w2i_dict, i2w_dict = get_w2i_dict(char_dict_path)
    cnn_model = build_model(num_classes=len(w2i_dict))
    cnn_model.load_weights(cnn_weights_path)
    print(f"加载CNN模型和字典完成，字符类别数: {len(w2i_dict)}")

    # 7. 对每个字符进行预测并添加结果
    char_dir = os.path.join(INTERMEDIATE_DIR, "characters")
    ensure_dir(char_dir)
    for bbox in sorted_bboxes:
        top, left, bottom, right = (
            int(bbox["top"]),
            int(bbox["left"]),
            int(bbox["bottom"]),
            int(bbox["right"]),
        )
        char_image = binary_image[top:bottom, left:right]
        char_image_path = os.path.join(char_dir, f"char_{bbox['order']}.png")
        cv2.imwrite(char_image_path, char_image)

        predicted_char, top_k_results = predict_character(
            char_image_path, cnn_model, w2i_dict, i2w_dict, top_k=top_k_predictions
        )
        bbox["predicted_char"] = predicted_char
        bbox["top_k_predictions"] = top_k_results
        # print(
        #     f"字符 {bbox['order']} 预测完成: {predicted_char}，单字图像保存至: {char_image_path}"
        # )

    # 8. 保存结果到JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_bboxes, f, indent=4, ensure_ascii=False)
    print(f"CNN初步识别完成，结果保存至: {output_json_path}")

    # 9. 调用 bert_optimize.py 用BERT优化结果
    process_json(
        input_file=output_json_path,
        output_file=output_json_path,  # 覆盖原文件
        confidence_threshold=0.9,  # 可调整
        context_window=5,  # 可调整
        lambda_weight=0.45,  # 可调整
        verbose=True,  # False 为安静模式，只返回结果不打印过程
    )
    print(f"BERT优化完成，最终结果保存至: {output_json_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OCR处理脚本，集成图像处理和字符预测")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_JSON_PATH, help="输出JSON文件路径"
    )
    parser.add_argument(
        "--cnn_weights",
        type=str,
        default=DEFAULT_CNN_WEIGHTS_PATH,
        help="CNN模型权重路径",
    )
    parser.add_argument(
        "--char_dict", type=str, default=DEFAULT_CHAR_DICT_PATH, help="字符字典路径"
    )
    parser.add_argument(
        "--hrcenternet_weights",
        type=str,
        default=DEFAULT_HRCENTERNET_WEIGHTS_PATH,
        help="HRCenterNet模型权重路径",
    )
    parser.add_argument("--nms_score", type=float, default=0.3, help="NMS置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="NMS IoU阈值")
    parser.add_argument("--top_k", type=int, default=5, help="返回前K个字符预测结果")
    args = parser.parse_args()

    try:
        process_image_to_ocr(
            input_image_path=args.input,
            output_json_path=args.output,
            cnn_weights_path=args.cnn_weights,
            char_dict_path=args.char_dict,
            hrcenternet_weights_path=args.hrcenternet_weights,
            nms_score_threshold=args.nms_score,
            iou_threshold=args.iou_threshold,
            top_k_predictions=args.top_k,
        )
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
