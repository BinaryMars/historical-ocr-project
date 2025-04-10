import json
import argparse
from typing import List, Dict


def add_center_points(bboxes: List[Dict]) -> List[Dict]:
    """为每个边界框添加中心点信息"""
    for box in bboxes:
        box["center_x"] = (box["left"] + box["right"]) / 2
        box["center_y"] = (box["top"] + box["bottom"]) / 2
    return bboxes


def calculate_dynamic_thresholds(bboxes: List[Dict]) -> tuple:
    """计算动态阈值：平均宽度和高度"""
    if not bboxes:
        return 10, 10  # 默认值
    avg_width = sum(b["right"] - b["left"] for b in bboxes) / len(bboxes)
    avg_height = sum(b["bottom"] - b["top"] for b in bboxes) / len(bboxes)
    return avg_width * 0.1, avg_height * 0.2  # 接近性阈值：0.1倍宽度，0.2倍高度


def group_main_columns(
    bboxes: List[Dict], threshold_x: float, direction: str = "right_to_left"
) -> List[List[Dict]]:
    """按中心点x坐标将字符分组为主列，支持从左到右或从右到左排序"""
    if direction == "right_to_left":
        bboxes.sort(key=lambda b: -b["center_x"])  # 从右到左按中心点x排序
    else:
        bboxes.sort(key=lambda b: b["center_x"])  # 从左到右按中心点x排序

    columns = []
    current_column = [bboxes[0]]

    for i in range(1, len(bboxes)):
        prev_box = bboxes[i - 1]
        curr_box = bboxes[i]
        if abs(prev_box["center_x"] - curr_box["center_x"]) < threshold_x:  # 同列判断
            current_column.append(curr_box)
        else:
            columns.append(current_column)
            current_column = [curr_box]
    columns.append(current_column)

    return columns


def line_detection(
    curr_box: Dict, target_box: Dict, direction: str, threshold_y: float
) -> bool:
    """线检测：判断两个框是否在指定方向上邻近（基于中心点）"""
    if direction == "up":
        return (
            abs(curr_box["center_y"] - target_box["center_y"]) < threshold_y
            and curr_box["center_x"] - target_box["center_x"] < threshold_y
            and curr_box["center_x"] - target_box["center_x"] > -threshold_y
        )
    elif direction == "down":
        return (
            abs(curr_box["center_y"] - target_box["center_y"]) < threshold_y
            and curr_box["center_x"] - target_box["center_x"] < threshold_y
            and curr_box["center_x"] - target_box["center_x"] > -threshold_y
        )
    return False


def point_detection(curr_box: Dict, target_box: Dict, threshold_x: float) -> bool:
    """点检测：判断两个框是否在水平方向上对齐（基于中心点）"""
    return (
        abs(curr_box["center_y"] - target_box["center_y"]) < threshold_x
        and abs(curr_box["center_x"] - target_box["center_x"]) < threshold_x * 2
    )


def sort_column_boxes(
    column: List[Dict], threshold_x: float, threshold_y: float
) -> List[Dict]:
    """对单个主列进行四向扫描排序，使用索引避免哈希问题"""
    # 为每个框分配索引
    column_with_idx = [(i, box) for i, box in enumerate(column)]
    column.sort(key=lambda b: b["top"])  # 按y1初步排序
    top_right_idx = max(
        range(len(column)), key=lambda i: column[i]["right"] - column[i]["top"]
    )
    bottom_left_idx = min(
        range(len(column)), key=lambda i: column[i]["left"] + column[i]["bottom"]
    )

    # 使用索引集合
    down1_boxes = set()  # Downscan #1: 正文和右注释
    down2_boxes = set()  # Downscan #2: 左注释
    up2_boxes = set()  # Upscan #2: 正文和左注释

    # Downscan #1: 从top_right向下扫描
    current_idx = top_right_idx
    down1_boxes.add(current_idx)
    while True:
        next_candidates = [
            i
            for i, b in column_with_idx
            if i not in down1_boxes
            and line_detection(column[current_idx], b, "down", threshold_y)
        ]
        if not next_candidates:
            break
        current_idx = min(next_candidates, key=lambda i: column[i]["top"])
        down1_boxes.add(current_idx)
        # 检查右注释
        right_candidates = [
            i
            for i, b in column_with_idx
            if i not in down1_boxes
            and point_detection(column[current_idx], b, threshold_x)
        ]
        if right_candidates:
            rightmost_idx = max(right_candidates, key=lambda i: column[i]["right"])
            down1_boxes.add(rightmost_idx)

    # Downscan #2: 从top_right向下扫描左注释
    current_idx = top_right_idx
    while True:
        next_candidates = [
            i
            for i, b in column_with_idx
            if i not in down1_boxes
            and i not in down2_boxes
            and line_detection(column[current_idx], b, "down", threshold_y)
        ]
        if not next_candidates:
            break
        current_idx = min(next_candidates, key=lambda i: column[i]["top"])
        down2_boxes.add(current_idx)

    # Upscan #2: 从bottom_left向上扫描
    current_idx = bottom_left_idx
    up2_boxes.add(current_idx)
    while True:
        next_candidates = [
            i
            for i, b in column_with_idx
            if i not in up2_boxes
            and line_detection(column[current_idx], b, "up", threshold_y)
        ]
        if not next_candidates:
            break
        current_idx = max(next_candidates, key=lambda i: column[i]["bottom"])
        up2_boxes.add(current_idx)

    # 分类：正文(B)、右注释(RA)、左注释(LA)
    body = down1_boxes & up2_boxes  # 正文是DOWN1和UP2的交集
    right_anno = down1_boxes - up2_boxes  # 右注释仅在DOWN1中
    left_anno = up2_boxes - down1_boxes  # 左注释仅在UP2中

    # 子列分割和排序
    sorted_boxes = []
    sub_column = []
    prev_type = None

    all_boxes = [
        (i, "B" if i in body else "RA" if i in right_anno else "LA")
        for i in range(len(column))
    ]
    all_boxes.sort(key=lambda x: column[x[0]]["top"])  # 按y1排序

    for idx, box_type in all_boxes:
        if prev_type and prev_type != box_type:
            sorted_boxes.extend(
                [column[i] for i in sorted(sub_column, key=lambda i: column[i]["top"])]
            )
            sub_column = []
        sub_column.append(idx)
        prev_type = box_type
    sorted_boxes.extend(
        [column[i] for i in sorted(sub_column, key=lambda i: column[i]["top"])]
    )

    return sorted_boxes


def sort_characters(bboxes: List[Dict]) -> List[Dict]:
    """主函数：对所有边界框进行排序"""
    bboxes = add_center_points(bboxes)  # 添加中心点信息
    threshold_x, threshold_y = calculate_dynamic_thresholds(bboxes)
    main_columns = group_main_columns(bboxes, threshold_x)

    sorted_boxes = []
    for column in main_columns:
        sorted_column = sort_column_boxes(column, threshold_x, threshold_y)
        sorted_boxes.extend(sorted_column)

    # 添加顺序字段
    for i, box in enumerate(sorted_boxes):
        box["order"] = i

    return sorted_boxes


def main():
    parser = argparse.ArgumentParser(
        description="为OCR边界框添加排序，适用于中文历史文档"
    )
    parser.add_argument("--json_path", required=True, help="输入JSON文件路径")
    parser.add_argument(
        "--output_path", default=None, help="输出JSON文件路径（默认覆盖原文件）"
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        bboxes = json.load(f)

    sorted_bboxes = sort_characters(bboxes)

    output_path = args.output_path if args.output_path else args.json_path
    with open(output_path, "w") as f:
        json.dump(sorted_bboxes, f, indent=4, ensure_ascii=False)

    print(f"Sorted bounding boxes saved to: {output_path}")


if __name__ == "__main__":
    main()
