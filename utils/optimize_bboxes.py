import os
import json
import numpy as np
import cv2
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation
import argparse


def load_json(json_path):
    """加载JSON文件，返回边界框列表和原始数据"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bboxes = [
        [int(item["top"]), int(item["left"]), int(item["bottom"]), int(item["right"])]
        for item in data
    ]
    return bboxes, data


def preprocess_image(image_path):
    """加载并二值化图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    image = cv2.bitwise_not(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}，请检查路径或格式！")
    return image


def get_extended_bbox(bbox, image_shape, extension_ratio=0.1):
    """动态比例扩展边界框，根据原始尺寸比例扩展"""
    top, left, bottom, right = bbox
    h, w = bottom - top, right - left
    vertical_extension = int(h * extension_ratio)
    horizontal_extension = int(w * extension_ratio)

    top = max(0, top - vertical_extension)
    left = max(0, left - horizontal_extension)
    bottom = min(image_shape[0], bottom + vertical_extension)
    right = min(image_shape[1], right + horizontal_extension)
    return [top, left, bottom, right]


def get_character_regions(image, bbox):
    """找到框内所有连通区域，并进行形态学膨胀和腐蚀"""
    top, left, bottom, right = bbox
    roi = image[top:bottom, left:right]
    roi_dilated = binary_dilation(roi, structure=np.ones((5, 5))).astype(np.uint8) * 255
    roi_eroded = cv2.erode(roi_dilated, np.ones((3, 3), np.uint8), iterations=1)
    _, labeled = cv2.connectedComponents(roi_eroded, connectivity=8)
    regions = regionprops(labeled)
    return regions, labeled, top, left


def compute_region_probability(region, bbox, labeled, top_offset, left_offset):
    """计算区域属于中心字符的概率"""
    top, left, bottom, right = bbox
    h, w = bottom - top, right - left
    y0, x0, y1, x1 = region.bbox
    y0_global, x0_global = y0 + top_offset, x0 + left_offset
    y1_global, x1_global = y1 + top_offset, x1 + left_offset

    # 计算区域中心与bbox中心的距离
    cy, cx = region.centroid
    cy_global, cx_global = cy + top_offset, cx + left_offset
    center_dist = np.sqrt(
        (cy_global - (top + h / 2)) ** 2 + (cx_global - (left + w / 2)) ** 2
    )
    dist_weight = 1 - (center_dist / max(h, w))

    # 计算区域与初始bbox的重叠比例
    overlap_top = max(y0_global, top)
    overlap_left = max(x0_global, left)
    overlap_bottom = min(y1_global, bottom)
    overlap_right = min(x1_global, right)
    overlap_area = max(0, overlap_bottom - overlap_top) * max(
        0, overlap_right - overlap_left
    )
    overlap_ratio = overlap_area / region.area if region.area > 0 else 0
    return 0.6 * dist_weight + 0.4 * overlap_ratio


def regions_intersect_bbox(regions, bbox, top_offset, left_offset):
    """找到与初始边界框相交的连通区域"""
    intersecting_regions = []
    for region in regions:
        y0, x0, y1, x1 = region.bbox
        y0_global, x0_global = y0 + top_offset, x0 + left_offset
        y1_global, x1_global = y1 + top_offset, x1 + left_offset
        if not (
            x1_global < bbox[1]
            or x0_global > bbox[3]
            or y1_global < bbox[0]
            or y0_global > bbox[2]
        ):
            intersecting_regions.append(region)
    return intersecting_regions


def compute_merged_bbox(intersecting_regions, top_offset, left_offset):
    """计算相交区域的合并边界框"""
    if not intersecting_regions:
        return None
    min_top = min(region.bbox[0] for region in intersecting_regions) + top_offset
    min_left = min(region.bbox[1] for region in intersecting_regions) + left_offset
    max_bottom = max(region.bbox[2] for region in intersecting_regions) + top_offset
    max_right = max(region.bbox[3] for region in intersecting_regions) + left_offset
    return [min_top, min_left, max_bottom, max_right]


def refine_bbox(image, bbox, extension=10):
    """优化单个边界框，扩展以包含相交的连通区域"""
    extended_bbox = get_extended_bbox(bbox, image.shape, extension)
    regions, labeled, top_offset, left_offset = get_character_regions(
        image, extended_bbox
    )
    if not regions:
        return bbox
    intersecting_regions = regions_intersect_bbox(
        regions, bbox, top_offset, left_offset
    )
    if not intersecting_regions:
        return bbox
    probabilities = [
        compute_region_probability(region, bbox, labeled, top_offset, left_offset)
        for region in intersecting_regions
    ]
    valid_regions = [
        region
        for region, prob in zip(intersecting_regions, probabilities)
        if prob > 0.75
        and region.centroid[0] + top_offset >= bbox[0]
        and region.centroid[0] + top_offset <= bbox[2]
        and region.centroid[1] + left_offset >= bbox[1]
        and region.centroid[1] + left_offset <= bbox[3]
    ]
    if not valid_regions:
        return bbox
    new_bbox = compute_merged_bbox(valid_regions, top_offset, left_offset)
    return new_bbox if new_bbox else bbox


def optimize_bboxes(image, initial_bboxes, extension=None):
    """优化所有边界框"""
    if extension is None:
        avg_height = np.mean([bbox[2] - bbox[0] for bbox in initial_bboxes])
        avg_width = np.mean([bbox[3] - bbox[1] for bbox in initial_bboxes])
        extension = int(min(avg_height, avg_width) * 0.1)
    return [refine_bbox(image, bbox, extension) for bbox in initial_bboxes]


def save_json(output_path, bboxes, original_data):
    """保存优化后的边界框到JSON"""
    output_data = []
    for i, bbox in enumerate(bboxes):
        item = original_data[i].copy()
        item["top"], item["left"], item["bottom"], item["right"] = map(int, bbox)
        output_data.append(item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def visualize_bboxes(image_path, initial_bboxes, optimized_bboxes, output_image_path):
    """绘制优化前后的边界框对比图"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    for bbox in initial_bboxes:
        cv2.rectangle(
            image,
            (bbox[1], bbox[0]),
            (bbox[3], bbox[2]),
            color=(0, 0, 255),
            thickness=1,
        )
    for bbox in optimized_bboxes:
        cv2.rectangle(
            image,
            (bbox[1], bbox[0]),
            (bbox[3], bbox[2]),
            color=(0, 255, 0),
            thickness=1,
        )
    cv2.imwrite(output_image_path, image)


def main():
    parser = argparse.ArgumentParser(description="优化边界框")
    parser.add_argument("--image_path", required=True, help="输入图像路径")
    parser.add_argument("--input_json", required=True, help="输入JSON文件路径")
    parser.add_argument("--output_json", help="输出JSON文件路径")
    parser.add_argument(
        "--output_image", default="bbox_comparison.png", help="输出对比图路径"
    )
    args = parser.parse_args()

    try:
        invert_image = cv2.bitwise_not(
            cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        )
        initial_bboxes, original_data = load_json(args.input_json)
        optimized_bboxes = optimize_bboxes(invert_image, initial_bboxes)

        output_path = args.output_json if args.output_json else args.input_json
        save_json(output_path, optimized_bboxes, original_data)
        print(f"优化后的边界框已保存至 {output_path}")

        visualize_bboxes(
            args.image_path, initial_bboxes, optimized_bboxes, args.output_image
        )
        print(f"可视化对比图已保存至 {args.output_image}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
