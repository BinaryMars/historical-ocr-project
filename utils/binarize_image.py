import cv2
import numpy as np
from scipy.signal import find_peaks
import os
import argparse


def sharpen_image(image):
    """
    对图像进行保留边缘的锐化处理，使用非锐化掩模技术。

    参数:
    image -- 输入的灰度图像（numpy 数组）

    返回:
    sharpened -- 锐化后的图像（numpy 数组）
    """
    # 对图像应用高斯模糊，得到低频部分
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

    # 计算高频部分（原始图像 - 模糊图像）
    high_freq = image.astype(np.float32) - blurred.astype(np.float32)

    # 将高频部分加回原始图像，增强边缘
    sharpened = image.astype(np.float32) + high_freq * 0.3  # 锐化强度因子

    # 限制像素值范围到 [0, 255] 并转换为 uint8
    sharpened = np.clip(sharpened, 0, 255)
    sharpened = sharpened.astype(np.uint8)

    return sharpened


def binarize_image(image):
    """
    对图像进行二值化处理。

    参数:
    image -- 输入的灰度图像（numpy 数组）

    返回:
    thresh_val -- 二值化阈值
    binary -- 二值化后的图像（numpy 数组）
    """

    # 应用CLAHE预处理，增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(2, 2))
    image = clahe.apply(image)

    image = cv2.fastNlMeansDenoising(
        image, h=5, templateWindowSize=10, searchWindowSize=21
    )

    image = sharpen_image(image)

    # 计算灰度直方图，范围为 [0, 256)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel()  # 将直方图展平为一维数组

    # 使用卷积平滑直方图，减少噪声影响
    kernel_size = 5
    smooth_hist = np.convolve(hist, np.ones(kernel_size) / kernel_size, mode="same")

    # 计算直方图的一阶导数，反映灰度变化率
    derivative = np.diff(smooth_hist)  # 一阶差分
    derivative = np.append(derivative, 0)  # 补齐最后一个值，与直方图长度一致

    # 平滑导数图，进一步减少噪声
    kernel_size = 15
    smooth_derivative_hist = np.convolve(
        derivative, np.ones(kernel_size) / kernel_size, mode="same"
    )

    # 在导数图中寻找显著波峰，波峰表示灰度值的聚集区域
    peaks, _ = find_peaks(smooth_derivative_hist, prominence=10)
    if len(peaks) < 2:
        # 如果波峰少于 2 个，使用 Otsu 阈值法进行二值化
        print("找不到两个显著波峰，使用 Otsu 阈值法。")
        thresh_val, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # 获取第二个波峰的位置
        peak2 = peaks[1]

        # 获取第二个波峰对应的导数值
        peak2_value = smooth_derivative_hist[peak2]

        # 设置阈值参考值，为第二个波峰值某个乘积
        threshold_value = peak2_value * 0.05

        # 从第二个波峰向左搜索，找到导数值小于等于 threshold_value 的位置
        threshold_index = peak2
        for i in range(peak2, -1, -1):
            if smooth_derivative_hist[i] <= threshold_value:
                threshold_index = i
                break

        # 将阈值设置为找到的索引对应的灰度值
        thresh_val = threshold_index

        # 使用计算得到的阈值进行二值化
        _, binary = cv2.threshold(image, thresh_val - 8, 255, cv2.THRESH_BINARY)

    return thresh_val, binary


def main():
    """
    主函数，解析命令行参数并调用图像处理逻辑。
    """
    parser = argparse.ArgumentParser(description="图像二值化处理工具（仅限单张图像）")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图像路径（默认在当前目录生成 binary_<input>）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在。")
        return

    if not os.path.isfile(args.input):
        print(f"错误：{args.input} 不是文件，仅支持单张图像处理。")
        return

    input_file = args.input
    output_file = (
        args.output if args.output else f"binary_{os.path.basename(args.input)}"
    )

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"错误：无法读取图像文件 {input_file}。")
        return
    try:
        thresh_val, binary = binarize_image(image)
    except ValueError as e:
        print(f"错误：处理文件 {input_file} 时发生错误 - {e}")
        return
    cv2.imwrite(output_file, binary)
    print(f"二值化图像已保存为：{output_file}，阈值：{thresh_val}")


if __name__ == "__main__":
    main()
