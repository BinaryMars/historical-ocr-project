"""
HRCenterNet目标检测脚本 - 单图版本(带命令行参数)

此脚本处理单张图片，输出带边界框的图片到当前目录，并生成边界框位置的JSON文件。
"""

import os
import json
import argparse
from typing import List, Tuple
import torch
from torch import nn
import torchvision
from torchvision.ops import nms
import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import rectangle_perimeter

# 配置设置
INPUT_SIZE = 512  # 输入图像尺寸
OUTPUT_SIZE = 128  # 输出特征图尺寸
# 选择合适的设备
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():  # Mac 上的 Metal 支持
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# 模型组件定义
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        bn_momentum: float = 0.1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        bn_momentum: float = 0.1,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class StageModule(nn.Module):
    def __init__(self, stage: int, output_branches: int, c: int, bn_momentum: float):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2**i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(
                                c * (2**j), c * (2**i), kernel_size=1, bias=False
                            ),
                            nn.BatchNorm2d(c * (2**i), momentum=bn_momentum),
                            nn.Upsample(scale_factor=(2.0 ** (j - i)), mode="nearest"),
                        )
                    )
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    c * (2**j),
                                    c * (2**j),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(c * (2**j), momentum=bn_momentum),
                                nn.ReLU(inplace=True),
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(
                                c * (2**j),
                                c * (2**i),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(c * (2**i), momentum=bn_momentum),
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
        return [self.relu(xf) for xf in x_fused]


class HRCenterNet(nn.Module):
    def __init__(self, c: int = 32, nof_joints: int = 5, bn_momentum: float = 0.1):
        super(HRCenterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample, bn_momentum=bn_momentum),
            Bottleneck(256, 64, bn_momentum=bn_momentum),
            Bottleneck(256, 64, bn_momentum=bn_momentum),
            Bottleneck(256, 64, bn_momentum=bn_momentum),
        )

        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(c, momentum=bn_momentum),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            256, c * 2, kernel_size=3, stride=2, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(c * 2, momentum=bn_momentum),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum)
        )
        self.transition2 = nn.ModuleList(
            [
                nn.Sequential(),
                nn.Sequential(),
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(c * 4, momentum=bn_momentum),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )
        self.transition3 = nn.ModuleList(
            [
                nn.Sequential(),
                nn.Sequential(),
                nn.Sequential(),
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            c * 4, c * 8, kernel_size=3, stride=2, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(c * 8, momentum=bn_momentum),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=1),
            nn.BatchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, nof_joints, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]),
        ]
        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]
        x = self.stage4(x)
        return self.final_layer(x[0])


# 图像预处理
test_tx = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        torchvision.transforms.ToTensor(),
    ]
)


def apply_nms(
    img: Image.Image,
    predict: torch.Tensor,
    nms_score: float = 0.3,
    iou_threshold: float = 0.1,
) -> Tuple[np.ndarray, List[dict]]:
    """
    对预测结果应用NMS并绘制边界框，返回图像和边界框信息

    Args:
        img: 输入图像
        predict: 模型预测结果
        nms_score: 置信度阈值
        iou_threshold: IoU阈值

    Returns:
        Tuple[np.ndarray, List[dict]]: 带边界框的图像和边界框信息列表
    """
    bbox = []
    score_list = []
    bbox_info = []
    im_draw = np.asarray(
        torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))
    ).copy()

    heatmap = predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]

    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:
        row = j // OUTPUT_SIZE
        col = j - row * OUTPUT_SIZE
        bias_x = offset_x[row, col] * (img.size[1] / OUTPUT_SIZE)
        bias_y = offset_y[row, col] * (img.size[0] / OUTPUT_SIZE)
        width = width_map[row, col] * OUTPUT_SIZE * (img.size[1] / OUTPUT_SIZE)
        height = height_map[row, col] * OUTPUT_SIZE * (img.size[0] / OUTPUT_SIZE)
        score = heatmap[row, col]

        row = row * (img.size[1] / OUTPUT_SIZE) + bias_y
        col = col * (img.size[0] / OUTPUT_SIZE) + bias_x
        top = int(row - width / 2)
        left = int(col - height / 2)
        bottom = int(row + width / 2)
        right = int(col + height / 2)

        bbox.append([top, left, bottom, right])
        score_list.append(score)
        bbox_info.append(
            {
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "score": round(float(score), 2),
            }
        )

    nms_index = nms(
        torch.FloatTensor(bbox), torch.FloatTensor(score_list), iou_threshold
    )

    filtered_bbox_info = [bbox_info[i] for i in nms_index]

    for k in nms_index:
        top, left, bottom, right = bbox[k]
        rr, cc = rectangle_perimeter(
            (top, left), end=(bottom, right), shape=(img.size[1], img.size[0])
        )
        im_draw[rr, cc] = (255, 0, 0)

    return im_draw, filtered_bbox_info


def pad_to_square(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    将图像填充为正方形，并返回填充后的图像和填充信息 (left, top, right, bottom)。

    Args:
        img (Image.Image): 输入的 PIL 图像

    Returns:
        Tuple[Image.Image, Tuple[int, int, int, int]]: 填充后的正方形图像和填充信息
    """
    width, height = img.size
    if width == height:
        return img, (0, 0, 0, 0)  # 已经是正方形，直接返回

    # 计算新图像的尺寸
    size = max(width, height)

    # 创建新的正方形图像，背景为白色
    new_img = Image.new("RGB", (size, size), (255, 255, 255))

    # 将原始图像粘贴到新图像的左上角，并记录填充信息
    if width > height:
        # 高度为短边，向下填充
        padding = (0, 0, 0, size - height)  # (left, top, right, bottom)
        new_img.paste(img, (0, 0))
    else:
        # 宽度为短边，向右填充
        padding = (0, 0, size - width, 0)  # (left, top, right, bottom)
        new_img.paste(img, (0, 0))

    return new_img, padding


def adjust_bboxes(bboxes: List[dict], padding: Tuple[int, int, int, int]) -> List[dict]:
    """
    根据填充信息调整边界框的坐标。

    Args:
        bboxes (List[dict]): 边界框信息列表
        padding (Tuple[int, int, int, int]): 填充信息 (left, top, right, bottom)

    Returns:
        List[dict]: 调整后的边界框信息列表
    """
    left, top, right, bottom = padding
    adjusted_bboxes = []
    for bbox in bboxes:
        adjusted_bboxes.append(
            {
                "top": bbox["top"] - top,
                "left": bbox["left"] - left,
                "bottom": bbox["bottom"] - top,
                "right": bbox["right"] - left,
                "score": bbox["score"],
            }
        )
    return adjusted_bboxes


def detect_characters(
    image_path: str,
    model_path: str = "models/HRCenterNet/HRCenterNet.pth.tar",
    nms_score: float = 0.3,
    iou_threshold: float = 0.1,
) -> List[dict]:
    """
    使用HRCenterNet检测图像中的字符边界框

    Args:
        image_path (str): 输入图像路径
        model_path (str): 预训练模型路径
        nms_score (float): NMS置信度阈值
        iou_threshold (float): NMS IoU阈值

    Returns:
        List[dict]: 边界框信息列表
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = HRCenterNet()
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE)
    model.eval()

    # 加载图像并填充为正方形
    img = Image.open(image_path).convert("RGB")
    padded_img, padding = pad_to_square(img)  # 填充为正方形

    # 将填充后的图像转换为张量
    image_tensor = test_tx(padded_img).unsqueeze(0).to(DEVICE, dtype=torch.float)

    # 模型预测
    with torch.no_grad():
        predict = model(image_tensor)

    # 应用NMS并获取边界框信息
    _, bbox_info = apply_nms(padded_img, predict, nms_score, iou_threshold)

    # 调整边界框坐标，去除填充部分的影响
    adjusted_bbox_info = adjust_bboxes(bbox_info, padding)

    return adjusted_bbox_info


def main():
    """
    主函数，用于命令行执行单张图片的目标检测
    """
    parser = argparse.ArgumentParser(description="HRCenterNet单图目标检测脚本")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--output", type=str, default=None, help="输出图像路径（默认在当前目录生成）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/HRCenterNet/HRCenterNet.pth.tar",
        help="预训练模型路径",
    )
    parser.add_argument("--nms_score", type=float, default=0.3, help="NMS置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="NMS IoU阈值")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # 调用 detect_characters 函数进行检测
    bbox_info = detect_characters(
        image_path=args.input,
        model_path=args.model,
        nms_score=args.nms_score,
        iou_threshold=args.iou_threshold,
    )

    # 加载原始图像
    img = Image.open(args.input).convert("RGB")

    # 绘制边界框
    draw = ImageDraw.Draw(img)
    for bbox in bbox_info:
        draw.rectangle(
            [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]],
            outline="red",
            width=2,
        )

    # 保存输出图片到当前目录
    input_filename = os.path.splitext(os.path.basename(args.input))[0]
    output_image_path = args.output if args.output else f"{input_filename}_detected.png"
    img.save(output_image_path)
    print(f"检测后的图像已保存到: {output_image_path}")

    # 保存边界框信息到JSON文件
    output_json_path = f"{input_filename}_bbox.json"
    with open(output_json_path, "w") as f:
        json.dump(bbox_info, f, indent=4)
    print(f"边界框信息已保存到: {output_json_path}")


if __name__ == "__main__":
    main()
