import os
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# 配置设置
RESIZE_SCALE = 64  # 图像缩放尺寸
NUM_CLASSES = 5592  # 类别数
W2I_DICT_PATH = os.path.join("models/cnn", "w2i.json")  # 字典路径
MODEL_WEIGHTS_PATH = os.path.join("models/cnn", "best.weights.h5")  # 模型权重路径


def build_model(input_shape=(RESIZE_SCALE, RESIZE_SCALE, 1), num_classes=NUM_CLASSES):
    """
    构建CNN模型

    Args:
        input_shape (tuple): 输入形状，默认使用RESIZE_SCALE
        num_classes (int): 分类类别数，默认使用NUM_CLASSES

    Returns:
        tf.keras.Model: 未编译的模型（适用于预测）
    """
    input_ = tf.keras.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(
        64, kernel_size=7, strides=2, padding="same", activation="relu"
    )(input_)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(
        256, kernel_size=3, strides=2, padding="same", activation="relu"
    )(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(
        512, kernel_size=3, strides=2, padding="same", activation="relu"
    )(pool2)
    pool3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
    flat = tf.keras.layers.Flatten()(pool3)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(flat)
    model = tf.keras.Model(inputs=input_, outputs=output)
    return model


def get_w2i_dict(dict_path=W2I_DICT_PATH):
    """
    加载字到ID的映射字典

    Args:
        dict_path (str): 字典文件路径，默认使用W2I_DICT_PATH

    Returns:
        tuple: (w2i字典, i2w字典)
    """
    if os.path.exists(dict_path):
        with open(dict_path, "r") as f:
            w2i = json.load(f)
        i2w = {idx: char for char, idx in w2i.items()}
        print(f"加载了{len(w2i)}个字符的字典")
        return w2i, i2w
    else:
        raise FileNotFoundError(f"未找到{dict_path}文件，请检查文件是否存在")


def preprocess_image(image_path):
    """
    预处理输入图像

    Args:
        image_path (str): 输入图像路径

    Returns:
        tf.Tensor: 预处理后的图像张量
    """
    img = Image.open(image_path).convert("L")
    img = np.array(img)
    img = 255 - img  # 反转颜色
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    img = tf.image.resize_with_pad(img[..., tf.newaxis], RESIZE_SCALE, RESIZE_SCALE)
    img = tf.expand_dims(img, 0)
    return img


def predict_character(image_path, cnn_model, w2i, i2w, top_k=5):
    """
    预测单字图像的字符

    Args:
        image_path (str): 输入图像路径
        cnn_model: 已加载权重的模型
        w2i (dict): 字到ID的映射字典
        i2w (dict): ID到字的映射字典
        top_k (int): 返回前K个预测结果

    Returns:
        tuple: (最可能字符, 前K个预测结果及其概率)
    """
    img = preprocess_image(image_path)
    output = cnn_model(img, training=False)
    top_k_values, top_k_indices = tf.nn.top_k(output, k=top_k)
    top_k_values = top_k_values.numpy()[0]
    top_k_indices = top_k_indices.numpy()[0]
    top_k_chars = [i2w[idx] for idx in top_k_indices]
    top_k_probs = [float(val) for val in top_k_values]
    predicted_char = top_k_chars[0]
    return predicted_char, list(zip(top_k_chars, top_k_probs))


def main():
    """
    主函数，用于命令行执行单字预测
    """
    parser = argparse.ArgumentParser(description="单字检测脚本")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--output", type=str, default=None, help="输出结果路径（可选，默认打印到终端）"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_WEIGHTS_PATH, help="模型权重路径"
    )
    parser.add_argument("--dict", type=str, default=W2I_DICT_PATH, help="字典路径")
    parser.add_argument("--top_k", type=int, default=5, help="返回前K个预测结果")
    args = parser.parse_args()

    w2i, i2w = get_w2i_dict(args.dict)
    cnn_model = build_model(num_classes=len(w2i))
    cnn_model.load_weights(args.model)
    print(f"从{args.model}加载模型权重")

    predicted_char, top_k_results = predict_character(
        args.input, cnn_model, w2i, i2w, args.top_k
    )

    result = {"predicted_char": predicted_char, "top_k_predictions": top_k_results}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"结果已保存到: {args.output}")
    else:
        print(f"\n预测结果：")
        print(f"最可能的字符: {predicted_char}")
        print(f"前{args.top_k}个预测结果及其概率:")
        for char, prob in top_k_results:
            print(f"字符: {char}, 概率: {prob:.4f}")


if __name__ == "__main__":
    main()
