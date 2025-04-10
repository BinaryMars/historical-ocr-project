from flask import Flask, request, jsonify, send_file, Response, send_from_directory
import os
import json
import cv2
from main_ocr import process_image_to_ocr

app = Flask(__name__, static_folder="static", static_url_path="/static")
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
CHARACTERS_FOLDER = os.path.join(RESULTS_FOLDER, "characters")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def split_image_by_columns(image_path, ocr_data, output_folder, filename_prefix):
    """按列切分图片，并为每个字符生成独立切片，返回每列的图片路径和字符数据"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    for item in ocr_data:
        item["x_center"] = (item["left"] + item["right"]) / 2
        item["y_center"] = (item["top"] + item["bottom"]) / 2

    ocr_data.sort(key=lambda x: x["x_center"])

    if ocr_data:
        avg_width = sum((item["right"] - item["left"]) for item in ocr_data) / len(
            ocr_data
        )
    else:
        avg_width = 50

    columns = []
    current_column = []
    last_x = None

    for item in ocr_data:
        x_center = item["x_center"]
        if last_x is None or abs(x_center - last_x) < avg_width * 0.5:
            current_column.append(item)
        else:
            columns.append(current_column)
            current_column = [item]
        last_x = x_center
    if current_column:
        columns.append(current_column)

    column_data = []
    os.makedirs(output_folder, exist_ok=True)

    for col_idx, column in enumerate(columns):
        if not column:
            continue

        left = min(item["left"] for item in column)
        right = max(item["right"] for item in column)
        top = min(item["top"] for item in column)
        bottom = max(item["bottom"] for item in column)

        padding = 10
        left = max(0, left - padding)
        right = min(width, right + padding)
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)

        column_image = image[top:bottom, left:right]
        column_image_filename = f"{filename_prefix}_column_{col_idx}.png"
        column_image_path = os.path.join(output_folder, column_image_filename)
        cv2.imwrite(column_image_path, column_image)

        column.sort(key=lambda x: (x["top"] + x["bottom"]) / 2)

        column_chars = []
        for char_idx, item in enumerate(column):
            char_left = max(0, item["left"] - 5)
            char_right = min(width, item["right"] + 5)
            char_top = max(0, item["top"] - 5)
            char_bottom = min(height, item["bottom"] + 5)

            char_image = image[char_top:char_bottom, char_left:char_right]
            char_image_filename = (
                f"{filename_prefix}_column_{col_idx}_char_{char_idx}.png"
            )
            char_image_path = os.path.join(output_folder, char_image_filename)
            cv2.imwrite(char_image_path, char_image)

            column_chars.append(
                {
                    "text": item.get("final_char", item["predicted_char"]),
                    "top": item["top"] - top,
                    "bottom": item["bottom"] - top,
                    "left": item["left"] - left,
                    "right": item["right"] - left,
                    "image_path": char_image_filename,
                    "order": item["order"],
                    "top_k_predictions": item["top_k_predictions"],
                }
            )

        column_data.append(
            {
                "image_path": column_image_filename,
                "image_height": bottom - top,
                "chars": column_chars,
            }
        )

    return column_data


@app.route("/")
def index():
    return send_file("templates/index.html")


@app.route("/download", methods=["POST"])
def download_file():
    text_content = request.form.get("text", "")
    return Response(
        text_content,
        mimetype="text/plain",
        headers={
            "Content-Disposition": 'attachment; filename="recognized_text.txt"',
            "Content-Length": str(len(text_content.encode("utf-8"))),
        },
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(input_path)

        # 使用统一格式的 JSON 文件名
        base_name = os.path.splitext(os.path.basename(filename))[0]
        binary_image = os.path.join(
            app.config["RESULTS_FOLDER"], f"0.{base_name}_binary.png"
        )
        sort_bboxes_path = os.path.join(
            app.config["RESULTS_FOLDER"], f"3.{base_name}_sortedBboxes.png"
        )
        output_json_path = os.path.join(
            app.config["RESULTS_FOLDER"], f"4.{base_name}_ocr_result.json"
        )

        try:
            process_image_to_ocr(
                input_image_path=input_path, output_json_path=output_json_path
            )
            if not os.path.exists(output_json_path):
                return jsonify(
                    {"error": f"JSON file not found: {output_json_path}"}
                ), 500
            if not os.path.exists(binary_image):
                return jsonify({"error": f"Image file not found: {binary_image}"}), 500

            with open(output_json_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            filename_prefix = base_name
            column_data = split_image_by_columns(
                binary_image, ocr_data, app.config["RESULTS_FOLDER"], filename_prefix
            )

            return jsonify(
                {
                    "original_image_path": sort_bboxes_path,
                    "columns": column_data,
                    "base_name": base_name,  # 返回 base_name 用于清理
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400


@app.route("/cleanup", methods=["POST"])
def cleanup_files():
    data = request.get_json()
    base_name = data.get("base_name")
    if not base_name:
        return jsonify({"error": "No base_name provided"}), 400

    try:
        # 定义保留文件的前缀
        # keep_prefixes = [f"{i}.{base_name}" for i in range(5)]
        keep_prefixes = [f"{i}." for i in range(5)]

        # 清理 results 文件夹中的非必要文件
        result_dir = app.config["RESULTS_FOLDER"]
        for filename in os.listdir(result_dir):
            file_path = os.path.join(result_dir, filename)
            if os.path.isfile(file_path) and not any(
                filename.startswith(prefix) for prefix in keep_prefixes
            ):
                os.remove(file_path)

        # 删除 characters 文件夹内的所有图片文件
        if os.path.exists(CHARACTERS_FOLDER):
            for filename in os.listdir(CHARACTERS_FOLDER):
                file_path = os.path.join(CHARACTERS_FOLDER, filename)
                if os.path.isfile(file_path):  # 只删除文件，不删除子目录（如果有）
                    os.remove(file_path)
            os.rmdir(CHARACTERS_FOLDER)

        return jsonify({"message": "Cleanup completed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


@app.route("/results/<path:filename>")
def serve_results(filename):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(app.config["RESULTS_FOLDER"], safe_filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404
    return send_file(file_path)


if __name__ == "__main__":
    app.run(debug=True)
