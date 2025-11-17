import os
import json
from pathlib import Path
import glob

from PIL import Image, ImageDraw

# ==========================
# 配置区域
# ==========================

# 数据根目录（与之前跑 bbox 的 BASE_DIR 一致）
BASE_DIR = Path("/inspire/hdd/global_user/konghanlin-253108540238/user_cache/datasets/tmp_symlink_only_last")

# 结果 JSONL 文件名（与你保存时用的一致）
RESULT_FILENAME = "qwen3_vl_plus_withoutNONE_bailian_bboxes.jsonl"

# 可视化图片输出根目录（会自动创建，并复用原始目录结构）
OUTPUT_BASE_DIR = BASE_DIR / "vis_bboxes"

# 是否只画 found==True 的记录
ONLY_WHEN_FOUND = False  # 如果只想画有检测到的，可以设为 True


# ==========================
# 工具函数
# ==========================

def find_all_result_files() -> list[Path]:
    """
    在 BASE_DIR 下递归查找所有结果 jsonl 文件
    """
    pattern = str(BASE_DIR / "**" / RESULT_FILENAME)
    paths = glob.glob(pattern, recursive=True)
    return [Path(p) for p in paths if Path(p).is_file()]


def load_records_from_jsonl(jsonl_path: Path):
    """
    从 jsonl 文件中逐行读取记录
    如果你的是普通 .json 文件（一个 list），可以改成：
        data = json.loads(jsonl_path.read_text('utf-8'))
        for record in data:
            yield record
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {jsonl_path} 某一行, 错误: {e}")
                continue
            yield record


def compute_output_image_path(image_path: Path) -> Path:
    """
    根据原始图片路径计算输出图片路径：
    - 相对于 BASE_DIR 的结构保持不变
    - 文件名加后缀 _bbox
    """
    try:
        rel = image_path.relative_to(BASE_DIR)
    except ValueError:
        # 不在 BASE_DIR 之下，就都丢到根目录下
        rel = image_path.name

    out_path = OUTPUT_BASE_DIR / rel
    out_path = out_path.with_name(out_path.stem + "_bbox" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def draw_bboxes_on_image(image_path: Path, boxes, extract_text: str | None = None) -> Image.Image:
    """
    在图片上绘制 bounding boxes。
    boxes: [[x1, y1, x2, y2], ...]
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size

    if boxes is None:
        boxes = []

    for bbox in boxes:
        if bbox is None:
            continue
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox

        # 简单边界裁剪，避免越界
        x1 = max(0, min(x1/1000*w, w - 1))
        x2 = max(0, min(x2/1000*w, w - 1))
        y1 = max(0, min(y1/1000*h, h - 1))
        y2 = max(0, min(y2/1000*h, h - 1))

        # 画框（默认颜色 / 宽度）
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # 简单在框左上角写一点说明（可选）
        if extract_text:
            text_pos = (x1 + 2, y1 + 2)
            draw.text(text_pos, extract_text, fill="red")

    return img


# ==========================
# 主流程
# ==========================

def main():
    print("=" * 80)
    print("开始可视化 bounding box")
    print("=" * 80)

    result_files = find_all_result_files()
    print(f"在 {BASE_DIR} 下找到 {len(result_files)} 个结果文件: {RESULT_FILENAME}")

    total_records = 0
    total_with_boxes = 0

    for jf in result_files:
        print(f"\n处理结果文件: {jf}")
        for record in load_records_from_jsonl(jf):
            total_records += 1

            image_path_str = record.get("image_path")
            boxes = record.get("boxes", [])
            found = record.get("found", False)
            extract_text = record.get("extract", None)

            if ONLY_WHEN_FOUND and not found:
                continue

            if not image_path_str:
                continue

            image_path = Path(image_path_str)

            if not image_path.is_file():
                print(f"  警告: 图片不存在: {image_path}")
                continue

            if not boxes:
                # 没有框就不画
                continue

            # 画框
            try:
                img = draw_bboxes_on_image(image_path, boxes, extract_text)
            except Exception as e:
                print(f"  处理图片 {image_path} 失败: {e}")
                continue

            out_path = compute_output_image_path(image_path)
            img.save(out_path)
            total_with_boxes += 1

            print(f"  已保存可视化图片: {out_path}")

    print("\n" + "=" * 80)
    print("可视化完成！")
    print(f"  总记录数: {total_records}")
    print(f"  已输出带框图片: {total_with_boxes}")
    print(f"  输出根目录: {OUTPUT_BASE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
