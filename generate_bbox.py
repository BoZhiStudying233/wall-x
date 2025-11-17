import os
import json
import base64
import asyncio
import time
from pathlib import Path
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import glob

# ==========================
# 配置区域
# ==========================
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-69995b98e7f94ccc82e8f7d87ee22fdb"  # 建议设置环境变量 export DASHSCOPE_API_KEY=sk-xxxx
MODEL_NAME = "qwen3-vl-plus"
MAX_CONCURRENT_REQUESTS = 10
SLEEP_BETWEEN_BATCHES = 0.1

# 根据你的真实数据路径设置 BASE_DIR
# 示例：/inspire/.../tmp_symlink_exclude_last/50/1/1-1/images/front/camera0_00035.jpg
# 则 BASE_DIR 为 /inspire/.../tmp_symlink_exclude_last
BASE_DIR = Path("/inspire/hdd/global_user/konghanlin-253108540238/user_cache/datasets/tmp_symlink_only_last")

OUTPUT_FILENAME = "qwen3_vl_plus_withoutNONE_bailian_bboxes.jsonl"

# 对应 p/n/n-m/images/front
IMAGE_DIR_PATTERN = "*/*/*/images/front"

PROMPT_TEMPLATE = (
    "给出物体{extract}的bounding box，其他什么都不要输出。"
    "输出格式：bounding box：[x1, y1, x2, y2]"
)

USE_PIL_FOR_SIZE = True


# ==========================
# 工具函数
# ==========================

def find_all_image_dirs() -> List[Path]:
    """查找所有包含 front 图片的目录：p/n/n-m/images/front"""
    pattern = str(BASE_DIR / IMAGE_DIR_PATTERN)
    dirs = glob.glob(pattern)
    return [Path(d) for d in dirs if os.path.isdir(d)]


def extract_subtask_id(image_dir: Path) -> Optional[str]:
    """
    期望路径：.../<BASE_DIR>/<p>/<n>/<n-m>/images/front
    返回 id: f"{p}_{n-m}"
    """
    try:
        nm_dir = image_dir.parents[1].name  # n-m
        p_dir_name = image_dir.parents[3].name  # p
        return f"{p_dir_name}_{nm_dir}"
    except Exception as e:
        print(f"警告: 从路径 {image_dir} 提取 subtask_id 失败: {e}")
        return None


def get_image_files(image_dir: Path) -> List[Path]:
    """获取目录中所有图片文件，按名称排序"""
    image_files = sorted(image_dir.glob("*.jpg"))
    return image_files


def encode_image_to_base64(image_path: Path) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def parse_bbox_from_response(text: str):
    """从响应中解析 bounding box"""
    reasoning_content = text.strip()

    if "none" in text.lower() or "未找到" in text or "没有" in text or "找不到" in text:
        return None, reasoning_content

    bbox_pattern1 = r'bounding box[：:]\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern1, text, re.IGNORECASE)
    if match:
        bbox = [int(match.group(i)) for i in range(1, 5)]
        return bbox, reasoning_content

    bbox_pattern2 = r'bbox_2d["\']?\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern2, text)
    if match:
        bbox = [int(match.group(i)) for i in range(1, 5)]
        return bbox, reasoning_content

    general_bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(general_bbox_pattern, text)
    if matches:
        bbox = [int(x) for x in matches[-1]]
        return bbox, reasoning_content

    return None, reasoning_content


def parse_image_size_from_response(text: str):
    """从响应中解析图片尺寸"""
    size_pattern = r'(?:picture size|图片尺寸|尺寸)[：:]\s*\[(\d+),\s*(\d+)\]'
    match = re.search(size_pattern, text, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# ========== instruction.txt 解析相关 ==========
INSTRUCTION_CACHE: Dict[Path, Tuple[Optional[str], Optional[str]]] = {}


def parse_instruction_file(p_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    从 p 目录下的 instruction.txt 中解析出：
    Catch: AAA.
    Put:   BBB
    返回 (catch_text, put_text)
    """
    if p_dir in INSTRUCTION_CACHE:
        return INSTRUCTION_CACHE[p_dir]

    instruction_path = p_dir / "instruction.txt"
    if not instruction_path.is_file():
        print(f"警告: {instruction_path} 不存在")
        INSTRUCTION_CACHE[p_dir] = (None, None)
        return INSTRUCTION_CACHE[p_dir]

    try:
        text = instruction_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"警告: 读取 {instruction_path} 失败: {e}")
        INSTRUCTION_CACHE[p_dir] = (None, None)
        return INSTRUCTION_CACHE[p_dir]

    # 示例：Pick up ... Catch: book. Put: bed
    catch_match = re.search(r'Catch:\s*([^\.]+)', text)
    put_match = re.search(r'Put:\s*([^\.]+)', text)

    catch_text = catch_match.group(1).strip() if catch_match else None
    put_text = put_match.group(1).strip() if put_match else None

    if catch_text is None or put_text is None:
        print(f"警告: 在 {instruction_path} 中未能解析 Catch/Put，原文为：{text}")

    INSTRUCTION_CACHE[p_dir] = (catch_text, put_text)
    return INSTRUCTION_CACHE[p_dir]


# ==========================
# 调用 Qwen3VL 接口
# ==========================
def call_api_sync(image_path: Path, extract_text: str, client: OpenAI) -> dict:
    """同步调用 API"""
    try:
        start_time = time.time()
        base64_image = encode_image_to_base64(image_path)
        prompt = PROMPT_TEMPLATE.format(extract=extract_text)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            extra_body={"enable_thinking": True},  # 启用深度思考
        )

        elapsed_time = time.time() - start_time
        msg = response.choices[0].message
        raw_text = msg.content

        # 兼容不同 SDK 字段
        thinking_output = getattr(msg, "thinking", None)
        if thinking_output is None and hasattr(msg, "reasoning_content"):
            thinking_output = msg.reasoning_content

        bbox, reasoning = parse_bbox_from_response(raw_text)
        image_size = parse_image_size_from_response(raw_text)

        return {
            'success': True,
            'bbox': bbox,
            'reasoning': reasoning,
            'raw_text': raw_text,
            'thinking_output': thinking_output,
            'image_size': image_size,
            'elapsed_sec': round(elapsed_time, 4)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'elapsed_sec': time.time() - start_time
        }


# ==========================
# 异步处理逻辑
# ==========================
async def process_image(image_path: Path, extract_text: str, subtask_id: str,
                       reference_image_path: Path, client: OpenAI,
                       executor: ThreadPoolExecutor) -> dict:
    """异步处理单张图片"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, call_api_sync, image_path, extract_text, client
    )

    record = {
        'image_path': str(image_path),
        'reference_image_path': str(reference_image_path),
        'subtask_id': subtask_id,
        'extract': extract_text,
        'model': MODEL_NAME,
        'elapsed_sec': result.get('elapsed_sec', 0)
    }

    if result['success']:
        record['found'] = result['bbox'] is not None
        record['boxes'] = [result['bbox']] if result['bbox'] else []
        record['raw_text'] = result['raw_text']
        record['reasoning'] = result['reasoning']
        record['thinking_output'] = result.get('thinking_output')

        if result['image_size']:
            record['width'], record['height'] = result['image_size']
        else:
            if USE_PIL_FOR_SIZE:
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        record['width'], record['height'] = img.size
                except Exception as e:
                    print(f"警告: 无法读取图片尺寸 {image_path.name}: {e}")
                    record['width'], record['height'] = None, None
            else:
                record['width'], record['height'] = None, None
    else:
        record['found'] = False
        record['boxes'] = []
        record['error'] = result.get('error', 'Unknown error')
        record['raw_text'] = None
        record['reasoning'] = None
        record['thinking_output'] = None

    return record


async def process_with_semaphore(semaphore: asyncio.Semaphore, image_path: Path,
                                 extract_text: str, subtask_id: str,
                                 reference_image: Path, client: OpenAI,
                                 executor: ThreadPoolExecutor, current: int, total: int) -> dict:
    """使用信号量控制并发"""
    async with semaphore:
        result = await process_image(
            image_path, extract_text, subtask_id,
            reference_image, client, executor
        )
        status = '找到' if result.get('found') else '未找到'
        elapsed = result.get('elapsed_sec', 0)
        print(f"    [{current}/{total}] {image_path.name} - {status} ({elapsed:.2f}s)")
        return result


async def process_subtask(subtask_id: str, extract_text: str, image_dir: Path,
                         client: OpenAI, executor: ThreadPoolExecutor,
                         semaphore: asyncio.Semaphore) -> List[dict]:
    """处理一个子任务：只标注该目录中最后一张 jpg"""
    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"警告: {image_dir} 中没有找到图片文件")
        return []

    # 只取最后一张图片
    target_image = image_files[-1]
    reference_image = target_image

    print(f"处理子任务 {subtask_id} - {extract_text}: 只处理最后一张图片 {target_image.name}")

    task = asyncio.create_task(
        process_with_semaphore(
            semaphore, target_image, extract_text, subtask_id,
            reference_image, client, executor,
            current=1, total=1
        )
    )

    result = await task
    return [result]


# ==========================
# 运行前清理旧结果
# ==========================
def cleanup_old_results():
    """在运行前清理旧的结果 jsonl 文件"""
    pattern = str(BASE_DIR / "**" / OUTPUT_FILENAME)
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("   没有发现需要删除的旧结果文件")
        return
    print(f"   将删除 {len(files)} 个旧结果文件...")
    for f in files:
        try:
            os.remove(f)
            print(f"     已删除: {f}")
        except Exception as e:
            print(f"     删除失败 {f}: {e}")


# ==========================
# 主函数
# ==========================
async def main():
    print("=" * 80)
    print("开始批量检测 Bounding Box (Qwen3VL-plus)")
    print("=" * 80)

    # 0. 先清理旧 jsonl
    print("\n0. 清理旧的结果文件...")
    cleanup_old_results()

    print("\n1. 不再读取 prompts 文件，改为从 instruction.txt 中解析指令")

    # 查找图片目录
    print("\n2. 查找图片目录...")
    image_dirs = find_all_image_dirs()
    print(f"   找到 {len(image_dirs)} 个 front 图片目录")

    # 初始化 Qwen 客户端
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    total_processed = total_found = total_not_found = total_errors = 0
    processed_subtasks = 0
    overall_start_time = time.time()

    print("\n3. 开始处理...")
    for image_dir in image_dirs:
        # image_dir: .../p/n/n-m/images/front
        try:
            nm_dir_name = image_dir.parents[1].name  # 'n-m'
            n_dir_name = image_dir.parents[2].name   # 'n'
            p_dir = image_dir.parents[3]             # Path('.../p')
        except Exception as e:
            print(f"\n跳过目录 {image_dir}: 无法解析 p/n/n-m 结构 ({e})")
            continue

        subtask_id = extract_subtask_id(image_dir)
        print(f"\n提取到子任务 ID: {subtask_id} 来自目录 {image_dir}")

        # 从 p 目录下的 instruction.txt 获取 Catch/Put
        catch_text, put_text = parse_instruction_file(p_dir)
        if not catch_text or not put_text:
            print(f"跳过子任务 {subtask_id}: instruction.txt 中 Catch/Put 不完整")
            continue

        # 可选：尝试从 n-m 中解析 m，仅用于日志
        m_int = None
        try:
            n_str, m_str = nm_dir_name.split("-", 1)
            m_int = int(m_str)
        except Exception as e:
            print(f"提示: 无法从 {nm_dir_name} 解析出 m（仅用于日志，可忽略）: {e}")

        # ⭐ 用 p 的奇偶决定使用 Catch 还是 Put
        try:
            p_int = int(p_dir.name)
        except ValueError as e:
            print(f"跳过子任务 {subtask_id}: p 目录名 {p_dir.name} 不是数字，无法做奇偶判断 ({e})")
            continue

        if p_int % 2 == 1:
            extract_text = catch_text
            which = "Catch"
        else:
            extract_text = put_text
            which = "Put"

        if m_int is not None:
            print(f"子任务 {subtask_id}: p={p_int}, n-m={nm_dir_name}, m={m_int} (使用 {which}: {extract_text})")
        else:
            print(f"子任务 {subtask_id}: p={p_int}, n-m={nm_dir_name} (使用 {which}: {extract_text})")

        output_dir = image_dir.parent      # .../images
        output_file = output_dir / OUTPUT_FILENAME

        print(f"\n处理目录: {image_dir}")
        results = await process_subtask(
            subtask_id, extract_text, image_dir,
            client, executor, semaphore
        )

        # 写结果（每个 front 目录现在只有一条记录）
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        found = sum(1 for r in results if r.get('found'))
        not_found = sum(1 for r in results if not r.get('found') and 'error' not in r)
        errors = sum(1 for r in results if 'error' in r)

        total_found += found
        total_not_found += not_found
        total_errors += errors
        total_processed += len(results)
        processed_subtasks += 1

        print(f"保存结果到: {output_file}")
        print(f"成功检测: {found} | 未找到: {not_found} | 错误: {errors} | 总计: {len(results)}")

        await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    executor.shutdown(wait=True)
    overall_elapsed = time.time() - overall_start_time

    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"  处理子任务: {processed_subtasks}")
    print(f"  总图片数: {total_processed}")
    if total_processed > 0:
        print(f"  找到物体: {total_found} ({total_found / total_processed * 100:.1f}%)")
        print(f"  未找到: {total_not_found} ({total_not_found / total_processed * 100:.1f}%)")
    else:
        print("  找到物体: 0 | 未找到: 0")
    print(f"  错误: {total_errors}")
    print(f"  总耗时: {overall_elapsed:.2f}秒")
    if overall_elapsed > 0:
        print(f"  平均速度: {total_processed / overall_elapsed:.2f} 张/秒")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
