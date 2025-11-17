import os
from modelscope.hub.api import HubApi

# ModelScope 仓库信息
MS_REPO_ID = "mikegoblin/walloss_weights"
MS_TOKEN = "ms-93ed7844-47d7-4250-abf7-7f53d83bdb78"

# --- 配置参数 ---
# 指定本地上传目录
# LOCAL_UPLOAD_DIR = "./wall_pt"
LOCAL_UPLOAD_DIR = "./wallx_pt"
# 是否递归上传子目录
UPLOAD_RECURSIVE = True


def collect_files(base_dir: str, recursive: bool = True):
    """收集目录下所有文件路径"""
    file_list = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, start=base_dir)
            file_list.append((abs_path, rel_path.replace("\\", "/")))
        if not recursive:
            break
    return file_list


def main():
    """上传指定目录下所有文件（不对比远程）"""
    print("正在登录 ModelScope...")
    api = HubApi()
    try:
        api.login(MS_TOKEN)
        print("ModelScope 登录成功。")
    except Exception as e:
        print(f"ModelScope 登录失败: {e}")
        return

    if not os.path.exists(LOCAL_UPLOAD_DIR):
        print(f"错误：指定的上传目录不存在: {LOCAL_UPLOAD_DIR}")
        return

    # 1. 收集本地文件
    print(f"正在扫描目录: {LOCAL_UPLOAD_DIR} ...")
    all_files = collect_files(LOCAL_UPLOAD_DIR, recursive=UPLOAD_RECURSIVE)

    if not all_files:
        print("未找到任何文件可上传。")
        return

    print(f"共找到 {len(all_files)} 个文件待上传。")

    # 2. 上传文件
    upload_counter = 0
    for abs_path, rel_path in sorted(all_files, key=lambda x: x[1]):
        upload_counter += 1
        print(f"\n--- ({upload_counter}/{len(all_files)}) 上传: {rel_path} ---")
        try:
            api.upload_file(
                repo_id=MS_REPO_ID,
                path_or_fileobj=abs_path,
                path_in_repo=rel_path,  # 保留目录结构
                repo_type="dataset",
                commit_message="Upload directory contents",
            )
            print("  文件上传成功。")
        except Exception as e:
            print(f"  上传失败: {e}")
            continue

    print("\n所有文件上传完毕！")


if __name__ == "__main__":
    main()