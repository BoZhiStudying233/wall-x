#!/usr/bin/env python3
import os
from pathlib import Path
import re

# ==========================
# 配置
# ==========================

# 你的根目录
BASE_DIR = Path(
    "/inspire/hdd/global_user/konghanlin-253108540238/user_cache/datasets/tmp_symlink_only_last"
)

# 试跑模式：True = 只打印不改名；确认无误后改成 False
DRY_RUN = False

# 匹配形如 "数字-数字" 的目录名，例如 "30-1"
NAME_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def main():
    if not BASE_DIR.is_dir():
        print(f"ERROR: BASE_DIR 不存在或不是目录: {BASE_DIR}")
        return

    print(f"根目录: {BASE_DIR}")
    print(f"DRY_RUN = {DRY_RUN}")
    print("=" * 80)

    rename_count = 0
    skip_count = 0
    conflict_count = 0

    # 结构假设：BASE_DIR / p / n / (a-b)
    # 例如：.../tmp_symlink_only_last/14/1/30-1
    for p_dir in sorted(BASE_DIR.iterdir()):
        if not p_dir.is_dir():
            continue

        # p 一般是数字（例如 14），也可以不强制
        p_name = p_dir.name
        print(f"\n处理 p 目录: {p_name} -> {p_dir}")

        for n_dir in sorted(p_dir.iterdir()):
            if not n_dir.is_dir():
                continue

            n_name = n_dir.name
            # 我们只对 n 是数字的目录进行修复（比如 1, 2, 3 ...）
            if not n_name.isdigit():
                print(f"  跳过非数字 n 目录: {n_dir}")
                continue

            print(f"  处理 n 目录: {n_name} -> {n_dir}")

            for child in sorted(n_dir.iterdir()):
                if not child.is_dir():
                    continue

                m = NAME_PATTERN.match(child.name)
                if not m:
                    # 例如 images/, others/ 等都跳过
                    continue

                old_a, old_b = m.groups()
                # 新名字的前半部分用父目录 n 的名字，后半部分用原来的 b
                new_name = f"{n_name}-{old_b}"

                if new_name == child.name:
                    # 本来就对，不用改
                    skip_count += 1
                    continue

                new_path = child.with_name(new_name)

                # 如果目标路径已经存在，避免覆盖，打印告警
                if new_path.exists():
                    print(
                        f"    [冲突] {child} -> {new_path} 已存在，跳过"
                    )
                    conflict_count += 1
                    continue

                print(f"    重命名: {child.name}  ->  {new_name}")
                rename_count += 1

                if not DRY_RUN:
                    try:
                        child.rename(new_path)
                    except Exception as e:
                        print(f"      [错误] 重命名失败: {e}")

    print("\n" + "=" * 80)
    print("完成扫描。")
    print(f"  需要重命名的目录数: {rename_count}")
    print(f"  本来就正确的目录数: {skip_count}")
    print(f"  发生命名冲突的目录数: {conflict_count}")
    print(f"  实际是否执行重命名: {'否（DRY_RUN）' if DRY_RUN else '是'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
