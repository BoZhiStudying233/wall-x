import pandas as pd
from PIL import Image
import io
import os

# 1️⃣ 读取 parquet 文件
file_path = "/inspire/hdd/global_user/konghanlin-253108540238/user_cache/lerobot/physical-intelligence/libero/data/chunk-000/episode_000000.parquet"

df = pd.read_parquet(file_path)

# 2️⃣ 获取 image 列的第一个样本
image_data = df["image"].iloc[1]

# image_data 应该是一个字典：{'bytes': b'...', 'path': 'xxx.png'}
binary_data = image_data["bytes"]
image_path = image_data.get("path", "recovered.png") or "recovered.png"

# 3️⃣ 从 bytes 还原图片
try:
    img_stream = io.BytesIO(binary_data)
    img = Image.open(img_stream)
    img.save(image_path)
    print(f"✅ 图片已恢复并保存为：{image_path}")
    img.show()
except Exception as e:
    print(f"❌ 恢复图片失败：{e}")
