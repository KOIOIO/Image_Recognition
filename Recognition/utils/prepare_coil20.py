import os
import re
import zipfile
import urllib.request
from urllib.error import URLError, HTTPError
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import random
from typing import Optional, Tuple

"""
COIL-20 数据集准备脚本

功能：
- 下载（或使用已有）COIL-20 压缩包
- 解压至 data/raw/coil-20
- 按对象编号（obj1~obj20）整理为 ImageFolder 结构
- 划分 train/test（默认 80/20）
- 转换为 RGB，确保与训练时的归一化维度一致

使用方法：
python utils/prepare_coil20.py

脚本会在项目根目录下的 data 目录中生成：
- data/train/obj1, obj2, ..., obj20
- data/test/obj1, obj2, ..., obj20
"""

COIL20_URLS = [
    # 常见镜像与路径（优先 https）
    "https://www.cs.columbia.edu/CAVE/databases/coil-20/coil-20-proc.zip",
    "https://www.cs.columbia.edu/CAVE/databases/coil-20/coil-20.zip",
    "http://www.cs.columbia.edu/CAVE/databases/coil-20/coil-20-proc.zip",
    "http://www.cs.columbia.edu/CAVE/databases/coil-20/coil-20.zip",
]

def download_coil20_zip(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    # 支持两种常见文件名
    candidates = [raw_dir / "coil-20-proc.zip", raw_dir / "coil-20.zip"]
    for zp in candidates:
        if zp.exists():
            print(f"已存在压缩包：{zp}")
            return zp
    # 逐个URL尝试下载
    last_err = None
    for url in COIL20_URLS:
        try:
            print(f"尝试下载 COIL-20：{url}")
            target = raw_dir / ("coil-20-proc.zip" if "proc" in url else "coil-20.zip")
            urllib.request.urlretrieve(url, str(target))
            print(f"下载完成：{target}")
            return target
        except (URLError, HTTPError) as e:
            last_err = e
            print(f"下载失败 {url}：{e}")
        except Exception as e:
            last_err = e
            print(f"下载异常 {url}：{e}")
    # 所有URL失败
    raise RuntimeError(
        f"无法下载 COIL-20 压缩包，请手动将文件放置到 {raw_dir}，文件名为 coil-20-proc.zip 或 coil-20.zip。最后错误：{last_err}"
    )

def unzip(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"解压 {zip_path} 至 {extract_to}")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_to))
    print("解压完成")

def organize_to_imagefolder(extracted_dir: Path, data_dir: Path, split_ratio: float = 0.8):
    """
    将解压后的文件整理为 ImageFolder 结构：data/train 和 data/test
    假设文件命名格式包含 objN（如 obj1__0.png、obj20__355.png）
    """
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"obj(\d+)")
    image_files = [p for p in extracted_dir.rglob("*.png")]
    if not image_files:
        # 某些版本可能是 .ppm 或其他格式
        image_files = [p for p in extracted_dir.rglob("*.ppm")]

    if not image_files:
        raise FileNotFoundError("未在解压目录中找到图像文件（*.png 或 *.ppm）")

    # 分组：objN -> [files]
    groups = {}
    for p in image_files:
        m = pattern.search(p.name)
        if not m:
            # 跳过不符合命名的文件
            continue
        obj_id = int(m.group(1))
        cls_name = f"obj{obj_id}"
        groups.setdefault(cls_name, []).append(p)

    # 为每个类别创建目录，并按比例划分 train/test
    for cls_name, files in groups.items():
        files.sort()
        n = len(files)
        split = int(n * split_ratio)
        train_files = files[:split]
        test_files = files[split:]

        cls_train_dir = train_dir / cls_name
        cls_test_dir = test_dir / cls_name
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_test_dir.mkdir(parents=True, exist_ok=True)

        # 复制并统一为 RGB PNG
        for src, dst_dir in [(f, cls_train_dir) for f in train_files] + [(f, cls_test_dir) for f in test_files]:
            try:
                img = Image.open(src)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # 保存到目标目录，使用原文件名
                target_path = dst_dir / (src.stem + ".png")
                img.save(target_path)
            except Exception as e:
                print(f"处理图像 {src} 失败：{e}")

    print(f"已整理为 ImageFolder 结构：{train_dir} 与 {test_dir}")

def _augment_image(img: Image.Image) -> Image.Image:
    """对单张图像应用一次随机增强，返回增强后的图像。
    增强包含：随机旋转、水平翻转、亮度/对比度/色彩抖动、轻微高斯噪声。
    """
    # 确保是RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    augmented = img.copy()

    # 随机旋转 (-25°, 25°)
    angle = random.uniform(-25, 25)
    augmented = augmented.rotate(angle, resample=Image.BILINEAR, expand=True)

    # 随机水平翻转
    if random.random() < 0.5:
        augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)

    # 颜色抖动：亮度、对比度、色彩
    brightness_factor = random.uniform(0.85, 1.15)
    contrast_factor = random.uniform(0.85, 1.15)
    color_factor = random.uniform(0.85, 1.15)
    augmented = ImageEnhance.Brightness(augmented).enhance(brightness_factor)
    augmented = ImageEnhance.Contrast(augmented).enhance(contrast_factor)
    augmented = ImageEnhance.Color(augmented).enhance(color_factor)

    # 轻微高斯噪声（标准差约5/255）
    arr = np.array(augmented).astype(np.float32)
    noise = np.random.normal(0, 5.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    augmented = Image.fromarray(arr)

    return augmented

def augment_train_images(train_dir: Path, augment_per_image: int = 1, target_size: Optional[Tuple[int, int]] = None):
    """对训练集进行离线增强，每张图片生成指定数量的增强样本。

    参数：
    - train_dir: 训练集根目录（包含各类别子目录）
    - augment_per_image: 每张原始图片生成的增强样本数
    - target_size: 可选，将增强后的图像统一缩放到该尺寸，如 (128, 128)
    """
    if augment_per_image <= 0:
        print("已跳过增强：augment_per_image <= 0")
        return

    classes = [d for d in train_dir.iterdir() if d.is_dir()]
    total_augmented = 0
    for cls_dir in classes:
        files = sorted([p for p in cls_dir.glob('*.png')])
        for src in files:
            try:
                base_name = src.stem
                # 避免对增强生成的文件再次增强
                if base_name.endswith('_aug') or '_aug' in base_name:
                    continue
                img = Image.open(src)
                for i in range(augment_per_image):
                    aug = _augment_image(img)
                    if target_size is not None:
                        aug = aug.resize(target_size, resample=Image.BILINEAR)
                    out_name = f"{base_name}_aug{i+1}.png"
                    out_path = cls_dir / out_name
                    aug.save(out_path)
                    total_augmented += 1
            except Exception as e:
                print(f"增强图像 {src} 失败：{e}")
    print(f"训练集离线增强完成，共生成 {total_augmented} 张增强样本")

def prepare_coil20_dataset(project_root: Path = None, split_ratio: float = 0.8, zip_source: Path = None,
                           augment: bool = True, augment_per_image: int = 1):
    if project_root is None:
        # utils 目录的父级即项目根目录
        project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    extracted_dir = raw_dir / "coil-20"

    # 下载并解压
    if zip_source is not None:
        zip_path = Path(zip_source)
        if not zip_path.exists():
            raise FileNotFoundError(f"指定的本地压缩包不存在：{zip_path}")
        print(f"使用本地压缩包：{zip_path}")
    else:
        zip_path = download_coil20_zip(raw_dir)
    unzip(zip_path, extracted_dir)

    # 整理为 ImageFolder
    organize_to_imagefolder(extracted_dir, data_dir, split_ratio)

    # 离线增强（仅训练集），并添加防重复标记
    aug_flag = data_dir / 'aug_coil20_done.flag'
    train_dir = data_dir / 'train'
    if augment:
        if aug_flag.exists():
            print("检测到增强标记，跳过重复增强。")
        else:
            # 可选：统一增强图像大小为与加载时一致的 128x128
            target_size = (128, 128)
            augment_train_images(train_dir, augment_per_image=augment_per_image, target_size=target_size)
            try:
                aug_flag.write_text("COIL-20 augmentation done\n", encoding='utf-8')
            except Exception:
                pass

    print("COIL-20 数据集准备完成！")

if __name__ == "__main__":
    prepare_coil20_dataset()