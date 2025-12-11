import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_dataset(data_dir, dataset_type='cifar10'):
    """
    加载图像数据集
    参数:
        data_dir: 数据目录路径
        dataset_type: 数据集类型 ('cifar10' 或 'new_dataset')
    返回:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        class_names: 类别名称列表
    """
    # 定义训练集的转换
    # 依据数据集类型设置图像大小与转换
    if dataset_type == 'coil20':
        image_size = 128
        # COIL-20 可能是灰度图，此处统一转换为 RGB，避免归一化维度不匹配
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 默认（如 CIFAR-10、自定义 ImageFolder）
        # 统一转换为 RGB（兼容灰度数据，如 FashionMNIST）并可配置随机灰度增强
        # 灰度增强与 RGB 统一说明：
        # - 为什么转换为 RGB：部分数据（如 FashionMNIST）是单通道灰度，
        #   本项目的归一化参数以 3 通道为准，统一转换避免通道维度不匹配。
        # - RandomGrayscale(p)：对 RGB 图像以概率 p 随机灰度化，
        #   提升模型在灰度域（弱光/无彩色环境）的鲁棒性，同时保留足够的彩色样本比例。
        # - ColorJitter 在灰度下的行为：仍然作用于亮度/对比度（无彩色信息时不会破坏分布），
        #   与 RandomGrayscale 协同提升泛化能力。
        # - 测试/验证阶段不进行随机增强（不包含 RandomGrayscale/Flip/Rotation 等），
        #   保证评估稳定性与可比性。

        # 读取可配置的增强参数（来自 static/model_params.json），若不可用则采用默认值
        enable_random_grayscale = True
        random_grayscale_p = 0.2
        try:
            with open('static/model_params.json', 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                enable_random_grayscale = bool(cfg.get('enable_random_grayscale', enable_random_grayscale))
                # 兼容非法值与字符串
                p_val = cfg.get('random_grayscale_p', random_grayscale_p)
                try:
                    random_grayscale_p = float(p_val)
                except Exception:
                    random_grayscale_p = random_grayscale_p
        except Exception:
            pass

        train_ops = [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert('RGB')),  # 保证后续归一化为3通道
            transforms.RandomHorizontalFlip(),                 # 随机水平翻转增强
            transforms.RandomRotation(10),                     # 随机旋转增强
        ]
        if enable_random_grayscale and random_grayscale_p and random_grayscale_p > 0:
            train_ops.append(transforms.RandomGrayscale(p=random_grayscale_p))  # 灰度增强：RGB数据随机灰度化
        train_ops.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 灰度/颜色抖动增强
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        train_transform = transforms.Compose(train_ops)
        
        # 定义测试集的转换（不做随机增强，保持评估一致性）
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert('RGB')),  # 统一至3通道
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 加载训练集和测试集
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    
    # 获取类别名称
    class_names = train_dataset.classes
    
    # 如果是新数据集类型，可以在这里添加特殊处理
    if dataset_type == 'new_dataset':
        # 可以根据需要对新数据集进行特殊处理
        pass
    
    if dataset_type == 'coil20':
        # COIL-20 无需额外处理，ImageFolder 会依据目录名作为类别标签
        # 但建议目录名为 obj1, obj2, ... obj20，以获得清晰的类别名称
        pass
    
    return train_dataset, test_dataset, class_names

def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    """
    创建数据加载器
    参数:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        batch_size: 批次大小
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 确定是否使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    # 添加drop_last=True以避免小批次导致的BatchNorm层错误
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 修复Windows多进程问题，使用单线程
        pin_memory=True if device.type == 'cuda' else False,  # GPU时锁定内存
        drop_last=True  # 丢弃最后一个不完整的批次，避免BatchNorm层在训练时的错误
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # 修复Windows多进程问题，使用单线程
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, test_loader

def visualize_sample_images(dataset, class_names, num_samples=5):
    """
    可视化数据集中的样本图像
    参数:
        dataset: 数据集
        class_names: 类别名称列表
        num_samples: 每个类别显示的样本数量
    """
    # 创建一个字典来跟踪每个类别的样本数量
    class_samples = {class_name: 0 for class_name in class_names}
    samples_to_display = []
    
    # 遍历数据集，收集每个类别的样本
    for img, label in dataset:
        class_name = class_names[label]
        if class_samples[class_name] < num_samples:
            samples_to_display.append((img, class_name))
            class_samples[class_name] += 1
        
        # 检查是否收集了所有类别的足够样本
        if all(count >= num_samples for count in class_samples.values()):
            break
    
    # 计算总样本数和行数
    total_samples = len(samples_to_display)
    rows = (total_samples + num_samples - 1) // num_samples
    
    # 创建图像网格
    plt.figure(figsize=(num_samples * 3, rows * 3))
    
    for i, (img, class_name) in enumerate(samples_to_display):
        plt.subplot(rows, num_samples, i + 1)
        # 反标准化图像
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    
    plt.tight_layout()
    return plt

def count_samples_per_class(dataset, class_names):
    """
    统计每个类别的样本数量
    参数:
        dataset: 数据集
        class_names: 类别名称列表
    返回:
        counts: 每个类别的样本数量字典
    """
    counts = {class_name: 0 for class_name in class_names}
    
    for _, label in dataset:
        class_name = class_names[label]
        counts[class_name] += 1
    
    return counts

def split_train_val(train_dataset, val_split=0.2):
    """
    将训练集分割为训练集和验证集
    参数:
        train_dataset: 原始训练数据集
        val_split: 验证集占比
    返回:
        train_subset: 分割后的训练集
        val_subset: 验证集
    """
    # 计算验证集大小
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    # 随机分割数据集
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    return train_subset, val_subset

def load_image(image_path):
    """
    加载单张图像并进行预处理
    参数:
        image_path: 图像路径
    返回:
        image_tensor: 预处理后的图像张量
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 打开并处理图像
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    
    return image_tensor

def save_dataset_statistics(train_dataset, test_dataset, class_names, output_dir):
    """
    保存数据集统计信息
    参数:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计训练集每个类别的样本数
    train_counts = count_samples_per_class(train_dataset, class_names)
    # 统计测试集每个类别的样本数
    test_counts = count_samples_per_class(test_dataset, class_names)
    
    # 生成统计信息文本
    stats_text = "数据集统计信息\n\n"
    stats_text += f"训练集总样本数: {len(train_dataset)}\n"
    stats_text += f"测试集总样本数: {len(test_dataset)}\n"
    stats_text += f"类别数量: {len(class_names)}\n\n"
    
    stats_text += "训练集类别分布:\n"
    for class_name in class_names:
        stats_text += f"  {class_name}: {train_counts[class_name]}\n"
    
    stats_text += "\n测试集类别分布:\n"
    for class_name in class_names:
        stats_text += f"  {class_name}: {test_counts[class_name]}\n"
    
    # 保存统计信息到文件
    with open(os.path.join(output_dir, 'dataset_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(stats_text)
    
    # 可视化类别分布
    plt.figure(figsize=(12, 6))
    
    # 训练集分布
    plt.subplot(1, 2, 1)
    plt.bar(class_names, [train_counts[cn] for cn in class_names])
    plt.title('训练集类别分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 测试集分布
    plt.subplot(1, 2, 2)
    plt.bar(class_names, [test_counts[cn] for cn in class_names])
    plt.title('测试集类别分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存分布图
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    return stats_text
