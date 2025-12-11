from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64
import json
import datetime
import threading
from queue import Queue
from models.cnn_models import SimpleCNN, ImprovedCNN, AdvancedCNN
from utils.data_utils import load_dataset, create_dataloaders
from utils.data_utils import save_dataset_statistics
from utils.prepare_coil20 import prepare_coil20_dataset
from utils.train_utils import train_model, evaluate_model, plot_results, load_model, get_confusion_matrix, plot_confusion_matrix

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/samples', exist_ok=True)

# 禁用 Flask/Werkzeug 的 HTTP 访问日志（仅影响请求访问日志）
# 将 werkzeug 的日志级别提升为 ERROR，避免常规的 INFO 请求日志打印
try:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # 也将 Flask 自身的 logger 设置为 WARNING，以减少输出
    logging.getLogger('flask.app').setLevel(logging.WARNING)
except Exception:
    pass

# 全局变量存储当前模型和参数
current_model = None
model_params = None
train_results = None
class_names = None

# 运行设备（在多个地方可能会使用到）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 异步任务状态：COIL-20 准备
coil20_jobs = {}


# 构建对齐至目标类别集的测试数据加载器：仅保留目标类别，并将标签重映射到 [0..N-1]
def build_aligned_test_loader(test_dataset, target_class_names, batch_size):
    """根据目标类别列表，过滤测试集为该子集并将标签重映射到 [0..N-1]。
    这样可避免 CrossEntropyLoss 在模型输出维度与数据集类别不一致时触发设备端断言。
    """
    if not target_class_names or len(target_class_names) == 0:
        raise ValueError('目标类别列表为空，无法构建测试集加载器')

    allowed_names = set(target_class_names)
    orig_class_names = getattr(test_dataset, 'classes', None)
    if orig_class_names is None:
        raise ValueError('测试数据集缺少 classes 属性，无法进行类别过滤')

    name_to_target_idx = {name: idx for idx, name in enumerate(target_class_names)}

    # ImageFolder 提供 samples/imgs 与 targets；优先使用 samples 加速过滤
    samples = getattr(test_dataset, 'samples', None)
    if samples is None:
        # 兜底：基于 targets/imgs 构建索引
        imgs = getattr(test_dataset, 'imgs', [])
        targets = getattr(test_dataset, 'targets', [])
        indices = [i for i, t in enumerate(targets) if orig_class_names[t] in allowed_names]
    else:
        indices = [i for i, (_, label) in enumerate(samples) if orig_class_names[label] in allowed_names]

    # 自定义数据集包装：提供重映射后的标签
    import torch as _torch
    from torch.utils.data import Dataset as _Dataset

    class RemappedSubset(_Dataset):
        def __init__(self, base_dataset, sel_indices, orig_names, name_to_target):
            self.base = base_dataset
            self.indices = sel_indices
            self.orig_names = orig_names
            self.name_to_target = name_to_target

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            base_idx = self.indices[idx]
            img, orig_label = self.base[base_idx]
            orig_name = self.orig_names[orig_label]
            # 安全映射到目标标签空间
            if orig_name not in self.name_to_target:
                # 正常不会发生（已过滤），但为健壮性提供兜底
                target_label = 0
            else:
                target_label = self.name_to_target[orig_name]
            return img, _torch.tensor(target_label, dtype=_torch.long)

    subset = RemappedSubset(test_dataset, indices, orig_class_names, name_to_target_idx)

    # 构建 DataLoader，确保在 CUDA 环境下启用 pin_memory
    from torch.utils.data import DataLoader as _DataLoader
    _device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
    test_loader = _DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if _device.type == 'cuda' else False,
    )

    return test_loader


def create_model_instance(model_type, num_classes):
    """根据 model_type 创建模型实例的简单工厂函数"""
    if model_type == 'simple':
        return SimpleCNN(num_classes=num_classes)
    elif model_type == 'improved':
        return ImprovedCNN(num_classes=num_classes)
    else:
        return AdvancedCNN(num_classes=num_classes)

# 训练状态跟踪
training_in_progress = False
training_status = {
    'current_epoch': 0,
    'total_epochs': 0,
    'current_batch': 0,
    'total_batches': 0,
    'train_loss': 0.0,
    'train_accuracy': 0.0,
    'test_loss': 0.0,
    'test_accuracy': 0.0,
    'status': 'idle',  # idle, training, completed, error, stopped
    'message': ''
}
train_logs = []
training_thread = None
training_lock = threading.Lock()
stop_training_flag = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        # 获取表单数据
        model_type = request.form.get('model_type', 'simple')
        batch_size = int(request.form.get('batch_size', 32))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        epochs = int(request.form.get('epochs', 500))
        optimizer_type = request.form.get('optimizer', 'adam')
        # 数据增强相关参数
        enable_random_grayscale = bool(request.form.get('enable_random_grayscale'))
        try:
            random_grayscale_p = float(request.form.get('random_grayscale_p', 0.2))
        except Exception:
            random_grayscale_p = 0.2
        
        # 保存模型参数
        global model_params
        model_params = {
            'model_type': model_type,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'optimizer_type': optimizer_type,
            'enable_random_grayscale': enable_random_grayscale,
            'random_grayscale_p': random_grayscale_p
        }
        # 立即写入静态参数文件，供数据加载阶段读取
        try:
            with open('static/model_params.json', 'w', encoding='utf-8') as f:
                json.dump(model_params, f)
        except Exception as e:
            print(f"保存模型参数到 static/model_params.json 失败: {e}")
        
        flash('模型配置已保存！请继续到训练页面进行训练。')
        return redirect(url_for('train'))
    
    return render_template('config.html')

# 自定义训练回调函数来更新状态
def update_training_status(epoch, epochs, batch, total_batches, loss, accuracy, test_loss=None, test_acc=None, status='training', message=''):
    global training_status, train_logs
    with training_lock:
        training_status.update({
            'current_epoch': epoch,
            'total_epochs': epochs,
            'current_batch': batch,
            'total_batches': total_batches,
            'train_loss': float(loss),
            'train_accuracy': float(accuracy),
            'test_loss': float(test_loss) if test_loss is not None else 0.0,
            'test_accuracy': float(test_acc) if test_acc is not None else 0.0,
            'status': status,
            'message': message
        })
        
        # 添加日志
        log_message = f'Epoch {epoch}/{epochs}, Batch {batch}/{total_batches}, Loss: {loss:.4f}, Acc: {accuracy:.2f}%'
        if test_loss is not None:
            log_message += f', Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%'
        train_logs.append(log_message)
        
        # 在终端中打印日志
        print(log_message)
        
        # 如果有message，也在终端打印
        if message:
            print(f'[状态] {message}')
        
        # 限制日志数量
        if len(train_logs) > 1000:
            train_logs = train_logs[-1000:]

def save_interrupted_model(model, current_epoch, train_losses, test_losses, train_accuracies, test_accuracies):
    """保存中断训练时的模型和结果"""
    global current_model, model_params, train_results, class_names
    
    try:
        # 保存当前训练结果
        train_results = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        
        # 创建当前模型的新实例并加载状态
        num_classes = len(class_names) if class_names else 10  # 默认值以防class_names未定义
        if model_params['model_type'] == 'simple':
            current_model = SimpleCNN(num_classes=num_classes)
        elif model_params['model_type'] == 'improved':
            current_model = ImprovedCNN(num_classes=num_classes)
        else:  # advanced
            current_model = AdvancedCNN(num_classes=num_classes)
        
        # 将模型移至设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_model.to(device)
        
        # 深拷贝状态字典
        state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
        current_model.load_state_dict(state_dict)
        
        # 使用模型包方式保存中断模型，包含训练结果和类别信息
        interrupted_results = {
            'train_losses': train_results['train_losses'],
            'test_losses': train_results['test_losses'],
            'train_accuracies': train_results['train_accuracies'],
            'test_accuracies': train_results['test_accuracies'],
            'stopped_epoch': current_epoch + 1,
            'class_names': class_names
        }

        try:
            save_model_bundle(model, model_params if model_params else {}, interrupted_results, class_names, name_prefix='interrupted')
            train_logs.append(f'已保存中断训练的模型包（第{current_epoch+1}轮）')
        except Exception as e:
            # 回退到老路径
            torch.save(model.state_dict(), 'static/interrupted_model.pth')
            with open('static/interrupted_results.json', 'w') as f:
                json.dump(interrupted_results, f)
            train_logs.append(f'已保存中断训练时的模型（第{current_epoch+1}轮） (回退保存)')
    except Exception as e:
        error_msg = f'保存中断模型时出错: {str(e)}'
        train_logs.append(error_msg)
        print(error_msg)

def resume_training_thread():
    """
    从中断的模型继续训练
    """
    global training_in_progress, stop_training_flag, training_status, current_model, model_params, train_results, class_names
    
    try:
        # 确保中断模型文件存在
        if not os.path.exists('static/interrupted_model.pth') or not os.path.exists('static/interrupted_results.json'):
            update_training_status(0, 0, 0, 0, 0.0, 0.0, status='error', message='未找到中断的模型文件')
            return
        
        # 加载中断的模型状态
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('static/interrupted_model.pth', map_location=device)
        
        # 加载训练结果
        with open('static/interrupted_results.json', 'r') as f:
            results = json.load(f)
        
        train_losses = results['train_losses']
        test_losses = results['test_losses']
        train_accuracies = results['train_accuracies']
        test_accuracies = results['test_accuracies']
        start_epoch = results['stopped_epoch']
        
        # 更新全局训练结果
        train_results = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        
        # 创建数据加载器
        update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message='加载数据集...')
        train_loader, test_loader, class_names, _ = create_data_loaders(
            model_params['batch_size'], 
            model_params['image_size'],
            model_params['data_dir']
        )
        
        # 加载模型
        update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message='加载中断的模型...')
        num_classes = len(class_names) if class_names else 10
        if model_params['model_type'] == 'simple':
            current_model = SimpleCNN(num_classes=num_classes)
        elif model_params['model_type'] == 'improved':
            current_model = ImprovedCNN(num_classes=num_classes)
        else:  # advanced
            current_model = AdvancedCNN(num_classes=num_classes)
        
        # 加载模型状态
        current_model.to(device)
        current_model.load_state_dict(checkpoint)
        
        # 创建优化器和损失函数
        if model_params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(current_model.parameters(), lr=model_params['learning_rate'], 
                                        weight_decay=model_params['weight_decay'] if 'weight_decay' in model_params else 0)
        else:
            optimizer = torch.optim.SGD(current_model.parameters(), lr=model_params['learning_rate'], 
                                      momentum=0.9, 
                                      weight_decay=model_params['weight_decay'] if 'weight_decay' in model_params else 0)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # 开始继续训练
        update_training_status(start_epoch, model_params['epochs'], 0, 0, 
                             train_losses[-1] if train_losses else 0.0, 
                             test_losses[-1] if test_losses else 0.0, 
                             status='training', 
                             message=f'从第{start_epoch}轮继续训练...')
        
        for epoch in range(start_epoch, model_params['epochs']):
            # 检查是否需要停止训练
            if stop_training_flag:
                update_training_status(epoch, model_params['epochs'], 0, 0, 0.0, 0.0, 
                                     status='stopped', message='训练已被用户中断')
                save_interrupted_model(current_model, epoch, train_losses, test_losses, train_accuracies, test_accuracies)
                print(f"训练在epoch {epoch} 被中断")
                break
            
            # 训练模式
            current_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 更新训练状态
            update_training_status(epoch, model_params['epochs'], 0, 0, 
                                 train_losses[-1] if train_losses else 0.0, 
                                 test_losses[-1] if test_losses else 0.0,
                                 status='training', 
                                 message=f'第 {epoch+1}/{model_params["epochs"]} 轮训练中...')
            
            # 遍历训练数据
            for i, (inputs, labels) in enumerate(train_loader):
                # 检查是否需要停止训练
                if stop_training_flag:
                    update_training_status(epoch, model_params['epochs'], 0, 0, 0.0, 0.0, 
                                         status='stopped', message='训练已被用户中断')
                    save_interrupted_model(current_model, epoch, train_losses, test_losses, train_accuracies, test_accuracies)
                    print(f"训练在epoch {epoch}, batch {i} 被中断")
                    break
                
                # 更新批处理状态
                progress = int((i + 1) / len(train_loader) * 100)
                update_training_status(epoch, model_params['epochs'], progress, i + 1,
                                     train_losses[-1] if train_losses else 0.0,
                                     test_losses[-1] if test_losses else 0.0)
                
                # 将数据移至设备
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = current_model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=0.3)
                
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # 检查是否在batch循环中被中断
            if stop_training_flag:
                break
            
            # 计算训练指标
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 验证
            current_model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = current_model(inputs)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            # 计算验证指标
            test_loss /= len(test_loader)
            test_acc = 100. * test_correct / test_total
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            # 更新训练状态
            update_training_status(epoch, model_params['epochs'], 100, len(train_loader),
                                 train_loss, test_loss, train_acc, test_acc,
                                 status='training', 
                                 message=f'第 {epoch+1}/{model_params["epochs"]} 轮完成')
            
            # 保存最新结果
            train_results = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }
            
            # 打印训练信息
            print(f'Epoch [{epoch+1}/{model_params["epochs"]}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # 如果训练正常完成
        if not stop_training_flag:
            # 保存最终模型
            update_training_status(epoch, model_params['epochs'], 100, len(train_loader),
                                 train_loss, test_loss, train_acc, test_acc,
                                 status='training', message='保存最终模型...')
            
            # 保存模型
            torch.save(current_model.state_dict(), 'static/cnn_model.pth')
            
            # 保存训练结果
            with open('static/train_results.json', 'w') as f:
                json.dump({
                    'train_losses': train_results['train_losses'],
                    'test_losses': train_results['test_losses'],
                    'train_accuracies': train_results['train_accuracies'],
                    'test_accuracies': train_results['test_accuracies'],
                    'total_epochs': len(train_results['train_losses']),
                    'class_names': class_names
                }, f)
            
            update_training_status(epoch, model_params['epochs'], 100, len(train_loader),
                                 train_loss, test_loss, train_acc, test_acc,
                                 status='completed', message='训练已完成')
            
    except Exception as e:
        print(f"继续训练时出错: {str(e)}")
        update_training_status(0, 0, 0, 0, 0.0, 0.0, 
                             status='error', message=f'训练出错: {str(e)}')
        
    finally:
        # 确保训练标志被重置
        with training_lock:
            training_in_progress = False
        stop_training_flag = False
        
        # 清理CUDA缓存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def train_model_thread():
    global current_model, model_params, train_results, class_names, training_in_progress, stop_training_flag
    
    try:
        # 重置训练状态和日志
        update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message='开始训练...')
        
        # 清理可能导致BatchNorm参数冲突的所有旧模型相关文件
        old_files = [
            'static/model_params.json', 
            'static/train_results.json',
            'static/cnn_model.pth',
            'static/best_model.pth',
            'cnn_model.pth',
            'best_model.pth',
            'model_params.json',
            'train_results.json'
        ]
        
        # 彻底清理所有可能的旧模型文件
        for old_file in old_files:
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message=f'清理旧文件: {os.path.basename(old_file)}')
                except Exception as e:
                    update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message=f'清理旧文件{os.path.basename(old_file)}时出错: {str(e)}')
        
        # 添加显式的CUDA内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message='清理CUDA缓存')
        
        # 加载数据集
        train_dataset, test_dataset, class_names = load_dataset(app.config['DATA_FOLDER'])
        train_loader, test_loader = create_dataloaders(
            train_dataset, test_dataset, model_params['batch_size']
        )
        
        # 根据选择创建模型
        num_classes = len(class_names)
        if model_params['model_type'] == 'simple':
            model = SimpleCNN(num_classes=num_classes)
        elif model_params['model_type'] == 'improved':
            model = ImprovedCNN(num_classes=num_classes)
        else:  # advanced
            model = AdvancedCNN(num_classes=num_classes)
        
        # 初始化后检查模型权重
        print(f"模型类型: {model_params['model_type']}, 类别数: {num_classes}")
        print("检查模型初始权重分布:")
        
        # 对所有模型应用统一的权重初始化改进 - 使用更保守的初始化参数
        print("应用改进的权重初始化策略...")
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # 使用更保守的Kaiming初始化参数
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 使用更小的标准差，防止梯度爆炸
                nn.init.normal_(m.weight, mean=0.0, std=0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        weight_stats = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_stats.append({
                    'name': name,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                })
                # 只打印前5个参数的统计信息
                if len(weight_stats) <= 5:
                    print(f"  {name}: 均值={param.data.mean():.6f}, 标准差={param.data.std():.6f}, 范围=[{param.data.min():.6f}, {param.data.max():.6f}]")
        
        # 检查是否有异常权重值
        has_abnormal_weights = False
        for stat in weight_stats:
            if abs(stat['max']) > 100 or abs(stat['min']) > 100:
                print(f"警告: {stat['name']} 包含异常大的权重值: [{stat['min']:.2f}, {stat['max']:.2f}]")
                has_abnormal_weights = True
        if has_abnormal_weights:
            print("注意: 检测到异常权重值，可能需要进一步调整初始化策略")
        
        # 确保全局变量正确重置
        global current_model
        current_model = None
        
        # 验证模型结构，确保BatchNorm层参数正确初始化
        def validate_model_batchnorm(model):
            """验证模型中所有BatchNorm层的参数一致性，支持复杂模型结构"""
            print("开始验证模型BatchNorm层...")
            device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device('cpu')
            batchnorm_count = 0
            fixed_count = 0
            
            for name, module in model.named_modules():
                # 支持所有BatchNorm类型
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    batchnorm_count += 1
                    try:
                        # 确保running_mean和running_var存在并大小正确
                        if hasattr(module, 'running_mean') and hasattr(module, 'running_var') and hasattr(module, 'num_features'):
                            # 检查running_mean大小
                            if module.running_mean is not None:
                                if module.running_mean.size(0) != module.num_features:
                                    print(f"警告: BatchNorm层 {name} 的running_mean大小不匹配: 期望 {module.num_features}, 实际 {module.running_mean.size(0)}")
                                    # 自动修复
                                    module.running_mean = torch.zeros(module.num_features, device=device)
                                    fixed_count += 1
                            else:
                                # 如果running_mean为None，初始化它
                                module.running_mean = torch.zeros(module.num_features, device=device)
                                fixed_count += 1
                                
                            # 检查running_var大小
                            if module.running_var is not None:
                                if module.running_var.size(0) != module.num_features:
                                    print(f"警告: BatchNorm层 {name} 的running_var大小不匹配: 期望 {module.num_features}, 实际 {module.running_var.size(0)}")
                                    # 自动修复
                                    module.running_var = torch.ones(module.num_features, device=device)
                                    fixed_count += 1
                            else:
                                # 如果running_var为None，初始化它
                                module.running_var = torch.ones(module.num_features, device=device)
                                fixed_count += 1
                            
                            # 确保num_batches_tracked存在并正确初始化
                            if hasattr(module, 'num_batches_tracked'):
                                module.num_batches_tracked = torch.tensor(0, device=device)
                            
                            # 确保eps和momentum参数合理
                            if hasattr(module, 'eps') and module.eps <= 0:
                                module.eps = 1e-5
                            if hasattr(module, 'momentum') and (module.momentum is None or module.momentum < 0 or module.momentum > 1):
                                module.momentum = 0.1
                                
                        print(f"BatchNorm层 {name} 验证通过")
                    except Exception as e:
                        print(f"验证BatchNorm层 {name} 时出错: {str(e)}")
                        # 尝试完全重置该层
                        try:
                            torch.nn.init.zeros_(module.running_mean)
                            torch.nn.init.ones_(module.running_var)
                            if hasattr(module, 'num_batches_tracked'):
                                module.num_batches_tracked.zero_()
                            print(f"已尝试重置BatchNorm层 {name}")
                        except:
                            print(f"无法重置BatchNorm层 {name}")
            
            print(f"模型BatchNorm层验证完成: 共检查 {batchnorm_count} 层, 修复 {fixed_count} 层")
            
            # 额外的模型结构验证
            try:
                # 运行一个足够大的批量数据通过模型以验证尺寸兼容性
                # 使用更大的batch size以避免BatchNorm层在训练模式下的问题
                dummy_batch_size = 4  # 增加到4以避免BatchNorm训练模式错误
                dummy_input = torch.randn(dummy_batch_size, 3, 224, 224, device=device)
                # 临时将模型设置为评估模式以避免BatchNorm训练模式的batch size限制
                model.eval()
                with torch.no_grad():
                    _ = model(dummy_input)
                # 恢复为训练模式
                model.train()
                print("模型前向传播测试通过")
            except Exception as e:
                print(f"警告: 模型前向传播测试失败: {str(e)}")
                # 不抛出异常，允许继续训练，但提供警告
            
            return True
        
        # 验证模型结构
        try:
            validate_model_batchnorm(model)
            update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message='模型结构验证通过')
        except Exception as e:
            update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='error', message=f'模型结构验证失败: {str(e)}')
            raise
        
        # 根据模型类型选择不同的训练参数 - 调整为更合理的参数设置以确保训练收敛
        print("应用优化的训练参数设置以确保良好的收敛性...")
        
        # 提高学习率以确保训练能够有效进行
        if model_params['model_type'] == 'simple':
            weight_decay = 0.0001  # 使用标准权重衰减
            learning_rate = model_params['learning_rate'] * 0.5  # 调整为更合理的学习率
        elif model_params['model_type'] == 'improved':
            weight_decay = 0.0001  # 标准权重衰减
            learning_rate = model_params['learning_rate'] * 1.0  # 大幅提高学习率以促进更好的收敛
        else:  # advanced
            weight_decay = 0.00005  # 适当降低权重衰减
            learning_rate = model_params['learning_rate'] * 0.5  # 提高高级模型的学习率以促进收敛
        
        # 创建优化器 - 添加权重衰减
        if model_params['optimizer_type'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                                 momentum=0.9, weight_decay=weight_decay, 
                                 nesterov=True)  # 使用Nesterov动量
        else:  # adam
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                                  weight_decay=weight_decay)  # Adam优化器添加L2正则化
        
        # 定义学习率调度器 - 优化版本，更适合长时间训练
        if model_params['model_type'] == 'simple':
            # 简单模型也使用余弦退火策略，更有效地探索参数空间
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
        elif model_params['model_type'] == 'improved':
            # 改进模型使用带有重启的余弦退火，有助于跳出局部最小值
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        else:  # advanced
            # 高级模型使用改进的预热+余弦退火带重启
            warmup_epochs = 10  # 增加预热轮数
            if model_params['epochs'] > warmup_epochs:
                # 分段学习率策略
                def lr_lambda(epoch):
                    # 预热阶段
                    if epoch < warmup_epochs:
                        return 0.1 + 0.9 * (epoch / warmup_epochs)  # 平滑过渡到基准学习率
                    # 预热后使用改进的余弦退火（带有小的周期性波动）
                    else:
                        main_epoch = epoch - warmup_epochs
                        # 使用多个余弦周期以鼓励探索
                        cycle_length = 100
                        cycle_position = main_epoch % cycle_length
                        progress = cycle_position / cycle_length
                        # 余弦退火基础上添加小的重启
                        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            else:
                # 如果训练轮次太少，使用余弦退火带重启
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 确定设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        update_training_status(0, model_params['epochs'], 0, 0, 0.0, 0.0, status='training', message=f'使用设备: {device}')
        
        # 将模型移至设备
        # 使用with torch.no_grad()减少内存使用
        with torch.no_grad():
            model.to(device)
        
        # 初始化记录列表
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        # 早停机制参数 - 优化版本
        patience = 2000  # 容忍多少轮没有改进
        min_improvement = 0.001  # 最小改进阈值，只有超过这个值才被视为有改进
        best_val_acc = 0.0
        best_epoch = 0  # 记录最佳模型的epoch
        counter = 0
        best_model_state = None
        
        # 开始训练
        for epoch in range(model_params['epochs']):
            # 检查是否需要停止训练
            if stop_training_flag:
                # 保存当前模型状态和训练结果
                save_interrupted_model(model, epoch, train_losses, test_losses, train_accuracies, test_accuracies)
                update_training_status(epoch+1, model_params['epochs'], batch_idx+1, len(train_loader),
                                      running_loss / total if total > 0 else 0.0,
                                      100. * correct / total if total > 0 else 0.0,
                                      status='stopped', message='训练已停止，模型已保存')
                train_logs.append('训练过程被用户中断，已保存当前模型')
                stop_training_flag = False
                break
                
            update_training_status(epoch+1, model_params['epochs'], 0, len(train_loader), 0.0, 0.0, status='training', 
                                  message=f'正在进行第 {epoch+1}/{model_params["epochs"]} 轮训练...')
            
            # 训练模式
            model.train()
            
            # 优化器参数检查 - 仅在第一个epoch打印
            if epoch == 0:
                initial_lr = optimizer.param_groups[0]['lr']
                print("\n======= 训练配置信息 =======")
                print(f"开始训练 - 初始学习率: {initial_lr:.6f}, 权重衰减: {weight_decay}")
                print(f"数据集信息 - 训练样本数: {len(train_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}, 类别数: {num_classes}")
                print(f"模型配置 - 类型: {model_params['model_type']}, 批量大小: {model_params['batch_size']}, 轮数: {model_params['epochs']}")
                print(f"训练策略 - 梯度裁剪阈值: 0.5, 学习率调度器: {scheduler.__class__.__name__}")
                print(f"权重初始化 - Kaiming初始化(卷积层) + 小方差Normal初始化(全连接层)")
                print(f"数据预处理 - 224x224尺寸 + ImageNet标准化")
                print("========================\n")
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 训练循环
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # 检查是否需要停止训练
                if stop_training_flag:
                    # 保存当前模型状态和训练结果
                    save_interrupted_model(model, epoch, train_losses, test_losses, train_accuracies, test_accuracies)
                    update_training_status(epoch+1, model_params['epochs'], batch_idx+1, len(train_loader),
                                          running_loss / total if total > 0 else 0.0,
                                          100. * correct / total if total > 0 else 0.0,
                                          status='stopped', message='训练已停止，模型已保存')
                    train_logs.append('训练过程被用户中断，已保存当前模型')
                    stop_training_flag = False
                    break
                
                # 将数据移至设备
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播 - 添加调试信息
                try:
                    # 打印输入特征图尺寸（仅在第一轮的第一个批次打印）
                    if epoch == 0 and batch_idx == 0:
                        print(f"输入特征图尺寸: {inputs.shape}")
                        
                    # 包装模型以捕获中间特征图尺寸和激活值统计
                    def debug_forward_hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            # 仅在第一个epoch的前3个batch详细监控激活值
                            if epoch == 0 and batch_idx < 3:
                                module_name = module.__class__.__name__
                                output_min = output.min().item()
                                output_max = output.max().item()
                                output_mean = output.mean().item()
                                output_std = output.std().item()
                                
                                # 检查激活值是否异常
                                is_abnormal = False
                                if output.abs().max().item() > 100:
                                    is_abnormal = True
                                if output_std > 100:
                                    is_abnormal = True
                                if torch.isnan(output).any() or torch.isinf(output).any():
                                    is_abnormal = True
                                
                                # 打印层信息和激活值统计
                                print(f"层 {module_name} 输出信息:")
                                print(f"  尺寸: {output.shape}")
                                print(f"  激活值统计 - 均值: {output_mean:.6f}, 标准差: {output_std:.6f}, 范围: [{output_min:.6f}, {output_max:.6f}]")
                                
                                # 如果异常，详细记录
                                if is_abnormal:
                                    print(f"  警告: {module_name} 激活值异常!")
                            else:
                                # 其他情况只打印尺寸
                                print(f"层 {module.__class__.__name__} 输出尺寸: {output.shape}")
                        return output
                    
                    # 仅在第一轮的第一个批次添加钩子
                    hooks = []
                    if epoch == 0 and batch_idx == 0:
                        for i, (name, module) in enumerate(model.named_modules()):
                            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                                hooks.append(module.register_forward_hook(debug_forward_hook))
                    
                    outputs = model(inputs)
                    
                    # 移除钩子
                    for hook in hooks:
                        hook.remove()
                    
                    # 检查输出层激活值分布
                    if epoch == 0 and batch_idx < 3:
                        print(f"\nBatch {batch_idx} 输出层统计:")
                        print(f"  输出最小值: {outputs.min().item():.6f}")
                        print(f"  输出最大值: {outputs.max().item():.6f}")
                        print(f"  输出绝对值最大值: {outputs.abs().max().item():.6f}")
                        print(f"  输出NaN数量: {torch.isnan(outputs).sum().item()}")
                        print(f"  输出无穷大数量: {torch.isinf(outputs).sum().item()}")
                    
                    # 打印输出尺寸（仅在第一轮的第一个批次打印）
                    if epoch == 0 and batch_idx == 0:
                        print(f"模型输出尺寸: {outputs.shape}")
                        print(f"标签尺寸: {labels.shape}")
                    
                    # 计算损失前进行更详细的输出检查
                    if batch_idx < 5 or (epoch < 5 and batch_idx % 2 == 0):
                        print(f"\n详细调试 - Epoch {epoch}, Batch {batch_idx}:")
                        # 检查输出分布
                        outputs_mean = outputs.mean().item()
                        outputs_std = outputs.std().item()
                        outputs_min = outputs.min().item()
                        outputs_max = outputs.max().item()
                        print(f"  输出统计 - 均值: {outputs_mean:.6f}, 标准差: {outputs_std:.6f}, 范围: [{outputs_min:.6f}, {outputs_max:.6f}]")
                        
                        # 检查标签分布
                        print(f"  标签统计 - 范围: [{labels.min().item()}, {labels.max().item()}], 批次大小: {labels.size(0)}")
                        
                        # 检查模型参数是否异常
                        if batch_idx == 0:
                            param_norms = []
                            for name, param in model.named_parameters():
                                if 'fc' in name and param.requires_grad:
                                    param_norm = param.data.norm().item()
                                    param_norms.append(param_norm)
                                    print(f"  {name} 参数范数: {param_norm:.6f}")
                                    if param_norm > 100:
                                        print(f"  警告: {name} 参数范数过大")
                    
                    # 计算损失前再次检查输出是否异常
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"\n严重警告: 前向传播输出异常 - Epoch {epoch}, Batch {batch_idx}")
                        print(f"  输出NaN数量: {torch.isnan(outputs).sum().item()}")
                        print(f"  输出无穷大数量: {torch.isinf(outputs).sum().item()}")
                        # 限制输出范围以防止损失计算异常
                        outputs = torch.clamp(outputs, min=-100, max=100)
                    
                    # 计算损失 - 使用梯度裁剪的方式计算损失
                    try:
                        # 对所有模型类型使用更稳定的损失计算方式
                        loss = criterion(outputs, labels)
                        
                        # 限制损失值范围
                        if loss.item() > 50.0:
                            # 如果损失值过大，使用裁剪后的输出重新计算
                            print(f"\n警告: 损失值过大，使用裁剪后的输出重新计算 - Epoch {epoch}, Batch {batch_idx}")
                            clipped_outputs = torch.clamp(outputs, min=-5, max=5)
                            loss = criterion(clipped_outputs, labels)
                            print(f"  裁剪后的损失值: {loss.item():.6f}")
                    except Exception as e:
                        print(f"\n错误: 损失计算异常 - Epoch {epoch}, Batch {batch_idx}, 错误信息: {str(e)}")
                        # 使用安全的损失值
                        loss = torch.tensor(1.0, device=device)
                    
                    # 立即检查损失值
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"\n严重错误: 损失值异常 - Epoch {epoch}, Batch {batch_idx}")
                        print(f"  损失值: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
                        print(f"  输出统计: 最小值={outputs.min().item():.6f}, 最大值={outputs.max().item():.6f}")
                        # 保存当前批次数据用于进一步分析
                        torch.save({'inputs': inputs, 'outputs': outputs, 'labels': labels}, f'debug_batch_{epoch}_{batch_idx}.pt')
                        # 设置安全的损失值以继续训练
                        loss = torch.tensor(1.0, device=device)
                    
                    # 检测0损失异常 - 使用更严格的检查
                    if loss.item() < 0.001:
                        print(f"\n警告: 检测到极低损失 - Epoch {epoch}, Batch {batch_idx}, 损失值: {loss.item():.6f}")
                        # 计算预测结果
                        _, predicted = torch.max(outputs, 1)
                        correct = (predicted == labels).sum().item()
                        print(f"  该批次准确率: {correct / labels.size(0) * 100:.2f}%")
                        print(f"  前10个预测: {predicted[:10]}")
                        print(f"  前10个标签: {labels[:10]}")
                        
                        # 检查是否所有预测都相同
                        if predicted.unique().numel() == 1:
                            print(f"  警告: 所有预测都相同: {predicted[0].item()}")
                            # 添加小的随机噪声到输出以打破这种状态
                            print("  添加随机噪声以打破异常预测状态")
                            with torch.no_grad():
                                noise = torch.randn_like(outputs) * 0.01
                                outputs = outputs + noise
                    
                    # 检测异常高损失
                    elif loss.item() > 10.0:
                        print(f"\n警告: 检测到高损失 - Epoch {epoch}, Batch {batch_idx}, 损失值: {loss.item():.6f}")
                        # 检查是否输出包含异常值
                        if outputs.abs().max().item() > 10.0:
                            print(f"  输出包含异常大的值: {outputs.abs().max().item():.6f}")
                        # 检查标签是否正确
                        if labels.min().item() < 0 or labels.max().item() >= num_classes:
                            print(f"  警告: 标签值超出有效范围: [{labels.min().item()}, {labels.max().item()}]")
                except RuntimeError as e:
                    error_msg = str(e)
                    print(f"前向传播错误: {error_msg}")
                    # 如果是尺寸不匹配错误，添加详细调试信息
                    if 'size mismatch' in error_msg or 'not match' in error_msg:
                        print(f"详细错误信息:")
                        print(f"  - 批次索引: {batch_idx}")
                        print(f"  - 输入尺寸: {inputs.shape}")
                        print(f"  - 标签尺寸: {labels.shape}")
                        try:
                            # 尝试打印模型结构
                            print("模型结构:")
                            print(model)
                        except:
                            print("无法打印完整模型结构")
                    raise
                
                # 反向传播
                loss.backward()
                
                # 对所有模型类型应用梯度裁剪以防止梯度爆炸（使用更严格的裁剪值）
                # 进一步降低裁剪阈值以确保所有模型的训练稳定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                
                # 每次迭代都检查梯度是否异常，不仅限于特定批次
                grad_norm = 0.0
                has_inf_grad = False
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isinf(param.grad).any():
                            has_inf_grad = True
                        if torch.isnan(param.grad).any():
                            has_nan_grad = True
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # 收集额外的梯度统计信息
                max_grad = -float('inf')
                min_grad = float('inf')
                for param in model.parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.data.max().item())
                        min_grad = min(min_grad, param.grad.data.min().item())
                
                # 如果检测到异常梯度，立即打印警告
                if has_inf_grad or has_nan_grad or grad_norm > 100:
                    print(f"警告: Epoch {epoch}, Batch {batch_idx} 检测到异常梯度!")
                    if has_inf_grad:
                        print("  - 梯度包含无穷大值")
                    if has_nan_grad:
                        print("  - 梯度包含NaN值")
                    print(f"  - 梯度范数: {grad_norm:.6f}")
                    print(f"  - 梯度范围: [{min_grad:.6f}, {max_grad:.6f}]")
                
                # 定期详细监控梯度（每5个epoch的第一个batch）
                if batch_idx == 0 and epoch % 5 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx} 梯度监控:")
                    print(f"  梯度范数: {grad_norm:.6f}")
                    print(f"  梯度最大值: {max_grad:.6f}")
                    print(f"  梯度最小值: {min_grad:.6f}")
                
                # 在第一个epoch监控梯度稳定性
                if epoch == 0 and batch_idx < 3:
                    for name, param in model.named_parameters():
                        if param.grad is not None and 'fc' in name:  # 只检查全连接层
                            print(f"{name} 梯度统计:")
                            print(f"  均值: {param.grad.data.mean():.6f}")
                            print(f"  标准差: {param.grad.data.std():.6f}")
                            print(f"  范围: [{param.grad.data.min():.6f}, {param.grad.data.max():.6f}]")
                            break  # 只打印第一个全连接层的梯度
                    print(f"Epoch {epoch}, Batch 0 - 梯度范数: {grad_norm:.6f}")
                
                # 记录参数更新前的某些关键权重（仅在epoch 0的前几个batch）
                if epoch == 0 and batch_idx == 0:
                    param_before = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad and 'fc' in name}
                
                # 优化器更新
                optimizer.step()
                
                # 检查参数更新后的变化（仅在epoch 0的第一个batch）
                if epoch == 0 and batch_idx == 0:
                    print("参数更新前后变化:")
                    for name, p in model.named_parameters():
                        if p.requires_grad and 'fc' in name:
                            before = param_before.get(name)
                            if before is not None:
                                change = (p.data - before).abs().mean().item()
                                print(f"  {name}: 平均变化={change:.6f}, 更新前均值={before.mean():.6f}, 更新后均值={p.data.mean():.6f}")
                
                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 在计算完损失和准确率后更新批次进度
                update_training_status(epoch+1, model_params['epochs'], batch_idx+1, len(train_loader),
                                      running_loss / total if total > 0 else 0.0,
                                      100. * correct / total if total > 0 else 0.0)
            
            # 计算本轮训练的平均损失和准确率
            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = 100. * correct / total
            
            # 记录训练指标
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)
            
            # 在测试集上评估模型
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            epoch_test_loss = test_loss / len(test_loader.dataset)
            epoch_test_acc = 100. * test_correct / test_total
            
            # 记录测试指标
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_acc)
            
            # 更新学习率
            scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新状态为测试结果
            update_training_status(epoch+1, model_params['epochs'], len(train_loader), len(train_loader), 
                                  epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc,
                                  status='training', 
                                  message=f'第 {epoch+1}/{model_params["epochs"]} 轮训练完成，学习率: {current_lr:.6f}')
            
            # 早停机制检查
            if model_params['model_type'] in ['improved', 'advanced']:  # 只对改进和高级模型使用早停
                # 只有当准确率提高超过最小改进阈值时才认为有改进
                improvement = epoch_test_acc - best_val_acc
                if improvement > min_improvement:
                    best_val_acc = epoch_test_acc
                    best_epoch = epoch + 1  # 记录最佳epoch
                    counter = 0
                    # 保存最佳模型状态
                    try:
                        # 深拷贝状态字典，避免引用问题
                        best_model_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                        improvement_percent = (improvement / (best_val_acc - improvement)) * 100
                        update_training_status(epoch+1, model_params['epochs'], len(train_loader), len(train_loader),
                                             epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc,
                                             status='training', 
                                             message=f'更新最佳模型状态: 测试准确率提升 {improvement_percent:.2f}%，当前最佳: {best_val_acc:.4f}')
                    except Exception as e:
                        update_training_status(epoch+1, model_params['epochs'], len(train_loader), len(train_loader),
                                             epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc,
                                             status='training', message=f'保存最佳模型状态时出错: {str(e)}')
                else:
                    counter += 1
                    message = f'无显著改进，早停计数器: {counter}/{patience}'
                    if improvement > 0:
                        message += f' (微小提升: {improvement:.6f})'
                    update_training_status(epoch+1, model_params['epochs'], len(train_loader), len(train_loader),
                                         epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc,
                                         status='training', message=message)
                    
                # 动态调整学习率策略（在较长时间没有改进时）
                if counter > patience // 3 and counter % 20 == 0:
                    # 记录长时间无改进的情况
                    update_training_status(epoch+1, model_params['epochs'], len(train_loader), len(train_loader),
                                         epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc,
                                         status='training', message=f'长时间无显著改进，继续训练以探索更好的解')
            
            # 早停条件
            if counter >= patience and model_params['model_type'] in ['improved', 'advanced']:
                update_training_status(epoch+1, model_params['epochs'], len(train_loader), len(train_loader),
                                     epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc,
                                     status='training', 
                                     message=f'触发早停机制: {patience}轮无显著改进。最佳模型在第{best_epoch}轮，测试准确率: {best_val_acc:.4f}')
                break
        
        # 如果没有被停止，完成训练流程
        if not stop_training_flag:
            # 训练完成
            train_results = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }
            
            # 如果有最佳模型状态，加载它
            if best_model_state is not None:
                try:
                    # 首先尝试严格加载，看是否能完全匹配
                    try:
                        model.load_state_dict(best_model_state, strict=True)
                        update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                             train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                             train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                             status='training', message='成功严格加载最佳模型状态')
                    except RuntimeError as strict_error:
                        # 如果严格加载失败，尝试使用strict=False
                        update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                             train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                             train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                             status='training', message=f'严格加载失败，尝试兼容加载: {str(strict_error)}')
                        
                        # 使用strict=False允许部分加载，避免BatchNorm参数不匹配问题
                        model.load_state_dict(best_model_state, strict=False)
                        update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                             train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                             train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                             status='training', message='使用strict=False加载最佳模型状态')
                        
                        # 验证加载后的模型BatchNorm层
                        try:
                            validate_model_batchnorm(model)
                            update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                                 train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                                 train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                                 status='training', message='加载后模型结构验证通过')
                        except Exception as validation_error:
                            update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                                 train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                                 train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                                 status='training', message=f'加载后模型验证警告: {str(validation_error)}')
                except RuntimeError as e:
                    if 'running_mean' in str(e) or 'running_var' in str(e):
                        # 特定处理BatchNorm相关错误
                        update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                             train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                             train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                             status='training', message=f'BatchNorm参数不匹配，尝试兼容处理: {str(e)}')
                        # 尝试只加载不包含BatchNorm运行统计的参数
                        filtered_state_dict = {}
                        for k, v in best_model_state.items():
                            if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k:
                                filtered_state_dict[k] = v
                        try:
                            model.load_state_dict(filtered_state_dict, strict=False)
                            update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                                 train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                                 train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                                 status='training', message='成功加载过滤后的模型状态（排除BatchNorm统计信息）')
                        except Exception as inner_e:
                            update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                                 train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                                 train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                                 status='training', message=f'过滤后加载仍失败: {str(inner_e)}')
                    else:
                        raise
            
            # 保存当前模型
            # 使用深拷贝避免后续操作影响模型状态
            try:
                # 创建模型的新实例并加载当前状态
                if model_params['model_type'] == 'simple':
                    current_model = SimpleCNN(num_classes=num_classes)
                elif model_params['model_type'] == 'improved':
                    current_model = ImprovedCNN(num_classes=num_classes)
                else:  # advanced
                    current_model = AdvancedCNN(num_classes=num_classes)
                
                # 将新模型移至设备
                current_model.to(device)
                
                # 深拷贝状态字典
                state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                current_model.load_state_dict(state_dict)
                update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                     train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                     train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                     status='training', message='成功保存当前模型')
            except Exception as e:
                update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                     train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                     train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                     status='training', message=f'保存模型时出错: {str(e)}')
                # 如果保存失败，使用原始模型作为备选
                current_model = model
            
            # 保存模型和参数到文件
            try:
                # 导入datetime模块
                from datetime import datetime
                
                # 保存前验证模型结构
                validate_model_batchnorm(model)
                
                # 保存模型到独立包目录，避免覆盖旧模型
                model_params_with_timestamp = model_params.copy()
                model_params_with_timestamp['save_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                try:
                    # 将 test_loader 传入以便保存混淆矩阵缩略图
                    bundle_dir = save_model_bundle(model, model_params_with_timestamp, train_results, class_names, name_prefix=model_params.get('model_type', 'model'), test_loader=test_loader)
                    update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                         train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                         train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                         status='training', message=f'模型和参数保存成功: {bundle_dir}')
                except Exception as e:
                    # 回退：保存到旧的静态路径以保证兼容
                    try:
                        torch.save(model.state_dict(), 'static/cnn_model.pth')
                        torch.save(model, 'static/best_model_complete.pth')
                        with open('static/model_params.json', 'w') as f:
                            json.dump(model_params_with_timestamp, f)
                        update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                             train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                             train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                             status='training', message='模型保存失败，已使用回退路径保存')
                    except Exception as inner_e:
                        update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                             train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                             train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                             status='training', message=f'保存模型时出错: {str(inner_e)}')
                
                update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                     train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                     train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                     status='training', message='模型和参数保存成功')
            except Exception as e:
                update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                     train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                     train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                     status='training', message=f'保存模型时出错: {str(e)}')
                
                # 尝试保存时排除BatchNorm运行统计信息
                try:
                    filtered_state_dict = {}
                    for k, v in model.state_dict().items():
                        if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k:
                            filtered_state_dict[k] = v
                    torch.save(filtered_state_dict, 'static/cnn_model_no_batchnorm.pth')
                    update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                         train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                         train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                         status='training', message='已保存不含BatchNorm统计信息的模型权重作为备份')
                except Exception as inner_e:
                    update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                         train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                         train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                         status='training', message=f'保存备份模型失败: {str(inner_e)}')
            
            # 保存结果供后续使用
            with open('static/train_results.json', 'w') as f:
                json.dump({
                    'train_losses': train_results['train_losses'],
                    'test_losses': train_results['test_losses'],
                    'train_accuracies': train_results['train_accuracies'],
                    'test_accuracies': train_results['test_accuracies'],
                    'epochs': model_params['epochs'],
                    'class_names': class_names
                }, f)
            
            # 保存模型参数供JavaScript使用
            with open('static/model_params.json', 'w') as f:
                json.dump(model_params, f)
            
            # 生成并保存训练图表
            plot_path = plot_results(train_results)
            
            # 更新状态为完成
            update_training_status(model_params['epochs'], model_params['epochs'], len(train_loader), len(train_loader),
                                  train_results['train_losses'][-1], train_results['train_accuracies'][-1],
                                  train_results['test_losses'][-1], train_results['test_accuracies'][-1],
                                  status='completed', message='训练完成！')
        
    except RuntimeError as e:
        # 特别处理BatchNorm相关错误
        error_msg = str(e)
        if 'running_mean' in error_msg or 'size mismatch' in error_msg:
            update_training_status(training_status['current_epoch'], training_status['total_epochs'],
                                  training_status['current_batch'], training_status['total_batches'],
                                  training_status['train_loss'], training_status['train_accuracy'],
                                  status='error', message=f'BatchNorm参数不匹配错误: {error_msg}\n建议: 清理旧模型文件并重新开始训练')
        else:
            update_training_status(training_status['current_epoch'], training_status['total_epochs'],
                                  training_status['current_batch'], training_status['total_batches'],
                                  training_status['train_loss'], training_status['train_accuracy'],
                                  status='error', message=f'训练过程中出错: {error_msg}')
    except Exception as e:
        # 捕获其他类型的错误
        update_training_status(training_status['current_epoch'], training_status['total_epochs'],
                              training_status['current_batch'], training_status['total_batches'],
                              training_status['train_loss'], training_status['train_accuracy'],
                              status='error', message=f'训练过程中出错: {str(e)}')
    finally:
        # 确保训练状态更新为非进行中
        with training_lock:
            training_in_progress = False

@app.route('/train', methods=['GET', 'POST'])
def train():
    global current_model, model_params, train_results, class_names, training_in_progress, training_thread, training_status, train_logs, stop_training_flag
    
    if request.method == 'POST':
        # 检查是否已经有训练在进行
        if training_in_progress:
            flash('训练已经在进行中，请不要重复点击！')
            return redirect(url_for('train'))
        
        # 加载数据集
        try:
            # 检查数据是否已准备好
            if not os.path.exists(os.path.join(app.config['DATA_FOLDER'], 'train')):
                flash('数据集未准备好，请先在首页准备数据！')
                return redirect(url_for('index'))
            
            # 重置训练状态和日志
            train_logs = []
            stop_training_flag = False
            training_status = {
                'current_epoch': 0,
                'total_epochs': model_params['epochs'],
                'current_batch': 0,
                'total_batches': 0,
                'train_loss': 0.0,
                'train_accuracy': 0.0,
                'test_loss': 0.0,
                'test_accuracy': 0.0,
                'status': 'idle',
                'message': '准备开始训练...'
            }
            
            # 标记训练为进行中
            with training_lock:
                training_in_progress = True
            
            # 在新线程中开始训练
            training_thread = threading.Thread(target=train_model_thread)
            training_thread.daemon = True
            training_thread.start()
            
            # 立即返回，不等待训练完成
            return redirect(url_for('train'))
            
        except Exception as e:
            # 如果准备阶段出错，确保训练状态正确
            with training_lock:
                training_in_progress = False
            flash(f'训练准备过程中出错：{str(e)}')
            return redirect(url_for('train'))
    
    if model_params is None:
        flash('请先在配置页面设置模型参数！')
        return redirect(url_for('config'))
    
    # 保存模型参数供JavaScript使用
    with open('static/model_params.json', 'w') as f:
        json.dump(model_params, f)
    
    # 保存模型参数供JavaScript使用
    if model_params:
        with open('static/model_params.json', 'w') as f:
            json.dump(model_params, f)
    
    return render_template('train.html')

@app.route('/api/training_status')
def api_training_status():
    """API端点，用于获取实时训练状态"""
    global training_in_progress, training_status, train_logs
    with training_lock:
        # 获取最新的日志（限制数量）
        recent_logs = train_logs[-100:]
        
        return jsonify({
            'training_in_progress': training_in_progress,
            'status': training_status,
            'logs': recent_logs
        })

@app.route('/api/stop_training', methods=['POST'])
def api_stop_training():
    """API端点，用于停止训练"""
    global training_in_progress, stop_training_flag
    
    if not training_in_progress:
        return jsonify({'status': 'error', 'message': '没有正在进行的训练'})
    
    # 设置停止标志
    stop_training_flag = True
    
    return jsonify({'status': 'success', 'message': '停止训练请求已发送'})

@app.route('/api/resume_training', methods=['POST'])
def api_resume_training():
    """API端点，用于继续训练"""
    global training_in_progress, stop_training_flag, training_thread, training_status
    
    # 检查是否有中断的模型可以继续
    if not os.path.exists('static/interrupted_model.pth'):
        return jsonify({'status': 'error', 'message': '没有找到可以继续训练的模型'})
    
    # 确保没有训练在进行中
    if training_in_progress:
        return jsonify({'status': 'error', 'message': '已有训练在进行中'})
    
    # 重置停止标志
    stop_training_flag = False
    
    # 标记训练为进行中
    with training_lock:
        training_in_progress = True
    
    # 更新训练状态
    training_status['status'] = 'training'
    training_status['message'] = '准备继续训练...'
    
    # 在新线程中继续训练
    training_thread = threading.Thread(target=resume_training_thread)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'status': 'success', 'message': '继续训练请求已发送'})

@app.route('/api/terminate_training', methods=['POST'])
def api_terminate_training():
    """API端点，用于终止训练并保存结果"""
    global training_in_progress, stop_training_flag
    
    # 如果训练仍在进行，先停止它
    if training_in_progress:
        stop_training_flag = True
    
    # 确保中断模型已保存
    if not os.path.exists('static/interrupted_model.pth'):
        return jsonify({'status': 'error', 'message': '没有找到中断的模型'})
    
    # 将中断模型复制为当前模型，以便在结果页面使用
    if os.path.exists('static/interrupted_model.pth'):
        try:
            shutil.copy2('static/interrupted_model.pth', 'static/cnn_model.pth')
            # 如果有中断结果，也复制为当前结果
            if os.path.exists('static/interrupted_results.json'):
                shutil.copy2('static/interrupted_results.json', 'static/train_results.json')
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'复制模型文件时出错: {str(e)}'})
    
    return jsonify({'status': 'success', 'message': '训练已终止，模型已保存'})

@app.route('/api/copy_sample_images')
def api_copy_sample_images():
    """API端点，用于复制样本图像到static/samples目录供预览使用"""
    try:
        import shutil
        import os
        
        # 确保samples目录存在
        samples_dir = 'static/samples'
        os.makedirs(samples_dir, exist_ok=True)
        
        # 复制每个类别的第一张图像
        train_dir = os.path.join(app.config['DATA_FOLDER'], 'train')
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)
                if images:
                    # 复制第一张图像
                    src = os.path.join(class_dir, images[0])
                    dst = os.path.join(samples_dir, f'{class_name}_0.png')
                    shutil.copy(src, dst)
        
        return jsonify({'status': 'success', 'message': '样本图像复制完成'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get_sample_image')
def get_sample_image():
    """API端点，用于提供测试数据集的图片访问功能"""
    # 获取请求参数
    folder = request.args.get('folder')
    file = request.args.get('file')
    
    if not folder or not file:
        return jsonify({'error': 'Missing folder or file parameter'}), 400
    
    # 构建文件路径
    test_dir = os.path.join(app.config['DATA_FOLDER'], 'test')
    file_path = os.path.join(test_dir, folder, file)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # 发送文件
    return send_from_directory(os.path.join(test_dir, folder), file)


@app.route('/api/list_models')
def api_list_models():
    """列出 static/models 目录下保存的模型包及其基本元数据"""
    model_dir = os.path.join('static', 'models')
    candidates = []

    try:
        if not os.path.exists(model_dir):
            return jsonify({'status': 'success', 'models': [], 'meta': {}})

        for entry in os.listdir(model_dir):
            folder = os.path.join(model_dir, entry)
            if os.path.isdir(folder):
                # read meta if exists
                meta = {}
                meta_path = os.path.join(folder, 'meta.json')
                params_path = os.path.join(folder, 'model_params.json')
                results_path = os.path.join(folder, 'train_results.json')
                class_path = os.path.join(folder, 'class_names.json')
                try:
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                except Exception:
                    meta = {}

                # build display name
                display = entry
                # try to include saved time from meta
                saved_at = meta.get('saved_at') if isinstance(meta, dict) else None
                if saved_at:
                    display = f"{entry} ({saved_at})"

                # prepare modified time: prefer meta.saved_at, otherwise folder mtime
                saved_at = meta.get('saved_at') if isinstance(meta, dict) else None
                if not saved_at:
                    try:
                        mtime = os.path.getmtime(folder)
                        saved_at = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        saved_at = ''

                # try to read model_params and available previews
                params = None
                params_path_pkg = os.path.join(folder, 'model_params.json')
                if os.path.exists(params_path_pkg):
                    try:
                        with open(params_path_pkg, 'r', encoding='utf-8') as f:
                            params = json.load(f)
                    except Exception:
                        params = None

                # preview images
                training_plot_rel = None
                confusion_rel = None
                tp = os.path.join(folder, 'training_plot.png')
                cm = os.path.join(folder, 'confusion_matrix.png')
                if os.path.exists(tp):
                    training_plot_rel = os.path.join('static', 'models', entry, 'training_plot.png').replace('\\', '/')
                if os.path.exists(cm):
                    confusion_rel = os.path.join('static', 'models', entry, 'confusion_matrix.png').replace('\\', '/')
                # try to read train_results if exists
                train_results_data = None
                trp = os.path.join(folder, 'train_results.json')
                if os.path.exists(trp):
                    try:
                        with open(trp, 'r', encoding='utf-8') as f:
                            train_results_data = json.load(f)
                    except Exception:
                        train_results_data = None

                candidates.append({
                    'id': entry,
                    'filename': entry,
                    'display': display,
                    'modified': saved_at,
                    'folder': folder,
                    'meta': meta,
                    'model_params': params,
                    'training_plot': training_plot_rel,
                    'confusion_matrix': confusion_rel,
                    'train_results': train_results_data
                })

        # read latest pointer if exists
        latest = {}
        latest_path = os.path.join(model_dir, 'latest.json')
        if os.path.exists(latest_path):
            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    latest = json.load(f)
            except Exception:
                latest = {}

        return jsonify({'status': 'success', 'models': candidates, 'meta': {'latest': latest}})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/evaluate_model', methods=['POST'])
def api_evaluate_model():
    """评估指定模型在测试集上的表现（返回loss和accuracy）"""
    # 现在支持传入模型包 id（对应 static/models/<id>）或兼容旧的 model file 路径
    model_file = request.form.get('model_file') or (request.json.get('model_file') if request.is_json else None)
    if not model_file:
        return jsonify({'status': 'error', 'message': '请提供 model_file 参数'}), 400

    # 优先当作模型包 id 处理
    candidate_folder = os.path.join('static', 'models', os.path.basename(model_file))
    candidate = None
    if os.path.exists(candidate_folder) and os.path.isdir(candidate_folder):
        # prefer complete model then state
        complete = os.path.join(candidate_folder, 'model_complete.pth')
        state = os.path.join(candidate_folder, 'model_state.pth')
        if os.path.exists(complete):
            candidate = complete
            model_params_path = os.path.join(candidate_folder, 'model_params.json')
            class_names_path = os.path.join(candidate_folder, 'class_names.json')
        elif os.path.exists(state):
            candidate = state
            model_params_path = os.path.join(candidate_folder, 'model_params.json')
            class_names_path = os.path.join(candidate_folder, 'class_names.json')
    else:
        # 兼容旧逻辑：传入文件名或路径
        alt = os.path.join('static', os.path.basename(model_file))
        if os.path.exists(alt):
            candidate = alt
            model_params_path = os.path.join('static', 'model_params.json')
            class_names_path = os.path.join('static', 'class_names.json')

    if candidate is None or not os.path.exists(candidate):
        return jsonify({'status': 'error', 'message': '模型文件/包不存在'}), 404

    try:
        # 载入数据集的测试集
        train_dataset, test_dataset, loaded_class_names = load_dataset(app.config['DATA_FOLDER'])
        # 优先使用模型包内的 class_names
        use_class_names = None
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    use_class_names = json.load(f)
            except Exception:
                use_class_names = loaded_class_names if loaded_class_names else class_names
        else:
            use_class_names = loaded_class_names if loaded_class_names else class_names

        # 构建重映射后的测试集 DataLoader（确保标签范围与模型输出一致）
        batch_size = model_params['batch_size'] if model_params and 'batch_size' in model_params else 32
        if use_class_names:
            test_loader = build_aligned_test_loader(test_dataset, use_class_names, batch_size)
        else:
            # 兜底：未能获取类别名时，直接使用原测试集
            test_loader = create_dataloaders(train_dataset, test_dataset, batch_size)[1]

        # 准备模型结构
        model_type = 'advanced'
        num_classes = len(use_class_names) if use_class_names else 10
        if os.path.exists(model_params_path):
            try:
                with open(model_params_path, 'r', encoding='utf-8') as f:
                    saved_params = json.load(f)
                model_type = saved_params.get('model_type', model_type)
                if 'num_classes' in saved_params:
                    num_classes = saved_params.get('num_classes')
            except Exception:
                pass

        model = create_model_instance(model_type, num_classes)
        # 尝试用兼容加载函数加载
        try:
            model = load_model(model, candidate)
        except Exception:
            state = torch.load(candidate, map_location=device)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()

        # 评估
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        return jsonify({'status': 'success', 'test_loss': test_loss, 'test_accuracy': test_acc})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/compare_models', methods=['POST'])
def api_compare_models():
    """比较多个模型在测试集上的指标，接受JSON数组 'models': [ 'cnn_model.pth', ... ]"""
    data = request.get_json() if request.is_json else request.form
    model_list = data.get('models') if data else None
    if not model_list or not isinstance(model_list, list):
        return jsonify({'status': 'error', 'message': '请提供 models 字段，值为模型文件名数组'}), 400

    results = []
    for mf in model_list:
        # mf expected to be a model package id (folder name under static/models)
        folder = os.path.join('static', 'models', os.path.basename(mf))
        if not os.path.exists(folder) or not os.path.isdir(folder):
            results.append({'model': mf, 'error': '模型包不存在'})
            continue

        try:
            # 载入测试集
            train_dataset, test_dataset, loaded_class_names = load_dataset(app.config['DATA_FOLDER'])
            # 批大小
            batch_size = model_params['batch_size'] if model_params and 'batch_size' in model_params else 32

            # 读取模型参数和类别
            model_params_path = os.path.join(folder, 'model_params.json')
            class_names_path = os.path.join(folder, 'class_names.json')
            model_type = 'advanced'
            num_classes = None
            if os.path.exists(class_names_path):
                try:
                    with open(class_names_path, 'r', encoding='utf-8') as f:
                        use_class_names = json.load(f)
                    num_classes = len(use_class_names)
                except Exception:
                    use_class_names = loaded_class_names
            else:
                use_class_names = loaded_class_names

            if os.path.exists(model_params_path):
                try:
                    with open(model_params_path, 'r', encoding='utf-8') as f:
                        saved_params = json.load(f)
                    model_type = saved_params.get('model_type', model_type)
                    if num_classes is None and 'num_classes' in saved_params:
                        num_classes = saved_params.get('num_classes')
                except Exception:
                    pass

            if num_classes is None:
                num_classes = len(use_class_names) if use_class_names else 10

            # 使用模型的类别集对齐测试集，避免 CrossEntropyLoss 标签越界
            if use_class_names:
                test_loader = build_aligned_test_loader(test_dataset, use_class_names, batch_size)
            else:
                test_loader = create_dataloaders(train_dataset, test_dataset, batch_size)[1]

            model = create_model_instance(model_type, num_classes)
            # 找到模型权重
            complete = os.path.join(folder, 'model_complete.pth')
            state = os.path.join(folder, 'model_state.pth')
            candidate = complete if os.path.exists(complete) else state
            if not candidate or not os.path.exists(candidate):
                results.append({'model': mf, 'error': '模型权重文件不存在'})
                continue

            try:
                model = load_model(model, candidate)
            except Exception:
                stateobj = torch.load(candidate, map_location=device)
                model.load_state_dict(stateobj, strict=False)
                model.to(device)
                model.eval()

            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

            results.append({'model': mf, 'test_loss': test_loss, 'test_accuracy': test_acc})
        except Exception as e:
            results.append({'model': mf, 'error': str(e)})

    return jsonify({'status': 'success', 'results': results})

@app.route('/results')
def results():
    global train_results, class_names
    
    if train_results is None:
        flash('请先训练模型！')
        return redirect(url_for('train'))
    
    return render_template('results.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global current_model, class_names
    # 不强制要求内存中存在 current_model 即可打开预测页面。
    # 预测时可通过下拉选择已保存的模型包，或使用内存中的 current_model（若存在）。
    # 保存类别名称供JavaScript使用
    if class_names:
        try:
            with open('static/class_names.json', 'w', encoding='utf-8') as f:
                json.dump(class_names, f)
        except Exception:
            pass

    return render_template('predict.html')


@app.route('/compare')
def compare():
    """前端页面：模型对比"""
    return render_template('compare.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global current_model, class_names
    
    # 统一解析类别名称，确保与模型输出维度一致且顺序正确
    def resolve_class_names(num_outputs: int, candidate_folder=None):
        """Return a list of class names aligned to num_outputs.
        Priority:
        1) class_names.json in the selected model package (if provided) and len==num_outputs
        2) static/class_names.json if len==num_outputs
        3) data/train subfolder names (sorted) if len==num_outputs
        4) CIFAR-10 default order when num_outputs==10
        5) fallback to generic names
        Also, handle cases where static/class_names.json contains extra categories by filtering to known CIFAR-10 set when possible.
        """
        # helper to load json safely
        def _load_json(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        
        # 1) package class names
        if candidate_folder:
            pkg_cn_path = os.path.join(candidate_folder, 'class_names.json')
            pkg_cn = _load_json(pkg_cn_path)
            if isinstance(pkg_cn, list) and len(pkg_cn) == num_outputs:
                return pkg_cn
        
        # 2) static/class_names.json
        static_cn_path = os.path.join('static', 'class_names.json')
        static_cn = _load_json(static_cn_path)
        if isinstance(static_cn, list):
            if len(static_cn) == num_outputs:
                return static_cn
            # If it contains extra entries and this looks like CIFAR-like, filter to canonical set
            cifar10_set = {'plane','car','bird','cat','deer','dog','frog','horse','ship','truck'}
            if len(static_cn) > num_outputs and num_outputs == 10:
                filtered = [c for c in static_cn if c in cifar10_set]
                if len(filtered) == num_outputs:
                    return filtered
        
        # 3) data/train folder names (ImageFolder order is sorted)
        train_dir = os.path.join(app.config.get('DATA_FOLDER', 'data'), 'train')
        if os.path.exists(train_dir):
            try:
                subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
                subdirs.sort()
                if len(subdirs) == num_outputs:
                    return subdirs
            except Exception:
                pass
        
        # 4) CIFAR-10 default order
        if num_outputs == 10:
            return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 5) generic fallback
        return [f'class_{i}' for i in range(num_outputs)]
    
    # 支持通过上传参数选择使用哪个模型进行预测: 接受可选字段 'model_file' 指定static目录下的模型文件名
    model_file = request.form.get('model_file') or request.args.get('model_file')

    predict_model = None
    # 支持模型包 id（static/models/<id>）或兼容旧的单文件路径
    if model_file:
        folder = os.path.join('static', 'models', os.path.basename(model_file))
        candidate = None
        model_params_path = None
        class_names_path = None
        if os.path.exists(folder) and os.path.isdir(folder):
            # prefer complete then state
            complete = os.path.join(folder, 'model_complete.pth')
            state = os.path.join(folder, 'model_state.pth')
            if os.path.exists(complete):
                candidate = complete
            elif os.path.exists(state):
                candidate = state
            model_params_path = os.path.join(folder, 'model_params.json')
            class_names_path = os.path.join(folder, 'class_names.json')
        else:
            # 兼容：直接给出单个文件路径（在 static/ 下）
            alt = os.path.join('static', os.path.basename(model_file))
            if os.path.exists(alt):
                candidate = alt
                model_params_path = os.path.join('static', 'model_params.json')
                class_names_path = os.path.join('static', 'class_names.json')

        if candidate is None:
            return jsonify({'status': 'error', 'message': f'指定的模型文件/包不存在'}), 400

        try:
            # 如果是完整模型对象
            if candidate.endswith('_complete.pth') or candidate.endswith('model_complete.pth'):
                predict_model = torch.load(candidate, map_location=device)
                predict_model.to(device)
                predict_model.eval()
            else:
                # 构建模型结构
                num_classes = None
                if os.path.exists(class_names_path):
                    try:
                        with open(class_names_path, 'r', encoding='utf-8') as f:
                            cn = json.load(f)
                        num_classes = len(cn)
                        # prefer using package class_names
                        class_names = cn
                    except Exception:
                        pass

                model_type = None
                if os.path.exists(model_params_path):
                    try:
                        with open(model_params_path, 'r', encoding='utf-8') as f:
                            mp = json.load(f)
                        model_type = mp.get('model_type')
                    except Exception:
                        pass

                if model_type:
                    predict_model = create_model_instance(model_type, num_classes if num_classes else 10)
                else:
                    predict_model = AdvancedCNN(num_classes=num_classes if num_classes else (len(class_names) if class_names else 10))

                try:
                    predict_model = load_model(predict_model, candidate)
                except Exception:
                    state = torch.load(candidate, map_location=device)
                    predict_model.load_state_dict(state, strict=False)
                    predict_model.to(device)
                    predict_model.eval()
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'加载指定模型时出错: {str(e)}'})
    else:
        # 没有指定模型文件，则使用内存中的 current_model（如果存在），否则尝试按最新包加载一次
        if current_model is None:
            load_saved_model_on_start()
        predict_model = current_model

    if predict_model is None:
        return jsonify({'status': 'error', 'message': '未能获取用于预测的模型，请先训练或指定模型文件'}), 400
    
    # 确保类别名称已加载（并在预测后与输出维度严格对齐）
    if class_names is None:
        # 暂不强求在此处加载，稍后根据模型输出维度进行解析
        pass
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': '没有文件部分'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    try:
        # 保存上传的文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(filename)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # 进行预测
        model_to_use = predict_model
        model_to_use.eval()
        # 确定模型所在设备并将输入移到相同设备
        model_device = next(model_to_use.parameters()).device
        image_tensor = image_tensor.to(model_device)
        with torch.no_grad():
            outputs = model_to_use(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # 按模型输出维度安全映射类别名称，避免越界
        num_outputs = probabilities.size(1)
        # 解析并对齐类别名称
        candidate_folder = None
        if model_file:
            candidate_folder = os.path.join('static', 'models', os.path.basename(model_file))
        used_class_names = resolve_class_names(num_outputs, candidate_folder if candidate_folder and os.path.isdir(candidate_folder) else None)
        # 将解析后的类别名称作为当前会话的类名，并写入 static/class_names.json 供前端显示
        class_names = used_class_names
        try:
            with open(os.path.join('static', 'class_names.json'), 'w', encoding='utf-8') as f:
                json.dump(class_names, f)
        except Exception:
            pass

        # 准备结果（使用对齐后的类别名称）
        pred_idx = int(predicted.item())
        if pred_idx >= num_outputs:
            # 极端兜底：不应发生，但确保不崩溃
            pred_idx = num_outputs - 1
        prediction = used_class_names[pred_idx]
        confidence_level = confidence.item() * 100
        
        # 生成所有类别的概率（按实际输出维度）
        all_probabilities = {used_class_names[i]: float(probabilities[0][i].item() * 100)
                             for i in range(num_outputs)}

        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'prediction': prediction,
            'confidence_level': confidence_level,
            'all_probabilities': all_probabilities,
            'num_outputs': num_outputs
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/prepare_data')
def prepare_data():
    # 这里可以添加数据准备的逻辑，比如下载示例数据集
    # 为了演示，我们可以创建一个简单的脚本
    return render_template('prepare_data.html')

# 添加全局变量来跟踪训练状态
training_in_progress = False

@app.route('/api/check_gpu')
def api_check_gpu():
    """API端点，用于检查GPU是否可用"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        return jsonify({'available': gpu_available})
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})

@app.route('/api/get_system_status')
def api_get_system_status():
    """API端点，用于获取系统各组件的状态"""
    try:
        # 检查数据集状态
        data_status = '未准备'
        data_class = 'bg-warning'
        train_dir = os.path.join(app.config['DATA_FOLDER'], 'train')
        test_dir = os.path.join(app.config['DATA_FOLDER'], 'test')
        
        # 检查是否有训练和测试文件夹，并且每个文件夹至少有一个类
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            
            if len(train_classes) > 0 and len(test_classes) > 0:
                # 检查每个类是否至少有一张图片
                all_ready = True
                for class_name in train_classes:
                    if len(os.listdir(os.path.join(train_dir, class_name))) == 0:
                        all_ready = False
                        break
                
                for class_name in test_classes:
                    if len(os.listdir(os.path.join(test_dir, class_name))) == 0:
                        all_ready = False
                        break
                
                if all_ready:
                    data_status = '已准备'
                    data_class = 'bg-success'
        
        # 检查模型配置状态
        config_status = '未配置'
        config_class = 'bg-warning'
        if model_params is not None:
            config_status = '已配置'
            config_class = 'bg-success'
        
        # 检查模型训练状态
        train_status = '未训练'
        train_class = 'bg-warning'
        if training_in_progress:
            train_status = '训练中'
            train_class = 'bg-info'
        elif current_model is not None:
            train_status = '已训练'
            train_class = 'bg-success'
        else:
            # 如果内存中没有模型，但存在已保存的模型包，也认为有训练结果可用
            model_dir = os.path.join('static', 'models')
            try:
                if os.path.exists(model_dir):
                    entries = [e for e in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, e))]
                    if len(entries) > 0:
                        train_status = '已训练'
                        train_class = 'bg-success'
            except Exception:
                pass
        
        # 检查GPU状态
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_status = '可用' if gpu_available else '不可用'
        gpu_class = 'bg-success' if gpu_available else 'bg-danger'
        
        return jsonify({
            'data_status': data_status,
            'data_class': data_class,
            'config_status': config_status,
            'config_class': config_class,
            'train_status': train_status,
            'train_class': train_class,
            'gpu_status': gpu_status,
            'gpu_class': gpu_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug_list_models')
def api_debug_list_models():
    """调试用：直接返回 static/models 下的目录与 meta.json 内容（不依赖任何复杂加载）。"""
    model_dir = os.path.join('static', 'models')
    out = []
    try:
        if not os.path.exists(model_dir):
            return jsonify({'status': 'success', 'models': []})
        for entry in sorted(os.listdir(model_dir)):
            folder = os.path.join(model_dir, entry)
            if os.path.isdir(folder):
                meta = {}
                meta_path = os.path.join(folder, 'meta.json')
                try:
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                except Exception as e:
                    meta = {'_error': str(e)}
                out.append({'folder': entry, 'meta': meta})
        return jsonify({'status': 'success', 'models': out})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/prepare_sample_data')
def api_prepare_sample_data():
    try:
        # 获取数据集类型参数
        dataset_type = request.args.get('dataset_type', 'cifar10')
        
        # 创建训练和测试文件夹
        train_dir = os.path.join(app.config['DATA_FOLDER'], 'train')
        test_dir = os.path.join(app.config['DATA_FOLDER'], 'test')
        
        # 根据数据集类型定义类别
        if dataset_type == 'new_dataset':
            # 新数据集类别
            classes = ['flowers', 'fruits', 'animals', 'vehicles', 'buildings']
        else:
            # 默认CIFAR-10类别
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 检查数据是否已经存在
        data_exists = True
        for class_name in classes:
            if not os.path.exists(os.path.join(train_dir, class_name)) or len(os.listdir(os.path.join(train_dir, class_name))) == 0:
                data_exists = False
                break
        
        if data_exists:
            return jsonify({'status': 'success', 'message': '数据集已存在！', 'data_already_prepared': True, 'classes': classes})
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # 为每个类别创建文件夹
        for class_name in classes:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        if dataset_type == 'new_dataset':
            # 对于新数据集，我们假设数据已经通过其他方式准备好了
            # 这里可以添加从网络下载图像的逻辑
            return jsonify({'status': 'success', 'message': '新数据集结构已创建！', 'classes': classes})
        else:
            # 下载示例数据集（使用CIFAR-10的一个子集）
            import shutil
            import random
            from torchvision import datasets
            
            dataset = datasets.CIFAR10(root='./temp_data', train=True, download=True)
            
            # 复制部分数据到我们的数据集文件夹
            for i in range(min(500, len(dataset))):  # 只使用前500张图片作为示例
                img, label = dataset[i]
                class_name = classes[label]
                
                # 80%作为训练集，20%作为测试集
                if random.random() < 0.8:
                    dest_dir = train_dir
                else:
                    dest_dir = test_dir
                
                img.save(os.path.join(dest_dir, class_name, f'image_{i}.png'))
            
            # 清理临时数据
            shutil.rmtree('./temp_data', ignore_errors=True)
        
        return jsonify({'status': 'success', 'message': '示例数据集准备完成！', 'classes': classes})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/prepare_coil20', methods=['POST'])
def api_prepare_coil20():
    """启动 COIL-20 数据集准备的异步任务，避免长请求被中途取消。
    可选参数：zip_source（表单或JSON）：本地压缩包路径，优先使用本地文件。
    返回：{status: 'started', job_id: '...'}
    """
    try:
        # 读取可选的本地压缩包路径
        zip_source = None
        if request.is_json:
            zip_source = (request.get_json() or {}).get('zip_source')
        else:
            zip_source = request.form.get('zip_source')

        # 创建任务ID
        job_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        coil20_jobs[job_id] = {
            'status': 'pending',
            'progress': 0,
            'message': '开始准备 COIL-20 数据集...',
            'result': None
        }

        # 在线程中执行耗时任务
        def run_prepare_coil20(job_id_local: str, zip_src: str = None):
            try:
                # 进度：下载/读取压缩包
                coil20_jobs[job_id_local]['status'] = 'running'
                coil20_jobs[job_id_local]['progress'] = 10
                coil20_jobs[job_id_local]['message'] = '正在下载或读取压缩包...'

                if zip_src:
                    prepare_coil20_dataset(zip_source=zip_src)
                else:
                    prepare_coil20_dataset()

                # 进度：解压与整理
                coil20_jobs[job_id_local]['progress'] = 60
                coil20_jobs[job_id_local]['message'] = '正在解压与整理为 ImageFolder...'

                # 加载数据集（COIL-20）
                data_dir = app.config['DATA_FOLDER']
                train_dataset, test_dataset, class_names_local = load_dataset(data_dir, dataset_type='coil20')

                # 保存统计信息到 data/stats
                stats_dir = os.path.join(data_dir, 'stats')
                os.makedirs(stats_dir, exist_ok=True)
                stats_text = save_dataset_statistics(train_dataset, test_dataset, class_names_local, stats_dir)

                # 将类别名称保存到静态文件，便于前端使用
                try:
                    with open('static/class_names.json', 'w', encoding='utf-8') as f:
                        json.dump(class_names_local, f, ensure_ascii=False)
                except Exception:
                    pass

                # 更新全局类别变量
                global class_names
                class_names = class_names_local

                # 完成
                coil20_jobs[job_id_local]['progress'] = 100
                coil20_jobs[job_id_local]['status'] = 'completed'
                coil20_jobs[job_id_local]['message'] = 'COIL-20 数据集准备完成并已生成统计信息'
                coil20_jobs[job_id_local]['result'] = {
                    'classes': class_names_local,
                    'train_size': len(train_dataset),
                    'test_size': len(test_dataset),
                    'stats_dir': stats_dir,
                    'stats_text': stats_text
                }
            except Exception as e:
                msg = (
                    '准备 COIL-20 数据集失败: ' + str(e) +
                    '。若为下载失败，请手动将 COIL-20 压缩包放置到 data/raw/ 并重试。'
                )
                coil20_jobs[job_id_local]['status'] = 'error'
                coil20_jobs[job_id_local]['message'] = msg

        th = threading.Thread(target=run_prepare_coil20, args=(job_id, zip_source))
        th.daemon = True
        th.start()

        return jsonify({'status': 'started', 'job_id': job_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/prepare_coil20_status')
def api_prepare_coil20_status():
    """查询 COIL-20 准备任务的状态。
    参数：job_id
    返回：{status, progress, message, result?}
    """
    job_id = request.args.get('job_id')
    if not job_id or job_id not in coil20_jobs:
        return jsonify({'status': 'error', 'message': '无效的 job_id'}), 404
    return jsonify(coil20_jobs[job_id])

# 清理可能导致BatchNorm参数冲突的旧模型文件
def cleanup_old_model_files():
    """清理所有可能导致BatchNorm参数冲突的旧模型相关文件"""
    # 仅清理那些明显是临时或中断的缓存文件，保留最终保存的模型和参数文件
    temp_files = [
        'static/interrupted_model.pth',
        'static/interrupted_results.json',
        'temp_debug.pt'
    ]

    cleaned_files = []
    for old_file in temp_files:
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                cleaned_files.append(os.path.basename(old_file))
            except Exception as e:
                print(f"清理临时文件 {old_file} 时出错: {str(e)}")

    if cleaned_files:
        print(f"已清理以下临时文件: {', '.join(cleaned_files)}")
    else:
        print("没有发现需要清理的临时模型文件")


def load_saved_model_on_start():
    """尝试在应用启动时加载已保存的模型和参数，恢复上次训练状态供预测使用"""
    global current_model, model_params, train_results, class_names, device

    # 优先加载完整保存的模型备份
    candidate_full = 'static/best_model_complete.pth'
    candidate_state = 'static/cnn_model.pth'
    candidate_no_bn = 'static/cnn_model_no_batchnorm.pth'
    params_path = 'static/model_params.json'
    results_path = 'static/train_results.json'

    # 尝试读取类别和模型参数
    try:
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                jr = json.load(f)
                if 'class_names' in jr and jr['class_names']:
                    class_names = jr['class_names']
                # 也保留训练结果
                train_results = {
                    'train_losses': jr.get('train_losses', []),
                    'test_losses': jr.get('test_losses', []),
                    'train_accuracies': jr.get('train_accuracies', []),
                    'test_accuracies': jr.get('test_accuracies', [])
                }
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                model_params = json.load(f)
    except Exception as e:
        print(f"加载参数/结果文件时出错: {e}")

    # 如果存在完整模型文件，直接加载
    try:
        if os.path.exists(candidate_full):
            print(f"启动: 加载完整模型 {candidate_full} ...")
            current_model = torch.load(candidate_full, map_location=device)
            current_model.to(device)
            current_model.eval()
            print("启动: 完整模型加载成功")
            return

        # 否则尝试加载state_dict
        if os.path.exists(candidate_state):
            print(f"启动: 加载模型权重 {candidate_state} ...")
            # 需要根据保存的model_params来构建模型结构
            if model_params and 'model_type' in model_params and class_names:
                num_classes = len(class_names)
                if model_params.get('model_type') == 'simple':
                    current_model = SimpleCNN(num_classes=num_classes)
                elif model_params.get('model_type') == 'improved':
                    current_model = ImprovedCNN(num_classes=num_classes)
                else:
                    current_model = AdvancedCNN(num_classes=num_classes)
            else:
                # 如果缺少参数或类别，尝试使用AdvancedCNN作为默认
                print('启动: 未找到模型参数或类别信息，使用 AdvancedCNN 作为默认结构')
                current_model = AdvancedCNN(num_classes=10)

            # 使用 train_utils.load_model 以获得更好的兼容性
            try:
                current_model = load_model(current_model, candidate_state)
            except Exception as e:
                print(f"启动: 使用兼容加载失败: {e}")
                try:
                    # 作为回退，直接load_state_dict
                    state = torch.load(candidate_state, map_location=device)
                    current_model.load_state_dict(state, strict=False)
                    current_model.to(device)
                    current_model.eval()
                except Exception as e2:
                    print(f"启动: 直接加载state_dict也失败: {e2}")
            print('启动: 模型权重加载完成')
            return

        # 最后尝试不含BatchNorm统计的权重
        if os.path.exists(candidate_no_bn):
            print(f"启动: 加载备份模型权重 {candidate_no_bn} ...")
            if model_params and 'model_type' in model_params and class_names:
                num_classes = len(class_names)
                if model_params.get('model_type') == 'simple':
                    current_model = SimpleCNN(num_classes=num_classes)
                elif model_params.get('model_type') == 'improved':
                    current_model = ImprovedCNN(num_classes=num_classes)
                else:
                    current_model = AdvancedCNN(num_classes=num_classes)
            else:
                current_model = AdvancedCNN(num_classes=10)

            try:
                state = torch.load(candidate_no_bn, map_location=device)
                current_model.load_state_dict(state, strict=False)
                current_model.to(device)
                current_model.eval()
                print('启动: 备份模型加载成功')
            except Exception as e:
                print(f"启动: 备份模型加载失败: {e}")
            return
    except Exception as e:
        print(f"启动: 加载模型时出错: {e}")

    print('启动: 未检测到已保存的模型')


def save_model_bundle(model, model_params, train_results, class_names, base_dir='static/models', name_prefix=None, test_loader=None):
    """
    将模型及其参数、训练结果、类别名单打包保存到独立目录中，目录名为 <name_prefix>_YYYYmmdd_HHMMSS
    返回保存的模型目录路径
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{name_prefix}_" if name_prefix else ''
        folder_name = f"{prefix}{ts}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=False)

        # 保存 state_dict
        state_path = os.path.join(folder_path, 'model_state.pth')
        torch.save(model.state_dict(), state_path)

        # 保存完整模型（备份）
        complete_path = os.path.join(folder_path, 'model_complete.pth')
        try:
            torch.save(model, complete_path)
        except Exception:
            # 如果保存整个模型对象失败（例如模型包含本地类），忽略
            pass

        # 保存参数、训练结果和类别名称
        if model_params is not None:
            with open(os.path.join(folder_path, 'model_params.json'), 'w', encoding='utf-8') as f:
                json.dump(model_params, f)

        if train_results is not None:
            with open(os.path.join(folder_path, 'train_results.json'), 'w', encoding='utf-8') as f:
                json.dump(train_results, f)

        if class_names is not None:
            with open(os.path.join(folder_path, 'class_names.json'), 'w', encoding='utf-8') as f:
                json.dump(class_names, f)

        # 生成并保存训练曲线图（如果有训练结果）
        try:
            if train_results is not None:
                plot_path = os.path.join(folder_path, 'training_plot.png')
                try:
                    # plot_results 已确保创建目录
                    plot_results(train_results, save_path=plot_path)
                except Exception as e:
                    print(f"保存训练曲线图失败: {e}")

        except Exception:
            pass

        # 尝试生成混淆矩阵缩略图（需要 model、test_loader、class_names）
        try:
            if model is not None and test_loader is not None and class_names is not None:
                try:
                    # 计算混淆矩阵
                    cm = get_confusion_matrix(model, test_loader, len(class_names), device)
                    cm_path = os.path.join(folder_path, 'confusion_matrix.png')
                    try:
                        plot_confusion_matrix(cm, class_names, save_path=cm_path)
                        plt_close = True
                    except Exception as e:
                        print(f"绘制混淆矩阵失败: {e}")
                except Exception as e:
                    print(f"生成混淆矩阵时出错: {e}")
        except Exception:
            pass

        # 保存元数据
        meta = {
            'folder': folder_name,
            'saved_at': ts,
            'state_path': os.path.relpath(state_path),
            'complete_path': os.path.relpath(complete_path)
        }
        with open(os.path.join(folder_path, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)

        # 更新 latest pointer
        latest_path = os.path.join(base_dir, 'latest.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump({'latest': folder_name, 'saved_at': ts}, f)

        print(f"已保存模型包到: {folder_path}")
        return folder_path
    except Exception as e:
        print(f"保存模型包时出错: {e}")
        raise

if __name__ == '__main__':
    # 清理旧模型文件，防止BatchNorm参数冲突
    cleanup_old_model_files()
    
    # 如果使用CUDA，清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA缓存已清理")
    
    # 确保必要的目录存在
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/samples', exist_ok=True)
    
    # 在启动时尝试加载已有的已保存模型（如果存在）
    load_saved_model_on_start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)