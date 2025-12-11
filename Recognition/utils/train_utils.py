import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    """
    训练模型并记录训练过程
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
    返回:
        results: 包含训练和测试损失、准确率的字典
    """
    # 确定设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 将模型移至设备
    model.to(device)
    
    # 初始化记录列表
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 开始训练
    for epoch in range(epochs):
        print(f"\n第 {epoch+1}/{epochs} 轮训练")
        
        # 训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                # 设置进度条描述
                tepoch.set_description(f"Epoch {epoch+1}")
                
                # 将数据移至设备
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条信息
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * correct / total)
        
        # 计算本轮训练的平均损失和准确率
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct / total
        
        # 记录训练指标
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # 在测试集上评估模型
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # 记录测试指标
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # 打印本轮训练结果
        print(f'训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.2f}%')
        print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
    
    # 返回训练结果
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def evaluate_model(model, test_loader, criterion, device):
    """
    在测试集上评估模型
    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 运行设备
    返回:
        test_loss: 测试集平均损失
        test_acc: 测试集准确率
    """
    # 评估模式
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 将数据移至设备
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def plot_results(results, save_path=None):
    """
    绘制训练结果图表
    参数:
        results: 包含训练结果的字典
        save_path: 图表保存路径
    返回:
        save_path: 保存的图表路径
    """
    epochs = range(1, len(results['train_losses']) + 1)
    
    # 创建图表
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_losses'], 'b-', label='训练损失')
    plt.plot(epochs, results['test_losses'], 'r-', label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_accuracies'], 'b-', label='训练准确率')
    plt.plot(epochs, results['test_accuracies'], 'r-', label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('轮数')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'static/training_results.png'
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def save_model(model, model_path):
    """
    保存模型
    参数:
        model: 要保存的模型
        model_path: 保存路径
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型状态字典
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

def load_model(model, model_path):
    """
    加载模型，支持增强的错误处理和BatchNorm参数兼容性
    参数:
        model: 模型架构
        model_path: 模型文件路径
    返回:
        model: 加载权重后的模型
    """
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"开始加载模型: {model_path}")
    
    try:
        # 先加载文件对象
        loaded = torch.load(model_path, map_location=device)

        # 如果保存的是完整的模型对象（nn.Module），直接返回它
        if isinstance(loaded, nn.Module):
            print(f"加载到完整模型对象，从: {model_path}")
            loaded.to(device)
            loaded.eval()
            return loaded

        # 否则假设是 state_dict-like 的对象（dict 或 OrderedDict），尝试加载到提供的 model 中
        state_dict = loaded
        try:
            # 尝试严格加载模型状态字典
            model.load_state_dict(state_dict, strict=True)
            print(f"模型已严格加载从: {model_path}")
        except RuntimeError as e:
            error_msg = str(e)
            if 'running_mean' in error_msg or 'running_var' in error_msg or 'size mismatch' in error_msg:
                print(f"BatchNorm参数不匹配，尝试兼容加载: {error_msg}")
                # 过滤掉不匹配的BatchNorm参数
                filtered_state_dict = {}
                for name, param in state_dict.items():
                    if name in model.state_dict():
                        if model.state_dict()[name].size() == param.size():
                            filtered_state_dict[name] = param
                        else:
                            print(f"跳过尺寸不匹配的参数: {name} (模型: {model.state_dict()[name].size()}, 文件: {param.size()})")

                # 加载过滤后的状态字典
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"已加载匹配的参数，共 {len(filtered_state_dict)} 个参数成功加载")

                # 重置所有BatchNorm层的统计信息
                for name, module in model.named_modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.running_mean = torch.zeros(module.num_features, device=device)
                        module.running_var = torch.ones(module.num_features, device=device)
                        module.num_batches_tracked = torch.tensor(0, device=device)
                        print(f"已重置BatchNorm层 {name} 的统计信息")
            else:
                raise
    except Exception as e:
        print(f"加载模型时发生错误: {str(e)}")
        raise
    
    # 将模型移至设备并设置为评估模式
    model.to(device)
    model.eval()
    
    print(f"模型加载完成并移至设备: {device}")
    return model

def get_confusion_matrix(model, test_loader, num_classes, device):
    """
    计算混淆矩阵
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_classes: 类别数量
        device: 运行设备
    返回:
        confusion_matrix: 混淆矩阵
    """
    # 初始化混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    # 评估模式
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # 更新混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix.numpy()

def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """
    绘制混淆矩阵
    参数:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    # 设置类别标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在矩阵中显示数值
    fmt = '.0f'  # 整数格式
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
    
    return plt

def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay=0.1, decay_epochs=30):
    """
    调整学习率
    参数:
        optimizer: 优化器
        epoch: 当前轮数
        initial_lr: 初始学习率
        lr_decay: 学习率衰减率
        decay_epochs: 每隔多少轮衰减一次
    """
    lr = initial_lr * (lr_decay ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_classification_report(model, test_loader, class_names, device):
    """
    获取分类报告
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        device: 运行设备
    返回:
        report: 分类报告文本
    """
    from sklearn.metrics import classification_report
    
    # 初始化真实标签和预测标签列表
    true_labels = []
    predicted_labels = []
    
    # 评估模式
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # 收集标签
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    
    # 生成分类报告
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=class_names
    )
    
    return report