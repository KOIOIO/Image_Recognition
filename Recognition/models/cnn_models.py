import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络模型
    包含2个卷积层和2个全连接层
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入通道3，输出通道16，卷积核大小3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 第二个卷积层：输入通道16，输出通道32，卷积核大小3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 全局平均池化层，避免硬编码特征图大小
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 输出层：输出类别数
        self.fc = nn.Linear(32, num_classes)
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 应用第一个卷积层，然后是ReLU激活和池化
        x = self.pool(F.relu(self.conv1(x)))
        # 应用第二个卷积层，然后是ReLU激活和池化
        x = self.pool(F.relu(self.conv2(x)))
        # 使用自适应全局平均池化
        x = self.avg_pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 应用Dropout
        x = self.dropout(x)
        # 应用输出层
        x = self.fc(x)
        return x


class ImprovedCNN(nn.Module):
    """
    改进的卷积神经网络模型
    增强复杂度以达到60%以上准确率
    """
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # 增强的卷积特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 输入 -> 1/2尺寸
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 尺寸再次减半
            
            # 第三个卷积块 - 新增
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 尺寸再次减半
        )
        
        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 全局平均池化
        x = self.avg_pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """残差块基础结构"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样（用于匹配跳跃连接的维度）
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # 第一个卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积操作
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接处理
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out


class AdvancedCNN(nn.Module):
    """
    高级卷积神经网络模型
    采用改进的ResNet风格设计，增强深度和特征提取能力，目标85%以上准确率
    """
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # 初始特征提取层
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块组
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1):
        """创建残差块层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块处理
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avg_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x