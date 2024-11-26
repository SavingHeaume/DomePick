import torch 
import torch.nn as nn 
import torchvision.models as models 
import os

def get_mobilenet_backbone(weights_dir='./weights'):
    """
    从本地加载MobileNetV2预训练权重
    
    :param weights_dir: 权重文件存储目录
    :return: MobileNetV2特征提取器
    """
    # 确保权重目录存在
    os.makedirs(weights_dir, exist_ok=True)
    
    # 权重文件名和URL
    weights_filename = 'mobilenet_v2-7ebf99e0.pth'
    weights_url = f'https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth'
    weights_path = os.path.join(weights_dir, weights_filename)
    
    # 如果本地没有权重文件，则下载
    if not os.path.exists(weights_path):
        print(f"下载预训练权重到 {weights_path}")
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, 
            model_dir=weights_dir
        )
        torch.save(state_dict, weights_path)
    else:
        print(f"从本地加载预训练权重 {weights_path}")
        state_dict = torch.load(weights_path)
    
    # 初始化模型并加载权重
    backbone = models.mobilenet_v2(weights=None)
    backbone.load_state_dict(state_dict)
    
    return backbone.features

class AgeGenderModel(nn.Module):
    def __init__(self, num_classes_gender=2, num_classes_age=8):
        super(AgeGenderModel, self).__init__()
        # 使用预训练的MobileNetV2作为特征提取器
        self.features = get_mobilenet_backbone()
        
        # 自适应池化
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 性别分类全连接层
        self.fc_gender = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_gender)
        )
        
        # 年龄分类全连接层
        self.fc_age = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_age)
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # 性别和年龄预测
        gender = self.fc_gender(x)
        age = self.fc_age(x)
        
        return gender, age
