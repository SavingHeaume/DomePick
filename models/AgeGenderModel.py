import torch 
import torch.nn as nn 
import torchvision.models as models 
from models.net_utils import get_backbone

class AgeGenderModel(nn.Module):
    def __init__(self, num_classes_gender=2, num_classes_age=8):
        super(AgeGenderModel, self).__init__()
        # 使用预训练的MobileNetV2作为特征提取器
        get_backbone('mobilenet_v2-7ebf99e0.pth', f'https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth')
        backbone = models.mobilenet_v2(weights=None)
        backbone.load_state_dict(torch.load("./weights/mobilenet_v2-7ebf99e0.pth"))
        self.features = backbone.features
        
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
