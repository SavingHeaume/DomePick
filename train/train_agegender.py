import os
import sys
import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.AgeGenderModel import AgeGenderModel 

# 年龄区间映射
AGE_GROUPS = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 13)': 2,
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 43)': 5,
    '(48, 53)': 6,
    '(60, )': 7
}

class AdienceDataset(Dataset):
    def __init__(self, data, base_path, transform=None):
        """
        :param data: 包含图像信息的DataFrame
        :param base_path: aligned数据集基础路径
        :param transform: 图像变换
        """
        self.data = data
        self.base_path = base_path
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 构建图像完整路径
        image_path = os.path.join(
            self.base_path, 
            str(row['user_id']), 
            "landmark_aligned_face." + row['face_id'] + "." + row['original_image']
        )
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 性别编码
        gender = 0 if row['gender'] == 'f' else 1
        
        # 年龄区间编码
        age_group = AGE_GROUPS[row['age']]
        
        # 图像变换
        if self.transform:
            image = self.transform(image)
        
        return image, gender, age_group

def parse_adience_fold_file(fold_file_path):
    """
    解析Adience数据集的fold文件
    :param fold_file_path: fold文件路径
    :return: DataFrame
    """
    # 定义列名
    columns = [
        'user_id', 'original_image', 'face_id', 'age', 
        'gender', 'x', 'y', 'dx', 'dy', 
        'tilt_ang', 'fiducial_yaw_angle', 'fiducial_score'
    ]
    
    # 读取文件
    data = pd.read_csv(fold_file_path, sep='\t', names=columns, header=None)
    
    # 过滤有效的年龄区间
    data = data[data['age'].isin(AGE_GROUPS.keys())]
    
    # 过滤性别
    data = data[data['gender'].isin(['f', 'm'])]
    
    return data

def load_adience_dataset(base_path, fold_files):
    """
    加载Adience数据集
    :param base_path: 数据集基础路径
    :param fold_files: fold文件列表
    :return: 合并后的数据集DataFrame
    """
    all_data = []
    
    for fold_file in fold_files:
        fold_path = os.path.join(base_path, fold_file)
        fold_data = parse_adience_fold_file(fold_path)
        all_data.append(fold_data)
    
    # 合并所有fold数据
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 去重
    combined_data = combined_data.drop_duplicates(
        subset=['user_id', 'original_image', 'face_id']
    )
    
    return combined_data

def train_age_gender_model(base_path):
    """
    训练年龄和性别分类模型
    :param base_path: Adience数据集基础路径
    """
    # fold文件列表
    fold_files = [
        'fold_0_data.txt', 
        'fold_1_data.txt', 
        'fold_2_data.txt', 
        'fold_3_data.txt', 
        'fold_4_data.txt'
    ]
    
    # 数据加载
    aligned_path = os.path.join(base_path, 'aligned')
    dataset_data = load_adience_dataset(base_path, fold_files)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        dataset_data, test_size=0.2, stratify=dataset_data['age'], random_state=42
    )
    
    # 创建数据集和数据加载器
    train_dataset = AdienceDataset(train_data, aligned_path, transform)
    val_dataset = AdienceDataset(val_data, aligned_path, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AgeGenderModel(num_classes_gender=2, num_classes_age=8).to(device)
    
    # 损失函数
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()  # 改为分类损失
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        gender_acc = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, genders, ages in progress_bar:
            images, genders, ages = images.to(device), genders.to(device), ages.to(device)
            
            optimizer.zero_grad()
            pred_genders, pred_ages = model(images)
            
            loss_gender = criterion_gender(pred_genders, genders)
            loss_age = criterion_age(pred_ages.squeeze(), ages)
            loss = loss_gender + 0.5 * loss_age
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算性别分类准确率
            gender_pred = torch.argmax(pred_genders, dim=1)
            gender_acc += (gender_pred == genders).float().mean().item()
            
            progress_bar.set_postfix({
                'Loss': loss.item(), 
                'Gender Acc': gender_acc / (progress_bar.n + 1)
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_gender_acc = 0.0
        
        with torch.no_grad():
            for images, genders, ages in val_loader:
                images, genders, ages = images.to(device), genders.to(device), ages.to(device)
                
                pred_genders, pred_ages = model(images)
                
                loss_gender = criterion_gender(pred_genders, genders)
                loss_age = criterion_age(pred_ages.squeeze(), ages)
                loss = loss_gender + 0.5 * loss_age
                
                val_loss += loss.item()
                
                gender_pred = torch.argmax(pred_genders, dim=1)
                val_gender_acc += (gender_pred == genders).float().mean().item()
        
        # 打印训练和验证指标
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Gender Acc: {gender_acc/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Gender Acc: {val_gender_acc/len(val_loader):.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '.\\weights\\best_adience_model.pth')
    
    print("模型训练完成")


if __name__ == "__main__":
    base_path = ".\\datasets\\Adience\\"
    train_age_gender_model(base_path)
