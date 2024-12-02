import torch
import os
import sys
import onnx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.AgeGenderModel import AgeGenderModel 
from models.RetinaFace import RetinaFace
from models.net_utils import cfg_mnet, cfg_re50
from models.net_utils import check_keys, remove_prefix

def export_AgeGender_to_onnx(model_path='best_age_gender_model.pth', 
                   onnx_path='age_gender_model.onnx', 
                   input_size=(1, 3, 224, 224)):
    """
    将训练好的模型导出为ONNX格式
    
    :param model_path: 模型权重路径
    :param onnx_path: ONNX模型导出路径
    :param input_size: 输入张量大小
    """
    # 初始化模型
    model = AgeGenderModel(num_classes_gender=2, num_classes_age=8)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path))
    
    # 设置为评估模式
    model.eval()
    
    # 创建虚拟输入
    x = torch.randn(input_size)
    
    # 动态轴配置
    dynamic_axes = {
        '0': {
            'gender': [0],
            'age': [0]
        }
    }
    
    # 导出ONNX模型
    torch.onnx.export(
        model,                   # 模型 
        x,                       # 示例输入
        onnx_path,               # 输出路径
        export_params=True,      # 导出模型参数
        opset_version=12,        # ONNX算子集版本
        do_constant_folding=True, # 常量折叠优化
        input_names=['input'],   # 输入节点名称
        output_names=['gender', 'age'],  # 输出节点名称
        dynamic_axes={
            'input': {0: 'batch_size'},
            'gender': {0: 'batch_size'},
            'age': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证成功")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")


def export_retinaface_to_onnx(output_path='retinaface.onnx'):
    model = RetinaFace(cfg_re50, phase = "test")
    pretrained_dict = torch.load("./weights/Resnet50_Final.pth", map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()

    # 创建示例输入
    input_shape = (1, 3, 640, 640)  
    x = torch.randn(input_shape)

    # 动态轴设置
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'bbox_output': {0: 'batch_size'},
        'class_output': {0: 'batch_size'},
        'landmark_output': {0: 'batch_size'}
    }

    # 导出模型
    torch.onnx.export(
        model,                   # 模型
        x,                       # 示例输入
        output_path,             # 输出文件路径
        export_params=True,      # 存储训练后的权重
        opset_version=11,        # ONNX 算子集版本
        do_constant_folding=True,# 是否执行常量折叠优化
        input_names=['input'],   # 输入节点名称
        output_names=[           # 输出节点名称
            'bbox_output', 
            'class_output', 
            'landmark_output'
        ],
        dynamic_axes=dynamic_axes
    )

    print(f"Model exported successfully to {output_path}")

def main():
    export_AgeGender_to_onnx(
        model_path='./weights/best_adience_model.pth', 
        onnx_path='./weights/age_gender.onnx'
    )

    export_retinaface_to_onnx(
        output_path="./weights/retinaface.onnx"
    )

if __name__ == "__main__":
    main()