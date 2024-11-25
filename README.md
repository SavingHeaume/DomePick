# DOMEPICK

# 模型

人脸检测模型

- NVIDIA FaceDetectIR
    
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facedetectir
    
- RetinaFace MobileNet
    
    [https://github.com/serengil/retinaface](https://github.com/serengil/retinaface)
    
    https://zhuanlan.zhihu.com/p/379730820
    
    [https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
    
    https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface
    
    [https://github.com/ppogg/Retinaface_Ghost](https://github.com/ppogg/Retinaface_Ghost)
    

年龄性别分类模型

- MobileNetV2
    
    https://blog.csdn.net/baidu_36913330/article/details/120079096
    

数据集

- **Adience**：专注于年龄和性别分类。
- **IMDB-WIKI**：超大规模年龄性别标注数据集。
- **UTKFace**：广泛使用的带年龄性别标签的人脸数据集。

# 加速

- **模型压缩与量化**：
    - **剪枝**：去掉不重要的通道或层。
    - **量化**：
        - 使用 TensorRT 的 INT8 精度优化，将权重从 FP32 降低到 INT8，显著加速推理。
        - 在训练后量化模型时，使用校准数据集保证精度。
        - 示例：
            
            ```bash
            trtexec --onnx=model.onnx --int8 --saveEngine=model.trt
            ```
            
- **使用 TensorRT**：
    
    https://developer.nvidia.com/tensorrt
    
    https://zhuanlan.zhihu.com/p/371239130
    
- **精简模型输入**：
    - 减少模型输入分辨率，如从 224x224 降到 112x112，节省计算量。

- 使用 **trtexec** 测试 TensorRT 性能：
    
    ```bash
    bash
    复制代码
    trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
    
    ```
    
- TensorRT 可视化工具（如 **Nsight Systems**）查看性能瓶颈。

# 优化

- **人脸检测**：
    - 仅对视频流中变化较大的帧运行完整人脸检测，其他帧直接使用追踪算法（如 SORT）。
- **多任务分发**：
    - 使用 Jetson 的 GPU 和 DLA（深度学习加速器）分别运行不同的任务，例如 DLA 用于分类，GPU 用于检测。