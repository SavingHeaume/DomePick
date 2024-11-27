import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path="model.trt"):
    # 创建 TensorRT builder 和 config
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # 创建构建配置
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # 从 ONNX 文件解析模型
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 优化配置
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)

        # 构建 TensorRT 引擎
        print("Building TensorRT Engine...")
        # 使用 build_serialized_network() 替代旧的 build_engine()
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("ERROR: Failed to build the engine.")
            return None

        # 保存引擎到文件
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_file_path}")
        return serialized_engine

if __name__ == "__main__":
    build_engine("./weights/age_gender.onnx", "./weights/age_gender.trt")
    build_engine("./weights/retinaface.onnx", "./weights/retinaface.trt")
