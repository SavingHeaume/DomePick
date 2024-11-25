import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

def load_model(model):
    return YOLO(model)

def on_vidio(model):
    video_file = st.file_uploader("上传视频文件", type=['mp4', 'mov', 'avi'])
        
    if video_file is not None:
            # 保存上传的视频到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        
        # 打开视频文件
        video = cv2.VideoCapture(tfile.name)
        process_video(video, model)

def on_cam(model):
    video = cv2.VideoCapture(0)
    process_video(video, model)

def on_image(model):
    image_file = st.file_uploader("上传图片文件", type=['jpg', 'png', 'jpeg'])
    
    if image_file is not None:
        # 保存上传的图片到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(image_file.read())
        
        # 打开图片文件
        image = cv2.imread(tfile.name)
        results = model(image)
        
        # 绘制检测结果
        annotated_image = results[0].plot()
        
        # 显示带注释的图片
        st.image(annotated_image, channels="BGR")

def main():
    st.title("YOLOv11 Object Detection")
    
    # 选择输入源
    input_source = st.radio("选择输入源", ("上传视频", "使用摄像头", "上传图片"))
    
    model_option = st.selectbox(
        "选择YOLOv11模型",
        ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
    )

    model = load_model(model_option)
    
    if input_source == "上传视频":
        on_vidio(model)
    
    elif input_source == "使用摄像头":
        on_cam(model)
    elif input_source == "上传图片":
        on_image(model)

def process_video(video, model):
    # 创建一个占位符来显示视频帧
    stframe = st.empty()

    while video.isOpened():
        ret, frame = video.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not ret:
            break
        
        results = model(frame)
        
        # 在帧上绘制检测结果
        annotated_frame = results[0].plot()
        
        # 显示带注释的帧
        stframe.image(annotated_frame, channels="BGR")
    
    video.release()

if __name__ == '__main__':
    main()
