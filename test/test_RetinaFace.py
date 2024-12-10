import torch
import os
import sys
import cv2 as cv
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.RetinaFace import RetinaFace
from models.net_utils import cfg_mnet, cfg_re50, check_keys, remove_prefix

def test():
  model = RetinaFace(cfg_re50, phase = "test")
  pretrained_dict = torch.load("./weights/Resnet50_Final.pth", map_location=lambda storage, loc: storage)
  if "state_dict" in pretrained_dict.keys():
      pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
  else:
      pretrained_dict = remove_prefix(pretrained_dict, 'module.')
  check_keys(model, pretrained_dict)
  model.load_state_dict(pretrained_dict, strict=False)
  model.eval()

  img = cv.imread("D:\\Pictures\\600x800.jpg", cv.IMREAD_COLOR)  # 默认读取为 HWC 格式
  img = np.float32(img)
  img = cv.resize(src=img, dsize=(640, 640))
  img -= (104, 117, 123)
  img = img.transpose(2, 0, 1)
  img = torch.from_numpy(img).unsqueeze(0)

import cv2

def draw_face_boxes(image, final_boxes):
    """
    在图像上绘制人脸边界框和置信度。
    :param image: 输入图像 (numpy array)
    :param final_boxes: 检测到的边界框列表，每个框的格式为 [x_min, y_min, x_max, y_max, confidence]
    """
    height, width = image.shape[:2]

    for box in final_boxes:
        if len(box) < 5:
            continue  # 确保边界框格式正确

        # 提取边界框坐标和置信度
        x_min, y_min, x_max, y_max, confidence = box
        x_min, y_min, x_max, y_max = (
            int(x_min * width),
            int(y_min * height),
            int(x_max * width),
            int(y_max * height),
        )

        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 绘制置信度
        label = f"Conf: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image

if __name__ == "__main__":
    # 加载图像
    img = cv.imread("D:\\Pictures\\600x800.jpg", cv.IMREAD_COLOR)  
    # img = cv.resize(src=img, dsize=(640, 640))

    # 示例检测结果（final_boxes）
    final_boxes = [
        [0.312132, 0.200474, 0.650681, 0.534067, 0.999514]
    ]

    # 绘制人脸边界框
    output_image = draw_face_boxes(img, final_boxes)

    # 显示图像
    cv2.imshow("Detected Faces", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#    test()