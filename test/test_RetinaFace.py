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
  img -= (104, 117, 123)
  img = img.transpose(2, 0, 1)
  img = torch.from_numpy(img).unsqueeze(0)


  a, b, c = model(img)
  print(a)
  print(b)
  print(c)

  for i, row in enumerate(b[0]):
    if row[1] > 0.95:  # 判断第二个值是否超过阈值
        print(f"Index: {i}, Array: {row.tolist()}")

if __name__ == "__main__":
   test()