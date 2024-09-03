#!/user/bin/env python
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel, RCCNet_v0_5
from rotate_captcha_crack.utils import process_captcha


# 主函数
def process_images(image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        with torch.no_grad():
            cls_num = DEFAULT_CLS_NUM
            model = RotNetR(cls_num=cls_num, train=False)
            # model = RCCNet_v0_5(train=False)
            model_path = WhereIsMyModel(model).with_index(-1).model_dir / "best.pth"
            print(f"Use model: {model_path}")
            model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
            model = model.to(device=device)
            model.eval()

            img = Image.open(
                image_path)

            img_ts = process_captcha(img)
            img_ts = img_ts.to(device=device)

            predict = model.predict(img_ts)
            degree = predict * 360
            print(f"Predict degree: {degree:.4f}°")

            img = img.rotate(
                -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255, 0)
            )  # use neg degree to recover the img

            # 文件路径
            file_path = Path(image_path)
            # 从路径中提取文件名
            file_name = file_path.name
            # 打印文件名
            print(file_name)
            # 保存旋转后的图片
            img.save(save_dir + f'/{file_name.replace("jpeg", "png")}')


# 批量测试旋转成功效果，用于观测而已
if __name__ == '__main__':
    process_images('/Users/liuqiang/Downloads/rotating', '/Users/liuqiang/Downloads/rotating_added222')
