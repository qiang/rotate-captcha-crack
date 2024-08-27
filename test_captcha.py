import argparse
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel, RCCNet_v0_5
from rotate_captcha_crack.utils import process_captcha

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        cls_num = DEFAULT_CLS_NUM
        model = RotNetR(cls_num=cls_num, train=False)
        # model = RCCNet_v0_5(train=False)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        # model.load_state_dict(torch.load(str('/Users/liuqiang/Downloads/rotate-captcha-crack/models/RotNetR/240826_15_01_32_001/best.pth'), map_location='cpu'))
        model = model.to(device=device)
        model.eval()

        # img = Image.open("datasets/tieba/1615096444.jpg")
        # path = '/Users/liuqiang/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/rotating/'
        path = '/Users/liuqiang/Downloads/rotating/'
        # path = '/home/tencent/deploy/code/rotating/'
        img = Image.open(
            path + "a0e931b159b30fc3e7e76c960560b404_rotating_pic.jpeg")

        home_dir = os.path.expanduser("~")
        # 拼接路径
        # pic = f'{home_dir}/Downloads/capthca/rest/zt/captcha/rotating/pic'
        # print("测试环境，加载测试环境代码-", pic)
        # img = Image.open(pic)

        img_ts = process_captcha(img)
        img_ts = img_ts.to(device=device)

        predict = model.predict(img_ts)
        degree = predict * 360
        print(f"Predict degree: {degree:.4f}°")

    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    plt.figure("debug")
    plt.imshow(img)
    plt.show()
