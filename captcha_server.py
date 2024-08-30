#!/user/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import os
import random
import time
import traceback
from pathlib import Path

import cv2
import requests
from urllib.parse import urlparse, parse_qs

import matplotlib.pyplot as plt
import torch
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel, RCCNet_v0_5
from rotate_captcha_crack.utils import process_captcha

from flask import Flask, request, jsonify

app = Flask(__name__)

test_captcha_sn = 'Cgp6dC5jYXB0Y2hhEtsCT6K2JR-TLgq0qjnr-HSWotdl3xhqU55LI-lgttr0ohzWOva0LrEbLGfbyQIiO3vCk85LhQlhHt6yjDLXBAP78X2s4gIjobBeBDwmwGslO7m3EEB9HCM2GlOUgZMni-GZBRK6up4piHRaYfZCFD8RahMlIyQPC9JyHTMICfB3LeozGAmAeRm5H_XBlnK5Fwv0F0pb_zpjjQSDDahpf62swde2APCsW2PkpHTbQ0hhzsq7w0BvChH0i3bjcs3jWWCae3OT-Cfl3GiVy8zL0ME1W_KzIicZ2-9wRON11qhhnIHXeOaZV-fGJwySHJMewyo2rQL5mutCIxD1wGtyfd6vQKvL7v4DmZkq9Ws18LljeavHxe3kO9t4sLUf3T9-nW2yxPGxjfbZzQYgwMjDuwwI2ctolMgAkTj9Nzdg49Cp9HZ4scE3K9TEf27arfwoQ3ypR9vcNY5jZXjJ9DgaEleIHYY8IfeWArTHzyF818xZgigFMAI'


def download_image(url, save_path, file_name=None):
    """
    下载图片并保存到指定路径。

    :param url: 图片的URL
    :param save_path: 保存图片的文件夹路径
    :param file_name: 保存图片的文件名（可选）。如果未指定，则使用URL中的文件名。
    :return: 成功返回文件路径，失败返回 None
    """
    # log.debug(f"download_image invoke : url== {url} \n save_path=={save_path} \n file_name={file_name}")

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 使用URL中的文件名（如果未提供文件名）
    if not file_name:
        file_name = os.path.basename(url)

    # 完整的文件路径
    file_path = os.path.join(save_path, file_name)

    try:
        # 发送HTTP GET请求获取图片内容
        response = requests.get(url, timeout=20)
        response.raise_for_status()  # 检查请求是否成功

        # 将图片保存到指定路径
        with open(file_path, 'wb') as file:
            file.write(response.content)

        # log.debug(f"download_image 图片已成功下载并保存为 {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        # log.debug(f"download_image 下载失败：{e}")
        print(f"download_image 失败:  \n {e}")
        return None


def get_cut_point_in_bg(bg_img_path, cut_img_path):
    """
    计算滑块在背景图片的位置,并返回x轴占比
    :param bg_img_path:  背景图片
    :param cut_img_path:  缺口图片
    :return:    缺口点在背景图中的x轴占比
    """
    # log.debug(f"get_cut_point_in_bg invoke : bg_img_path== {bg_img_path} \n cut_img_path=={cut_img_path}")
    if not os.path.exists(bg_img_path) or not os.path.exists(cut_img_path):
        raise FileNotFoundError("get_cut_point_in_bg cut or bg not found!!")

    if not Path(bg_img_path).exists():
        # log.debug(f"get_cut_point_in_bg return: {bg_img_path} 不存在！！！")
        raise Exception("file not exist:" + bg_img_path)
    bg_img = cv2.imread(bg_img_path)
    cut_img = cv2.imread(cut_img_path)
    bg_width = bg_img.shape[1]

    if 'captcha_cut_grey_edge' in cut_img_path:
        cut_pic = cut_img
    else:
        cut_edge = cv2.Canny(cut_img, 100, 200)
        cut_pic = cv2.cvtColor(cut_edge, cv2.COLOR_GRAY2RGB)

    bg_edge = cv2.Canny(bg_img, 100, 200)
    bg_pic = cv2.cvtColor(bg_edge, cv2.COLOR_GRAY2RGB)

    res = cv2.matchTemplate(bg_pic, cut_pic, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # # max_loc 是滑块位置的左上角坐标
    # top_left = max_loc
    #
    # # 计算滑块的位置
    # bottom_right = (top_left[0] + 115, top_left[1] + 115)
    #
    # print(top_left)
    # # 在目标图片上绘制矩形框，标出滑块位置
    # cv2.rectangle(bg_img, top_left, bottom_right, 255, 2)
    #
    # # 显示结果
    # cv2.imshow('Detected', bg_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if abs(min_val) >= abs(max_val):
        left_top = min_loc
    else:
        left_top = max_loc
    print(max_loc)
    print(left_top)
    # x 轴滑动比例，因为界面会缩放。。。。
    x_rate = left_top[0] / bg_width

    phone_bg_width = 830
    phone_slide_distance = 830 * x_rate
    phone_slide_distance = int(phone_slide_distance)
    phone_slide_distance = phone_slide_distance - random.randint(10, 30)
    print("phone_slide_distance", phone_slide_distance)
    return phone_slide_distance


def cut_distance_gpt(bg, cut):
    """gpt 给生成的代码"""
    import cv2
    # 读取目标图片和滑块模板图片
    # target_image = cv2.imread('target_image.png', 0)
    target_image = cv2.imread(bg, 0)
    # slider_image = cv2.imread('slider_image.png', 0)
    slider_image = cv2.imread(cut, 0)

    # 使用模板匹配方法查找滑块位置
    result = cv2.matchTemplate(target_image, slider_image, cv2.TM_CCOEFF_NORMED)

    # 获取匹配结果的最大值和其对应的坐标
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # max_loc 是滑块位置的左上角坐标
    top_left = max_loc
    h, w = slider_image.shape

    # 计算滑块的位置
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在目标图片上绘制矩形框，标出滑块位置
    cv2.rectangle(target_image, top_left, bottom_right, 255, 2)

    # 显示结果
    cv2.imshow('Detected', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 输出滑块的位置
    print(f'Slider top-left position: {top_left}')


def get_captcha_sn_md5(captcha_url):
    print('get_captcha_sn_md5==> captcha_url', captcha_url)
    from urllib import parse
    if "?" in captcha_url:
        url_str = captcha_url[captcha_url.index('?') + 1:]
    else:
        url_str = captcha_url
    q = dict(parse.parse_qsl(
        url_str,
        keep_blank_values=True))
    captchaSn = q['captchaSn']

    # 创建 MD5 哈希对象
    md5 = hashlib.md5()
    # 更新哈希对象，字符串需要编码为字节
    md5.update(captchaSn.encode('utf-8'))
    # 获取十六进制的哈希值
    captcha_sn_md5 = md5.hexdigest()
    return captcha_sn_md5


@app.route('/do_captcha', methods=['POST', 'GET'])
def do_captcha():
    # return jsonify({
    #     'status': 0,
    #     'errorMsg': 'success',
    #     'distance': 200,
    #     'type': 'sliding',
    # }), 200

    print('调用 do_captcha ')
    captcha_url = ''
    if request.method == 'GET':
        # 处理 GET 请求
        captcha_url = request.args.get('captcha_url', default='', type=str)
    elif request.method == 'POST':
        try:
            data = request.json
        except Exception as e:
            data = None
        if data:
            captcha_url = data.get('captcha_url')
        else:
            # 如果POST数据不是JSON格式，尝试获取表单数据
            captcha_url = request.form.get('captcha_url')

    time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    error = ''
    if captcha_url:
        try:
            captcha_sn_md5 = get_captcha_sn_md5(captcha_url)
            if '/sliding/bgPic?captchaSn=' in captcha_url:
                if test_captcha_sn in captcha_url:
                    # 获取home目录
                    home_dir = os.path.expanduser("~")
                    # 拼接路径
                    bg_path = f'{home_dir}/WorkSpace/TencentPGC/capthca_api_mapping/rest/zt/captcha/sliding/bgPic'
                    cut_path = f'{home_dir}/WorkSpace/TencentPGC/capthca_api_mapping/rest/zt/captcha/sliding/cutPic'
                    print("测试环境，加载测试环境代码")
                else:
                    bg_path = download_image(captcha_url, f"./sliding/",
                                             f'{captcha_sn_md5}_slid_bg.jpeg')

                    cut_path = download_image(captcha_url.replace('/bgPic?captchaSn', '/cutPic?captchaSn'),
                                              f"./sliding/",
                                              f'{captcha_sn_md5}_slid_cut.jpeg')
                if not cut_path or not os.path.exists(cut_path):
                    home_dir = os.path.expanduser("~")
                    cut_path = f'{home_dir}/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/captcha_cut_grey_edge.png'

                print('bg_path', bg_path)
                if bg_path:
                    distance = get_cut_point_in_bg(bg_path,
                                                   cut_path)
                    return jsonify({
                        'status': 0,
                        'errorMsg': 'success',
                        'distance': distance,
                        'type': 'sliding',
                    }), 200
                else:
                    error = 'bg_path is null '
                    return jsonify({
                        'status': -1,
                        'errorMsg': error,
                        'distance': 600,
                        'type': 'sliding',
                    }), 200

            elif '/sliding/cutPic?captchaSn' in captcha_url:
                # 暂时没用，可以通过拼接url的方式下载这个滑块
                cut_path = download_image(captcha_url, f"./sliding/",
                                          f'{captcha_sn_md5}_slid_bg.jpeg')

            elif 'rotating/pic' in captcha_url:

                with torch.no_grad():
                    cls_num = DEFAULT_CLS_NUM
                    model = RotNetR(cls_num=cls_num, train=False)
                    # model = RCCNet_v0_5(train=False)
                    model_path = WhereIsMyModel(model).with_index(-1).model_dir / "best.pth"
                    print(f"Use model: {model_path}")
                    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
                    # model.load_state_dict(torch.load(str('/Users/liuqiang/Downloads/rotate-captcha-crack/models/RotNetR/240826_15_01_32_001/best.pth'), map_location='cpu'))
                    model = model.to(device=device)
                    model.eval()

                    if test_captcha_sn in captcha_url:
                        # 获取home目录
                        home_dir = os.path.expanduser("~")
                        # 拼接路径
                        pic = f'{home_dir}/WorkSpace/TencentPGC/capthca_api_mapping/rest/zt/captcha/rotating/pic'
                        print("测试环境，加载测试环境代码-", pic)

                    else:
                        pic = download_image(captcha_url, f"./rotating/",
                                             f'{captcha_sn_md5}_rotating_pic.jpeg')

                    # img = Image.open("datasets/tieba/1615096444.jpg")
                    # path = '/Users/liuqiang/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/rotating/'
                    # path = '/Users/liuqiang/Downloads/rotating/'
                    # path = '/home/tencent/deploy/code/rotating/'
                    # img = Image.open(
                    #     path + "0bb16cd4eb132348ef1e7d20155be8b0_rotating_pic.jpeg")
                    img = Image.open(
                        pic)
                    img_ts = process_captcha(img)
                    img_ts = img_ts.to(device=device)

                    predict = model.predict(img_ts)
                    degree = predict * 360
                    # print(f"Predict degree: {degree:.4f}°")
                    # 转换成距离
                    distance = int(predict * 720)
                    print('旋转：', predict, '角度：', degree, '转化成距离：', distance)
                return jsonify({
                    'status': 0,
                    'errorMsg': 'success',
                    'distance': distance,
                    'type': 'rotating',
                }), 200
            elif 'wordClick/pic' in captcha_url:
                pic = download_image(captcha_url, f"./wordClick/",
                                     f'{captcha_sn_md5}_word_pic.jpeg')
                pass
        except KeyError as ke:
            error = 'KeyError-->' + str(ke)
            traceback.print_exc()
            pass

    return jsonify({
        'status': -1,
        'errorMsg': error,
        'distance': 0,
        'type': 'unknown',
    }), 200


def write_to_file(filepath, content):
    # 获取目录路径
    directory = os.path.dirname(filepath)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 写入文件
    with open(filepath, 'a') as file:
        file.write(content)


@app.route('/seave_word', methods=['POST'])
def seave_word():
    if request.method == 'POST':
        try:
            data = request.json
        except Exception as e:
            data = None
        if data:
            seave_word_html = data.get('captcha_url')
        else:
            # 如果POST数据不是JSON格式，尝试获取表单数据
            seave_word_html = request.form.get('captcha_url')
        print('seave_word_html: ', seave_word_html)

        md5 = hashlib.md5()
        # 更新哈希对象 with the bytes of the input string
        md5.update(seave_word_html.encode('utf-8'))
        write_to_file(f'./word_content/{md5.hexdigest()}.txt', seave_word_html)

        return jsonify({
            'status': 0,
            'errorMsg': 'ok',
        }), 200

    return jsonify({
        'status': -1,
        'errorMsg': 'error',
    }), 200


if __name__ == '__main__':
    #  adb reverse tcp:5005 tcp:5005
    app.run(debug=True, host='0.0.0.0', port=5005)
    # cut_distance_gpt(
    #     '/Users/liuqiang/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/sliding/acb1103288cc691c6dd5f71dcb4d4406_slid_bg.jpeg',
    #     '/Users/liuqiang/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/sliding/acb1103288cc691c6dd5f71dcb4d4406_slid_cut.jpeg',
    # )
    # get_cut_point_in_bg(
    #     '/Users/liuqiang/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/sliding/acb1103288cc691c6dd5f71dcb4d4406_slid_bg.jpeg',
    #     '/Users/liuqiang/WorkSpace/Python/fridamodels-scripts/kws/v10_7_20/captcha/sliding/acb1103288cc691c6dd5f71dcb4d4406_slid_cut.jpeg',
    # )
