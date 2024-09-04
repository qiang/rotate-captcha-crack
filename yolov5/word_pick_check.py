#!/user/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import cv2
import ddddocr
import torch
from PIL import Image
from PIL import ImageDraw, ImageFont


def draw_on_image(image_path, word_list, word_boxes, saved_name, is_show=False):
    """
    在图片上绘制圆圈并标注文字，但不修改原图。

    :param image_path: 输入图片的路径
    :param word_list: 要点击的文字列表
    :param word_boxes: 上面计算出来的文字中心点
    :return: None
    """

    print("draw_on_image==>", word_list, word_boxes, saved_name)

    # 每项为 (中心点坐标, 半径, 标注文字)
    circles_with_labels = [
        # ((100, 100), 50, '1'),
        # ((200, 200), 50, '2'),
        # ((300, 300), 50, '3'),
        # ((400, 400), 50, '4')
    ]
    count = 1
    for w in word_list:
        circles_with_labels.append((word_boxes[w], 50, str(count)))
        count += 1

    try:
        # 打开原始图片
        original_img = Image.open(image_path)

        # 创建图片副本进行绘制
        img_copy = original_img.copy()

        # 创建绘图对象
        draw = ImageDraw.Draw(img_copy)

        # 加载字体（需要指定字体文件路径，这里使用一个常见的路径）
        # 把问题绘制到图片上，便于查看
        try:
            # 矩形的参数
            rect_width, rect_height = 180, 50  # 矩形的宽度和高度
            rect_x, rect_y = 0, 0  # 矩形的左上角坐标

            # 绘制黑色矩形
            draw.rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height], fill='black')

            # 设置要绘制的中文汉字
            text = ",".join(word_list)
            # 确保你有一个合适的中文字体文件，例如 SimHei.ttf
            try:
                import platform
                # 获取系统类型
                system = platform.system()
                if system == "Linux":
                    print("当前操作系统是 Linux")
                    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                              20)  # 确保路径和字体文件存在
                else:
                    font = ImageFont.truetype("/Library/Fonts/PingFang.ttc", 20)  # 确保路径和字体文件存在
                    print("当前操作系统是 macOS")
            except IOError:
                print("出现异常，使用默认字体。。。。")
                font = ImageFont.load_default()
            # 计算文本的边界框
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 计算文本的位置（居中于矩形内）
            text_x = rect_x + (rect_width - text_width) / 2
            text_y = rect_y + (rect_height - text_height) / 2

            # 绘制文本
            draw.text((text_x, text_y), text, font=font, fill='white')
        except Exception as e:
            print("绘制文字异常", e)

        # 加载字体（可以根据需要调整字体和大小）
        # font = ImageFont.load_default()
        # 加载自定义字体（放大字体）
        # 加载系统默认字体
        try:
            font_size = 60
            import platform
            # 获取系统类型
            system = platform.system()
            if system == "Linux":
                print("当前操作系统是  Linux 绘制数字")
                font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                          60)  # 确保路径和字体文件存在
            else:
                font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)  # 确保路径和字体文件存在
                print("当前操作系统是 macOS 绘制数字")
        except IOError:
            # 如果指定字体文件无法找到，则使用默认字体
            font = ImageFont.load_default()

        # 在副本上绘制圆圈和添加文字
        for (center, radius, label) in circles_with_labels:
            # 绘制圆圈
            left_up = (center[0] - radius, center[1] - radius)
            right_down = (center[0] + radius, center[1] + radius)
            draw.ellipse([left_up, right_down], outline='red', width=2)

            # 计算文本尺寸
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            # 添加文字
            # text_position = (center[0] - radius, center[1] - radius - 10)
            # 计算文本位置，使其居中于圆圈内
            text_x = center[0] - text_width / 2
            text_y = center[1] - text_height / 2
            text_position = (text_x, text_y)

            draw.text(text_position, label, fill='red', font=font)

        if is_show:
            # 显示修改后的图像
            img_copy.show()

        directory = os.path.dirname(saved_name)
        # 如果目录不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存复制后的图片
        img_copy.save(saved_name)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(e)


def calculate_center(left_top, right_bottom):
    """
    计算矩形的中心点坐标。

    :param left_top: 小矩形的左上角坐标 (x1, y1)
    :param right_bottom: 小矩形的右下角坐标 (x2, y2)
    :return: 矩形中心点坐标 (cx, cy)
    """
    x1, y1 = left_top
    x2, y2 = right_bottom

    # 计算中心点坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy


def match_words(crop_img_path, left_top, right_bottom, word_list):
    try:
        # 打开图片
        img = Image.open(crop_img_path)

        for angle in range(0, 360, 20):
            # 旋转图片
            rotated_img = img.rotate(angle, expand=True)
            ocr = ddddocr.DdddOcr(show_ad=False)
            result = ocr.classification(rotated_img)
            print(result)
            if result in word_list:
                return {result: calculate_center(left_top, right_bottom)}

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def crop_rectangle(image_path, left_top, right_bottom, output_path):
    """
    从图片中裁剪出一个矩形区域并保存结果。如果输出目录不存在，则创建目录。

    :param image_path: 输入图片的路径
    :param left_top: 小矩形的左上角坐标 (x1, y1)
    :param right_bottom: 小矩形的右下角坐标 (x2, y2)
    :param output_path: 裁剪后图片的保存路径
    """
    try:
        # 打开图片
        img = Image.open(image_path)

        # 确保坐标有效
        if (left_top[0] < 0 or left_top[1] < 0 or
                right_bottom[0] > img.width or right_bottom[1] > img.height):
            raise ValueError("Coordinates are out of the image boundaries.")

        # 定义裁剪区域 (left, upper, right, lower)
        crop_box = (left_top[0], left_top[1], right_bottom[0], right_bottom[1])

        # 裁剪图片
        cropped_img = img.crop(crop_box)

        # 确保输出目录存在，如果不存在则创建
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存裁剪后的图片
        cropped_img.save(output_path)

        # 可选：显示裁剪后的图片
        # cropped_img.show()
        return output_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def do_check(img_path, word_list, model_path=None):
    if model_path:
        model = torch.hub.load('./yolov5', 'custom', path=model_path,
                               source='local')
    else:
        model = torch.hub.load('.', 'custom', path='runs/train/exp3/weights/best.pt',
                               source='local')
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp3/weights/best.onnx')

    # 读取图片
    # img_path = '/Users/liuqiang/Downloads/yolov5/dataset/images/val/1a853b93359b27a8228bded9f2231fb3_word_pic.jpeg'
    img = cv2.imread(img_path)
    # 调整图像大小以匹配模型的输入尺寸
    # img = cv2.resize(img,(640, 640))  # 假设模型期望的输入大小是 416x416

    # 获取图像大小
    height, width, channels = img.shape

    # 打印图像大小
    print(f'Image Width: {width}px')
    print(f'Image Height: {height}px')
    print(f'Number of Channels: {channels}')

    # 将图像转换为 RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 进行预测
    results = model(img, )

    # 打印检测结果
    # results.print()

    # 获取检测结果的边界框、类别和置信度
    # results.xyxy[0] 是一个 Tensor，其中包含检测结果
    # 结果的每一行包含 [x1, y1, x2, y2, confidence, class]
    boxes = results.xyxy[0].cpu().numpy()  # 转换为 NumPy 数组

    word_boxes = {}
    count = 0
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        print(f"Class: {int(cls)}, Confidence: {conf:.2f}")
        print(f"Bounding box: ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f})")
        crop_img_path = crop_rectangle(img_path, (x1, y1), (x2, y2),
                                       f'./crop_img_cache/{Path(img_path).stem}_{count}.png')
        word_coordinate = match_words(crop_img_path, (x1, y1), (x2, y2), word_list)
        if word_coordinate:
            print("word_coordinate===>", word_coordinate)
            word_boxes.update(word_coordinate)
        else:
            print('word_coordinate 获取error', word_coordinate)
        count += 1

    # 绘制检测结果并显示
    # results.show()

    # 保存检测结果
    # results.save()
    print(json.dumps(word_boxes, ensure_ascii=False, ))

    try:
        draw_on_image(img_path, word_list, word_boxes, f'./recognized_img_cache/{Path(img_path).stem}.png')
    except BaseException as e:
        error_path = f'./recognized_img_cache_error/{Path(img_path).stem}.png'
        directory = os.path.dirname(error_path)
        # 如果目录不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        original_img = Image.open(img_path)
        # 创建图片副本进行绘制
        img_copy = original_img.copy()
        img_copy.save(error_path)

    return word_boxes


if __name__ == '__main__':
    do_check('dataset/images/val/1a853b93359b27a8228bded9f2231fb3_word_pic.jpeg',
             ['潭', '稚', '炼', '感'])

    # r = do_check('/Users/liuqiang/Downloads/8a34139f8becbabcb9a1ce8adcd654ba_猴斡愤党_word_pic.jpeg',
    #              list('猴斡愤党'))
    # print(r)

    # r = do_check('/Users/liuqiang/Downloads/9c4a02300733914c6c272a65bbab3096_暖咸疆新_word_pic.jpeg',
    #              list('暖咸疆新'))
    # print(r)

    # do_check('/Users/liuqiang/Downloads/wordClick/530906b12eca6438b7c1eab3f3531cc2_word_pic.jpeg',
    #          ['堵', '崎', '室', '沾'])

    pass
