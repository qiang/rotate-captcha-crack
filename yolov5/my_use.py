#!/user/bin/env python
# -*- coding: utf-8 -*-
import cv2
import torch
from ultralytics import YOLO


def do_check():
    model = torch.hub.load('.', 'custom', path='/Users/liuqiang/Downloads/yolov5/runs/train/exp3/weights/last.pt',
                           source='local')

    # 读取图片
    img_path = '/Users/liuqiang/Downloads/yolov5/dataset/images/val/1a853b93359b27a8228bded9f2231fb3_word_pic.jpeg'
    img = cv2.imread(img_path)

    # 将图像转换为 RGB 格式
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 进行预测
    results = model(img)

    # 获取检测结果的边界框、类别和置信度
    # results.xyxy[0] 是一个 Tensor，其中包含检测结果
    # 结果的每一行包含 [x1, y1, x2, y2, confidence, class]
    boxes = results.xyxy[0].cpu().numpy()  # 转换为 NumPy 数组

    word_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        print(f"Class: {int(cls)}, Confidence: {conf:.2f}")
        print(f"Bounding box: ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f})")
        word_boxes.append([(x1, y1), (x2, y2)])

    # 绘制检测结果并显示
    results.show()

    # 保存检测结果
    # results.save()
    return word_boxes


def use_yolo():
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # 加载模型
    model = YOLO('/Users/liuqiang/Downloads/yolov5/runs/train/exp3/weights/last.pt')

    # 对单张图像进行推理
    results = model('/Users/liuqiang/Downloads/yolov5/dataset/images/val/1a853b93359b27a8228bded9f2231fb3_word_pic.jpeg')

    # 获取检测结果
    boxes = results.xyxy[0].numpy()  # [x1, y1, x2, y2, confidence, class]
    labels = results.names  # 类别标签

    # 可视化结果
    def visualize_results(image_path, boxes, labels):
        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax = plt.gca()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制边界框
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            plt.text(x1, y1, f'{labels[int(cls)]} {conf:.2f}', color='red', fontsize=12)

        plt.axis('off')
        plt.show()

    # 调用可视化函数
    visualize_results('path/to/your/image.jpg', boxes, labels)


if __name__ == '__main__':
    # do_check()
    use_yolo()
    # model = YOLO('/Users/liuqiang/Downloads/yolov5/runs/train/exp2/weights/best.pt',task='detect')
    # model.export(format='onnx',imgsz=640, )
