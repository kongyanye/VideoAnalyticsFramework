import time
import yaml

import cv2
import numpy as np


def retry(func):
    # retry decorator
    def func_proxy(*args, **kwargs):
        count = 0
        while True:
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if e.__class__.__name__ != 'ConnectionError':
                    raise e
                count += 1
                time.sleep(3)

    return func_proxy


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def wrapc(text, c='r'):
    """Print colored texts."""
    colors = {
        'r': '\033[1;031m',  # red
        'g': '\033[1;032m',  # green
        'o': '\033[1;033m',  # orange
        'p': '\033[1;035m',  # purple
        'w': '\033[1;037m',  # white
        'y': '\033[1;093m'  # yellow
    }

    start = colors.get(c, 'w')
    end = '\033[0m'
    return f'{start}{text}{end}'


def add_boxes_to_img(img,
                     tp_box=[],
                     fp_box=[],
                     fn_box=[],
                     normed_box=True,
                     idx=None):
    """Add bounding boxes detections to img"""
    if isinstance(tp_box, np.ndarray):
        tp_box = tp_box.tolist()
    if isinstance(fp_box, np.ndarray):
        fp_box = fp_box.tolist()
    if isinstance(fn_box, np.ndarray):
        fn_box = fn_box.tolist()

    # stack gray image to be three channels
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=-1)

    # specify color to draw
    colors = {'r': (0, 0, 255), 'g': (0, 255, 0), 'y': (0, 255, 255)}
    height, width = img.shape[:2]
    color_list = ['g'] * len(tp_box) + ['y'] * len(fp_box) + ['r'
                                                              ] * len(fn_box)
    box_list = tp_box + fp_box + fn_box

    for color_name, box in zip(color_list, box_list):
        color = colors[color_name]
        x1, y1, x2, y2, conf, label = box
        if normed_box:
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(
                x2 * width), int(y2 * height)

        if label == 0:
            label = 'person'
        else:
            label = 'vehicle'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _ = cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    if idx is not None:
        _ = cv2.putText(img, str(idx), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 2)
    return img
