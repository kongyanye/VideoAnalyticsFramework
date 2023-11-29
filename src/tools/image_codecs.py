import cv2
import numpy as np
from PIL import Image


def encode_img(img, codecs, qfactor=95):
    """Encode the image"""
    if codecs is None or codecs == 'none':
        return img
    elif codecs == 'raw':
        img = Image.fromarray(img).tobytes()
    elif codecs == 'jpg':
        _, img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, qfactor])
    else:
        success, img = cv2.imencode(f'.{codecs}', img)
        assert success, f'Fail to encode image with codecs {codecs}'
    return img


def decode_img(img, codecs, height=720, width=1080):
    if codecs in [None, 'none']:
        return img
    elif codecs in ['raw']:
        img = Image.frombytes('RGB', (width, height), img)
        img = np.array(img)
    else:
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    return img
