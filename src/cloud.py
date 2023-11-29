import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import tqdm
from ultralytics import YOLO

from tools.eval import Evaluation
from tools.image_codecs import decode_img
from tools.log import logger
from tools.socket import connect_socket, create_socket, recv_data, send_data
from tools.utils import add_boxes_to_img, load_config


class Cloud:

    def __init__(self, config_path):
        self.config = load_config(config_path)
        logger.info(f'CONFIGURATION - {self.config}')
        self.show_image = self.config['cloud']['show_image']

        # check HOME_DIR existence
        if not os.path.exists(self.config['HOME_DIR']):
            raise Exception(f'HOME_DIR in config.yaml {self.config["HOME_DIR"]} does not exist. Do you forget to modify it?')
        
        # initialize variables
        self.last_idx = None
        self.queue_to_decode = queue.Queue()
        self.queue_to_infer = queue.Queue()
        self.queue_to_show = queue.Queue()

        # initialize detect models
        self.model = YOLO(
            os.path.join(self.config['HOME_DIR'],
                         self.config['cloud']['infer_ckpt_path']))
        _ = self.model.predict(np.zeros((640, 640, 3)),
                               verbose=False)  # warm up

    def __enter__(self):
        # used to receive data from edge
        self.cloud_socket = create_socket(self.config['cloud']['host'],
                                          self.config['cloud']['port'])

        return self

    def receive_frames(self):

        # Sync configuration with edge
        data = recv_data(self.edge_conn)
        self.edge_config = data['edge_config']
        self.total_num = data['total_num']  # actual number of frames
        self.video_name = data['video_name']
        self.last_idx = self.edge_config['video_config'][
            'read_start'] + self.total_num - 1
        logger.info('configuration synced with edge')

        # load ground truth results for evaluation
        self.evaluation = Evaluation(
            src_path=self.edge_config['video_config']['src_path'],
            gt_dir=os.path.join(self.config['HOME_DIR'], self.config['cloud']['gt_dir']),
            gt_ckpt_path=self.config['cloud']['gt_dir'],
            read_start=self.edge_config['video_config']['read_start'],
            total_num=self.total_num)

        # notify edge to be ready
        logger.info('cloud is ready')
        send_data(self.edge_socket, 'ready')

        # progress bar
        self.iterator = tqdm.tqdm(range(self.total_num),
                                  total=self.total_num,
                                  leave=True)

        # receive frames from edge
        for _ in self.iterator:
            data = recv_data(self.edge_conn)  # {'idx': idx, 'frame': frame}
            # idx = data['idx']
            # frame = data['frame']
            self.queue_to_decode.put(data)

    def decode_frames(self):
        # decode frames
        for data in iter(self.queue_to_decode.get, None):
            idx = data['idx']
            frame_enc = data['frame']
            height = data['height']
            width = data['width']
            codecs = self.edge_config['codecs']

            # decode image
            frame_dec = decode_img(frame_enc, codecs, height, width)
            self.iterator.set_description(
                f'received frame with idx {idx}, shape: {frame_dec.shape}')

            # send to infer
            data = {'idx': idx, 'frame': frame_dec}
            self.queue_to_infer.put(data)

            # check if finished
            if idx == self.last_idx:
                break

    def infer_frames(self):
        # infer frames
        for data in iter(self.queue_to_infer.get, None):
            idx = data['idx']
            frame = data['frame']

            # infer
            t1 = time.time()
            preds = self.model.predict(frame, verbose=False)
            boxes = np.array(preds[0].boxes.xyxyn.cpu())
            conf = np.array(preds[0].boxes.conf.cpu())
            classes = np.array(preds[0].boxes.cls.cpu())
            preds = np.concatenate((boxes, conf[:, None], classes[:, None]),
                                   axis=1)
            t2 = time.time()
            lat_infer = (t2 - t1) * 1000  # in ms

            # evaluate
            (map50, tp_box, fp_box,
             fn_box), mean_obj_size, obj_num = self.evaluation.eval(
                 idx, preds)

            # send results back to the edge
            data = {
                'idx': idx,
                'map50': map50,
                'lat_infer': lat_infer,
                'mean_obj_size': mean_obj_size,
                'obj_num': obj_num,
            }
            send_data(self.edge_socket, data)

            # cache to show
            data = {
                'idx': idx,
                'frame': frame,
                'tp_box': tp_box,
                'fp_box': fp_box,
                'fn_box': fn_box
            }
            self.queue_to_show.put(data)

            # check if finished
            if idx == self.last_idx:
                break

    def show(self):
        # show images and detection results
        if not self.show_image:
            return
        cv2.namedWindow('cloud', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('cloud', cv2.WND_PROP_TOPMOST, 1)

        for data in iter(self.queue_to_show.get, None):

            frame = add_boxes_to_img(img=data['frame'],
                                     tp_box=data['tp_box'],
                                     fp_box=data['fp_box'],
                                     fn_box=data['fn_box'],
                                     idx=data['idx'])
            cv2.imshow('cloud', frame)
            cv2.waitKey(1)

            if data['idx'] == self.last_idx:
                break

    def run(self):
        # used to send data to edge
        self.edge_socket = connect_socket(self.config['edge']['host'],
                                          self.config['edge']['port'])

        self.edge_conn, self.edge_addr = self.cloud_socket.accept()

        tasks = []
        tasks.append(threading.Thread(target=self.receive_frames, args=()))
        tasks.append(threading.Thread(target=self.decode_frames, args=()))
        tasks.append(threading.Thread(target=self.infer_frames, args=()))
        # tasks.append(threading.Thread(target=self.show, args=()))

        for task in tasks:
            task.start()

        # macOS requires a window to be shown in the main thread
        self.show()

        for task in tasks:
            task.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.cloud_socket.close()
        self.edge_socket.close()
        self.iterator.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        logger.info(f'using custom config file {sys.argv[1]}')
        if not os.path.exists(sys.argv[1]):
            raise Exception(f'config file {sys.argv[1]} does not exist')

        config_path = sys.argv[1]
    else:
        config_path = 'config.yaml'

    with Cloud(config_path=config_path) as cloud:
        cloud.run()
