import os
import sys
import time
from datetime import datetime
from threading import Thread

import tqdm

from tools.image_codecs import encode_img
from tools.log import logger
from tools.perf import PerfMeter
from tools.socket import connect_socket, create_socket, recv_data, send_data
from tools.utils import load_config
from tools.video_source import get_video_gen


class Edge:

    def __init__(self, config_path):
        self.config = load_config(config_path)
        logger.info(f'CONFIGURATION - {self.config}')

        # video loader
        self.video_gen = get_video_gen(
            src_path=os.path.join(
                self.config['HOME_DIR'],
                self.config['edge']['video_config']['src_path']),
            resize=self.config['edge']['video_config']['resize'],
            fps=self.config['edge']['video_config']['fps'],
            read_start=self.config['edge']['video_config']['read_start'],
            max_num=self.config['edge']['video_config']['max_num'],
            batch_size=self.config['edge']['video_config']['batch_size'])
        self.video_name = self.video_gen.get_name()
        self.total_num = len(self.video_gen)  # acutal number of frames

        # initialize variables
        self.ready = False  # whether the cloud is ready

        # performance log
        if self.config['edge']['log_dir'] is None:
            log_dir = '../log'
        else:
            log_dir = self.config['edge']['log_dir']
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if self.config['edge']['log_name'] is None:
            s = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = f'{self.video_name}_{s}.csv'
        else:
            log_name = self.config['edge']['log_name']
        self.perf = PerfMeter(log_path=f'{log_dir}/{log_name}')

    def send_frames(self):
        # send edge configuration to the cloud
        data = {
            'edge_config': self.config['edge'],
            'total_num': self.total_num,
            'video_name': self.video_name
        }
        send_data(self.cloud_socket, data)

        # wait for cloud to be ready
        _ = recv_data(self.cloud_conn)
        logger.info('got cloud is ready signal')
        self.ready = True

        # send frames to the cloud
        for idx, frame in self.video_gen.read():
            frame_enc = encode_img(frame, self.config['edge']['codecs'],
                                   self.config['edge']['qfactor'])
            sent_size = send_data(
                self.cloud_socket, {
                    'idx': idx,
                    'frame': frame_enc,
                    'height': frame.shape[0],
                    'width': frame.shape[1]
                })
            self.perf.add('size', sent_size, 'KB')

    def receive_preds(self):
        # wait for cloud to be ready, otherwise recv_data will block
        while not self.ready:
            time.sleep(2)

        # progress bar
        self.iterator = tqdm.tqdm(range(self.total_num),
                                  total=self.total_num,
                                  leave=True)

        # receive predictions from the cloud
        for _ in self.iterator:
            data = recv_data(self.cloud_conn)
            idx = data['idx']
            map50 = data['map50']
            lat_infer = data['lat_infer']
            self.perf.add('map50', map50, '')
            self.perf.add('lat_infer', lat_infer, 'ms')

            # update progress bar
            self.iterator.set_description(
                f'[{idx}], map50: {map50:.2f}, lat_infer: {lat_infer:.2f} ms')

        # get summary
        self.perf.summary()

    def __enter__(self):
        # used to receive data from cloud
        self.edge_socket = create_socket(self.config['edge']['host'],
                                         self.config['edge']['port'])
        logger.info('waiting for the cloud to connect...')

        self.cloud_conn, self.cloud_addr = self.edge_socket.accept()

        # used to send data to the cloud
        self.cloud_socket = connect_socket(self.config['cloud']['host'],
                                           self.config['cloud']['port'])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cloud_socket.close()
        self.edge_socket.close()
        self.iterator.close()

    def run(self):
        tasks = []
        tasks.append(Thread(target=self.send_frames, args=()))
        tasks.append(Thread(target=self.receive_preds, args=()))

        for t in tasks:
            t.start()

        for t in tasks:
            t.join()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        logger.info(f'using custom config file {sys.argv[1]}')
        if not os.path.exists(sys.argv[1]):
            raise Exception(f'config file {sys.argv[1]} does not exist')

        config_path = sys.argv[1]
    else:
        config_path = 'config.yaml'

    with Edge(config_path=config_path) as edge:
        edge.run()
