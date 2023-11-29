import os
import time
from pathlib import Path

config_template = """
edge:
  host: '127.0.0.1'
  port: 6666
  video_config:
    src_path: {src_path}
    resize: [360, 540]
    fps: 30
    read_start: 0
    max_num: 100
    batch_size: 1
  codecs: 'jpg'
  qfactor: {qfactor} # only valid for jpg
  log_dir: {log_dir}
  log_name: {log_name}

cloud:
  host: '127.0.0.1'
  port: 7777
  gt_dir: '/home/sig/files/VideoAnalyticsFramework/ground_truth/'
  show_image: False
"""

current_dir = str(Path('./').resolve())
save_config_path_full = f'{current_dir}/tmp.yaml'

for src_path in ['../../dataset/road.mp4', '../../dataset/uav']:
    for qfactor in [10, 30, 50, 70, 90]:
        video_name = Path(src_path).name
        log_dir = f'{current_dir}/{video_name}'
        log_name = f'{qfactor}.csv'
        config = config_template.format(src_path=src_path,
                                        qfactor=qfactor,
                                        log_dir=log_dir,
                                        log_name=log_name)

        if os.path.exists(f'{log_dir}/{log_name}'):
            print(f'{log_dir}/{log_name} already exists, skipping...')
            continue

        with open(save_config_path_full, 'w') as f:
            f.write(config)

        os.system(f'python /home/sig/files/VideoAnalyticsFramework/src/cloud.py {save_config_path_full} &')
        os.system(f'python /home/sig/files/VideoAnalyticsFramework/src/edge.py {save_config_path_full}')

        time.sleep(3)
