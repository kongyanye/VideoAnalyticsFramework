import numpy as np
import pandas as pd

from tools.log import logger
from tools.utils import wrapc


class PerfMeter:

    def __init__(self, log_path):
        self.data = {}
        self.allowed_keys = ['size', 'map50', 'lat_infer']
        self.log_path = log_path

    def add(self, key, value, unit):
        assert key in self.allowed_keys, f'`key` {key} not allowed'
        if key not in self.data:
            self.data[key] = {'unit': unit, 'value': []}
        else:
            assert unit == self.data[key]['unit'], (
                f'different unit found for `key` {key}, '
                f'old: {self.data[key]["unit"]}, new: {unit}')
        self.data[key]['value'].append(value)

    def get(self, key, idx):
        assert key in self.data
        return self.data[key]['value'][idx]

    def summary(self):
        line = ''
        for key in self.data.keys():
            unit = self.data[key]['unit']

            if key == 'size':
                num = len(self.data[key]['value'])
                total = sum(self.data[key]['value'])
                mean = total / num
                line += f'{num} frames processed, {key}: {mean:.2f} {unit} (MEAN)'
            elif key == 'map50':
                mean = np.mean(self.data[key]['value'])
                line += f', mAP50: {mean:.2f} (MEAN)'
            elif key.startswith('lat_') or key.startswith('t_'):
                mean = np.mean(self.data[key]['value'])
                line += f', {key}: {mean:.2f} ms (MEAN)'
            elif key.startswith('obj_'):
                mean = np.mean(self.data[key]['value'])
                line += f', {key}: {mean:.2f} (MEAN)'

        logger.info(wrapc(f'SUMMARY - {line}', 'g'))

        # saving to dataframe
        df = {}
        for key in self.data.keys():
            df[key] = self.data[key]['value']
        df = pd.DataFrame(df)
        df.to_csv(self.log_path, index=True)
