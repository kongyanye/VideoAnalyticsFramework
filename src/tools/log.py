import logging
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent.parent / 'log/log.txt'
LOG_LEVEL = 'DEBUG'


def _reset_logger(log):
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    for handler in log.handlers:
        handler.close()
        log.removeHandler(handler)
        del handler
    log.handlers.clear()
    log.propagate = False
    console_handle = logging.StreamHandler(sys.stdout)
    console_handle.setFormatter(
        logging.Formatter(
            "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    file_handle = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handle.setFormatter(
        logging.Formatter(
            "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    log.addHandler(file_handle)
    log.addHandler(console_handle)


def _get_logger():
    log = logging.getLogger("log")
    _reset_logger(log)
    if LOG_LEVEL == 'INFO':
        log.setLevel(logging.INFO)
    elif LOG_LEVEL == 'DEBUG':
        log.setLevel(logging.DEBUG)
    elif LOG_LEVEL == 'ERROR':
        log.setLevel(logging.ERROR)
    else:
        log.error(f'unsupported LOG_LEVEL: {LOG_LEVEL}')
    return log


# 日志句柄
logger = _get_logger()
logger.info(f'log path: {LOG_PATH}')
