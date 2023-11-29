import pickle
import socket
import struct
import time

from tools.log import logger


def create_socket(host, port, listen=1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(listen)
    logger.info(f'Listening on {host}:{port}')
    return s


def connect_socket(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    notified = False
    while True:
        try:
            s.connect((host, port))
            logger.info(f'Connected to {host}:{port}')
            break
        except Exception as e:
            if e.__class__.__name__ not in ['ConnectionRefusedError', 'OSError']:
                # OSError for macOS
                logger.error(e)
            if not notified:
                logger.warning(f'Fail to connect to {host}:{port}, retry...')
                notified = True
            time.sleep(2)
    return s


def send_data(s, data):
    serialized_data = pickle.dumps(data)

    message = struct.pack('>Q', len(serialized_data)) + serialized_data
    sent_size = len(message) / 1024  # in KB
    success = s.sendall(message)
    if success is not None:
        raise Exception('Fail to send data')
    return sent_size


def recv_data(s):
    # Receive the length of the incoming message
    size = s.recv(8)
    if not size:
        raise Exception('Fail to receive data')

    # Decode the message size
    message_size = struct.unpack('>Q', size)[0]

    # Receive the actual message
    data = b''
    bufsize = 4096
    while len(data) < message_size:
        remaining_length = message_size - len(data)
        if remaining_length > bufsize:
            chunk = s.recv(bufsize)
        else:
            chunk = s.recv(remaining_length)
        data += chunk

    data = pickle.loads(data)
    return data
