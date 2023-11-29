import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from tools.utils import wrapc


class VideoSource(ABC):
    """Base class for reading video sources."""

    def __init__(self,
                 resize=None,
                 fps=30,
                 read_start=0,
                 max_num=None,
                 batch_size=1):
        """
        Args:
            resize: resize the image to a specific size, can be a tuple of
                (height, width), a float ratio in (0, 1), or an int max(height, width)
            fps: frames per second
            read_start: the batch number to start reading
            max_num: maximum number of frames to read, in unit of batch_size
            batch_size: number of frames to read at a time
        """

        self.resize = resize
        self.fps = fps
        self.read_start = read_start
        self.max_num = max_num
        self.batch_size = batch_size

        # initialize variables
        self.height, self.width = None, None
        self.last_send_time = time.time()

        # sanity check
        self.check_read_num()
        self.read_end = self.read_start + self.max_num

    def wait_for_next_frame(self):
        """Wait for the next frame to be read to meet the fps requirement."""
        if self.fps is not None:
            diff = self.last_send_time + 1 / self.fps * self.batch_size - time.time(
            )
            if diff > 0:
                time.sleep(diff)
        self.last_send_time = time.time()

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_name(self):
        """Get the dataset name. Default is (null)."""
        return self.name

    def __len__(self):
        """Return the number of batches to read."""
        return self.max_num

    def reset(self):
        """Reset the video source loader. Useful for video dataset to reset cap."""
        pass

    def check_read_num(self):
        total_frame_num = self.get_total_frame_num()
        # check read_start and max_num
        assert self.read_start >= 0, f'read_start must be non-negative, got {self.read_start}'
        if self.max_num is None:
            self.max_num = total_frame_num // self.batch_size - self.read_start
            assert self.max_num > 0, (
                f'got only {total_frame_num//self.batch_size} batches for video dataset {self.name}, '
                f'while read_start is {self.read_start}, please use smaller read_start or batch_size'
            )

        else:
            if (self.read_start +
                    self.max_num) * self.batch_size > total_frame_num:
                new_max_num = total_frame_num // self.batch_size - self.read_start
                assert new_max_num > 0, (
                    f'got only {total_frame_num//self.batch_size} batches for video dataset {self.name}, '
                    f'while read_start is {self.read_start}, please use smaller read_start or batch_size'
                )

                print(
                    wrapc(
                        f'warning: got only {total_frame_num//self.batch_size} batches for video dataset {self.name}, '
                        f'read_start={self.read_start}, max_num={self.max_num}, reset max_num to {new_max_num}',
                        'y'))
                self.max_num = new_max_num

    def read(self, **kwargs):
        """Return an iterator of BGR images."""
        if self.batch_size == 1:
            return self.read_one(**kwargs)
        else:
            return self.read_batch(**kwargs)

    @abstractmethod
    def get_total_frame_num(self):
        """Return the total number of frames in the video."""
        raise NotImplementedError

    @abstractmethod
    def read_one(self):
        """Return an iterator of one BGR images."""
        raise NotImplementedError

    @abstractmethod
    def read_batch(self):
        """Return an iterator of batch of BGR images."""
        raise NotImplementedError

    def resize_img(self, img, resize):
        """Resize image to target size"""
        if self.height is None or self.width is None:
            self.height, self.width = img.shape[:2]
            if resize is not None:
                # resize to a specifc width and height
                if isinstance(resize, tuple) or isinstance(resize, list):
                    assert isinstance(resize[0], int) and isinstance(
                        resize[1], int), 'size in tuple/list must be int'
                    self.height = resize[0]
                    self.width = resize[1]

                # resize a specific ratio
                elif resize == 1 or isinstance(resize, float):
                    assert resize > 0 and resize <= 1, (
                        'resize should be in '
                        f'(0, 1) when setting to a float, got {resize}')

                    self.width = int(self.width * resize)
                    self.height = int(self.height * resize)

                # resize to a specific max(height, width)
                elif isinstance(resize, int):
                    assert resize % 2 == 0, (
                        'when setting resize to a specific width, '
                        f'it should be an even value, got {resize}')

                    ratio = resize / max(self.width, self.height)
                    self.width = int(self.width * ratio)
                    self.height = int(self.height * ratio)

        img = cv2.resize(img, (self.width, self.height))
        return img


class FromImgDir(VideoSource):
    """Read images from a directory of image files."""

    def __init__(self, src_path, filetype='jpg', **kwargs):
        """
        Args:
            src_path: the directory to read image files
            filetype: the image suffix
        """

        assert Path(src_path).is_dir(), (f'source path {src_path} is'
                                         'not a valid image directory')

        src_dir = Path(src_path)
        self.flist = sorted(
            src_dir.glob(f'*.{filetype}'))  # list all the images
        assert len(
            self.flist) > 0, f'no {filetype} images found under {src_dir}'

        self.name = Path(src_path).name
        super().__init__(**kwargs)

    def get_total_frame_num(self):
        return len(self.flist)

    def read_one(self):

        for idx, each in enumerate(
                self.flist[self.read_start * self.batch_size:self.read_end *
                           self.batch_size]):
            img = cv2.imread(str(each))
            if self.resize is not None:
                img = self.resize_img(img, self.resize)
            else:
                if self.height is None or self.width is None:
                    self.height, self.width = img.shape[:2]

            self.wait_for_next_frame()
            yield self.read_start * self.batch_size + idx, img

    def read_batch(self):
        for idx, s in enumerate(
                range(self.read_start * self.batch_size,
                      self.read_end * self.batch_size, self.batch_size)):
            if s + self.batch_size > len(self.flist):
                break
            e = min(s + self.batch_size, len(self.flist))

            imgs = []
            for each in self.flist[s:e]:
                img = cv2.imread(str(each))
                if self.resize is not None:
                    img = self.resize_img(img, self.resize)
                else:
                    if self.height is None or self.width is None:
                        self.height, self.width = img.shape[:2]
                imgs.append(img)

            self.wait_for_next_frame()
            yield self.read_start + idx, np.array(imgs)


class FromRawVideo(VideoSource):
    """Read images directly from a video file."""

    def __init__(self, src_path, **kwargs):
        """
        Args:
            src_path: the file path to read video
        """
        assert Path(src_path).is_file(), f'src_file {src_path} is not a video'
        self.cap = cv2.VideoCapture(src_path)

        self.name = Path(src_path).name
        super().__init__(**kwargs)

    def get_total_frame_num(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read_one(self):
        # skip the beginning `read_start` frames
        if self.read_start != 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.read_start)
        idx = 0

        while True:
            ret, img = self.cap.read()
            if not ret:
                break
            if self.resize is not None:
                img = self.resize_img(img, self.resize)
            else:
                if self.height is None or self.width is None:
                    self.height, self.width = img.shape[:2]

            self.wait_for_next_frame()
            yield self.read_start + idx, img

            # if already read `self.batch_num` images
            idx += 1
            if idx == self.max_num:
                break

    def read_batch(self, read_start=0):
        # skip the first the `read_start` batch
        if read_start != 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,
                         self.read_start * self.batch_size)
        idx = 0

        while True:
            imgs = []
            # read a batch of images
            for _ in range(self.batch_size):
                ret, img = self.cap.read()
                if not ret:
                    if len(imgs) == self.batch_size:
                        self.wait_for_next_frame()
                        return self.read_start + idx, np.array(imgs)
                    else:
                        return None
                if self.resize is not None:
                    img = self.resize_img(img, self.resize)
                else:
                    if self.height is None or self.width is None:
                        self.height, self.width = img.shape[:2]
                imgs.append(img)

            self.wait_for_next_frame()
            yield self.read_start + idx, np.array(imgs)

            # if already read `num` batches
            idx += 1
            if idx == self.max_num:
                break


def get_video_gen(src_path, **kwargs):
    if os.path.isdir(src_path):
        return FromImgDir(src_path=src_path, **kwargs)
    elif os.path.isfile(src_path):
        return FromRawVideo(src_path=src_path, **kwargs)
    else:
        raise ValueError(f'Unknown video path: {src_path}')
