import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 20000000000


class ImgReader:
    """used for jpg, png, etc."""

    default_dims = [1.0, 2.0, 4.0, 8.0, 16.0]

    def __init__(self, filename) -> None:
        self.filename = filename
        img = cv2.imread(filename)[:, :, ::-1]  # To RGB
        h, w, _ = img.shape
        # openslide (width, height)
        self.img = img
        self._shape = [w, h]


    def read_region(self, location, level, size):
        # convert coors, the coors always on level 0
        x, y = location
        w, h = size
        _w = int(w * self.level_downsamples[level])
        _h = int(h * self.level_downsamples[level])

        img = self.img[y : y + _h, x : x + _w].copy()
        img = Image.fromarray(img).resize((w, h))
        return img

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    @property
    def level_count(self):
        return len(self.default_dims)

    @property
    def level_downsamples(self):
        shape = [self._shape[0] / r[0] for r in self.level_dimensions]
        return shape

    @property
    def level_dimensions(self):
        shape = [(int(self._shape[0] / r), int(self._shape[1] / r)) for r in self.default_dims]
        return shape

    def get_best_level_for_downsample(self, scale):
        preset = [i * i for i in self.level_downsamples]
        err = [abs(i - scale) for i in preset]
        level = err.index(min(err))
        return level

    def close(self):
        pass


