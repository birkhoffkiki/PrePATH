import math
import numpy as np

from openslide import AbstractSlide, _OpenSlideMap
from . import tmap_lowlevel


Tags = {
    'thumbail': 0,
    'navigate': 1,
    'macro': 2,
    'label': 3,
}



class TmapSlide(AbstractSlide):
    def __init__(self, filename):
        AbstractSlide.__init__(self)
        self.__filename = filename
        self._osr = tmap_lowlevel.open_tmap_file(filename)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.__filename)

    @classmethod
    def detect_format(cls, filename):
        return "un"

    @classmethod
    def get_scan_scale(self):
        return tmap_lowlevel.get_scan_scale(self._osr)

    # close the tmap file
    def close(self):
        tmap_lowlevel.close_tmap_file(self._osr)

    # get focus layer
    @property
    def get_focus_layer(self):
        return tmap_lowlevel.get_focus_layer(self._osr)

    # get tile number
    @property
    def get_tile_mumber(self):
        return tmap_lowlevel.get_tile_mumber(self._osr)

    # get tmap version
    @property
    def get_tmap_version(self):
        return tmap_lowlevel.get_tmap_version(self._osr)

    # get tmap pixel size
    @property
    def get_pixel_size(self):
        return tmap_lowlevel.get_pixel_size(self._osr)

    @property
    def dimensions(self):
        return tmap_lowlevel.get_dimensions(self._osr)

    @property
    def level_count(self):
        return tmap_lowlevel.get_level_count(self._osr)

    @property
    def level_dimensions(self):
        return tmap_lowlevel.get_level_dimensions(self._osr)

    @property
    def level_downsamples(self):
        return tmap_lowlevel.get_level_downsamples(self._osr)
    
    
    def get_best_level_for_downsample(self, downsample):
      
        preset = [i*i for i in self.level_downsamples]
        err = [abs(i-downsample) for i in preset]
        level = err.index(min(err))
        return level

    # get image meta data
    def get_image_info_ex(self, etype):
        return tmap_lowlevel.get_image_info_ex(self._osr, etype)

    # get image data
    def get_image_data(self, etype):
        return tmap_lowlevel.get_image_data(self._osr, etype)

    # get image format
    def get_image_size_ex(self, location, size, fScale=40.0):
        nLeft = location[0]
        nTop = location[1]
        nRight = nLeft + size[0]
        nBottom = nTop + size[1]
        return tmap_lowlevel.get_image_size_ex(self._osr, nLeft, nTop, nRight, nBottom, fScale)

    def associated_images(self, tag):
        if tag in Tags:
            return tmap_lowlevel.get_image_data(self._osr, Tags[tag])
        else:
            # raise Exception("Unrecgnized associated_images type [{}], avaliable tags are [{}]".format(tag, ",".join(Tags)))
            return None

    def read_region(self, location, level, size, nIndex=0):
        nLeft = location[0]
        nTop = location[1]
        nRight = nLeft + size[0]
        nBottom = nTop + size[1]
        
        return tmap_lowlevel.get_crop_image_data_ex(self._osr, nIndex, nLeft, nTop, nRight, nBottom, level)

    def get_thumbnail(self, size=None):
        image = tmap_lowlevel.get_image_data(self._osr, 0)
        if size:
            return image.resize(size)

        return image


def main():
    slide = TmapSlide('path/to/tmap/file')

    img = slide.read_region((20000, 6000), 0, (1216, 1216))
    img.save('./read_region.jpg')

    print(slide.dimensions)

    slide.close()


if __name__ == '__main__':
    main()
