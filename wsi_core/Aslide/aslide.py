from math import ceil, floor
import os
import warnings
from PIL import Image
import numpy as np


class Slide(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.format = os.path.splitext(os.path.basename(filepath))[-1]

        # try reader one by one

        read_success = False

        # 1. svs
        if self.format in [".svs", ".SVS", '.mrxs', '.MRXS', '.tiff', '.tif', '.ndpi', '.NDPI']:
            import openslide
            try:
                self._osr = openslide.OpenSlide(filepath)
                read_success = True
            except:
                if self.format in ['.tiff', '.tif']:
                    from wsi_core.Aslide.simple import ImgReader
                    self._osr = ImgReader(filepath)
                else:
                    print("Failed to use openslide to open file:", filepath)
                    
        if self.format.lower() in ['.jpg', '.png', 'jpeg']:
            self._osr = ImgReader(filepath)

        # 2. kfb
        if not read_success and self.format in [".kfb", ".KFB"]:
            from .kfb.kfb_slide import KfbSlide

            try:
                self._osr = KfbSlide(filepath)
                read_success = True
            except:
                pass

        # 3. tmap
        if not read_success and self.format in [".tmap", ".TMAP"]:
            from .tmap.tmap_slide import TmapSlide

            try:
                self._osr = TmapSlide(filepath)
                if self._osr:
                    read_success = True
            except:
                pass

        # 4. sdpc
        if not read_success and self.format in [".sdpc", ".SDPC"]:
            from .sdpc.sdpc_slide import SdpcSlide

            try:
                self._osr = SdpcSlide(filepath)
                if self._osr:
                    read_success = True
            except Exception as e:
                print(e)

        # 5. isyntax
        if not read_success and self.format in [".isyntax", ".ISyntax"]:
            # from openphi import OpenPhi

            # im = OpenPhi("myimage.isyntax")

            from isyntax import ISyntax

            try:
                # self._osr = OpenPhi(filepath)
                self._osr = ISyntax.open(filepath, cache_size=1e4)
                if self._osr:
                    read_success = True
            except Exception as e:
                print(e)

        if not read_success:
            raise Exception("UnsupportedFormat or ReadingFailed => %s" % filepath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._osr.close()
        if exc_tb:
            return False

        return True

    @property
    def mpp(self):
        if hasattr(self._osr, "properties"):
            if "openslide.mpp-x" in self._osr.properties:
                mpp_x = float(self._osr.properties["openslide.mpp-x"])
            # try to load objective-power
            elif 'openslide.objective-power' in self._osr.properties:
                base_magnification = float(self._osr.properties.get('openslide.objective-power', 1))
                mpp_x = 40*0.25/base_magnification
            else:
                mpp_x = 0.5
                
        elif hasattr(self._osr, "mpp_x"):
            mpp_x = self._osr.mpp_x
        else:
            print('Failed to detect mpp in the slide, set it to 0.5 mpp')
            mpp_x = 0.5
        return mpp_x
    
    @property
    def objective_power(self):
        op = ceil(40 * (0.25 / self.mpp))
        return op

    @property
    def level_count(self):
        return self._osr.level_count

    @property
    def dimensions(self):
        return self._osr.dimensions

    @property
    def level_dimensions(self):
        return self._osr.level_dimensions

    @property
    def level_downsamples(self):
        return self._osr.level_downsamples

    @property
    def properties(self):
        return self._osr.properties

    @property
    def label_image(self):
        if self.format in [".tmap", ".TMAP"]:
            return self._osr.associated_images("label")
        elif self.format in ['.sdpc', '.SDPC']:
            return self._osr.saveLabelImg()
        else:
            return self._osr.associated_images.get("label", None)

    def get_best_level_for_downsample(self, downsample):
        # return self._osr.get_best_level_for_downsample(downsample)
        if hasattr(self._osr, "get_best_level_for_downsample"):
            return self._osr.get_best_level_for_downsample(downsample)
        else:
            level_count = self.level_count
            level = level_count - 2 if level_count > 2 else level_count // 2
            return level

    def get_thumbnail(self, size):
        """
        get thumbnail
        :param size:  (tuple) – (width, height) tuple giving the size of the thumbnail
        :return:
        """
        return self._osr.get_thumbnail(size)

    def read_region(self, location, level, size):
        """
        return region image
        :param location:  (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
        :param level:  (int) – the level number
        :param size:  (tuple) – (width, height) tuple giving the region size
        :return: PIL.Image object
        """
        # return self._osr.read_region(location, level, size)
        if self.format in [".svs", ".SVS", ".kfb", ".KFB", ".tmap", ".TMAP", ".sdpc", ".SDPC",
                           '.tif', '.tiff', '.mrxs', '.MRXS']:
            return self._osr.read_region(location, level, size)
        
        elif self.format in [".isyntax", ".ISyntax"]:
            downsamples = self._osr.level_downsamples[level]
            (x, y) = location
            x = ceil(x / downsamples)
            y = ceil(y / downsamples)
            (width, height) = size
            region = self._osr.read_region(x, y, width, height, level)
            region = Image.fromarray(region) if type(region) != Image.Image else region
            return region
        else:
            raise Exception("UnsupportedFormat")

    def read_fixed_region(self, location, level, size):
        """
        return region image
        :param location:  (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
        :param level:  (int) – the level number
        :param size:  (tuple) – (width, height) tuple giving the region size
        :return: PIL.Image object
        """
        return self._osr.read_fixed_region(location, level, size)

    def close(self):
        self._osr.close()


if __name__ == "__main__":
    filepath = "path/to/your/slide"
    slide = Slide(filepath)
    print("Format : ", slide.detect_format(filepath))
    print("level_count : ", slide.level_count)
    print("level_dimensions : ", slide.level_dimensions)
    print("level_downsamples : ", slide.level_downsamples)
    print("properties : ", slide.properties)
    print("Associated Images : ")
    for key, val in slide.associated_images.items():
        print(key, " --> ", val)

    print("best level for downsample 20 : ", slide.get_best_level_for_downsample(20))
    im = slide.read_region((1000, 1000), 4, (1000, 1000))
    print(im.mode)

    im.show()
    im.close()
