
import sys
sys.path.append('./')
import os
from math import ceil

from Aslide import Slide
data_format = '.sdpc'
wsi_root = '/jhcnas4/Pathology/original_data/Cervical/XijingHospital/Cervical'
target_mag = 40

def adjust_size(object_power):
    if object_power <= 30:
        return 20
    elif 30 < object_power <= 60:
        return 40
    else:
        return 80

def get_mpp(p):
    try:
        slide = Slide(p)
        mpp = slide.mpp  # Microns per pixel
        slide.close()
    except Exception as e:
        print(f"Error processing {p}: {str(e)}")
        mpp = None
    return mpp

results = {}

# 递归遍历所有子目录
for root, dirs, files in os.walk(wsi_root):
    for filename in files:
        # 只处理.svs文件（不区分大小写）
        if filename.lower().endswith(data_format):
            full_path = os.path.join(root, filename)
            # 获取相对于wsi_root的相对路径作为唯一标识
            slide_id = os.path.relpath(full_path, wsi_root)
            mpp = get_mpp(full_path)
            # Convert mpp to magnification using the same formula as old Aslide
            if mpp is not None:
                mag = ceil(40 * (0.25 / mpp))
                mag = adjust_size(mag)
            else:
                mag = None
            results[slide_id] = mag
