
import sys
sys.path.append('./')
import os

from wsi_core.Aslide.aslide import Slide
# remeber to export LD_LIBRARY_PATH in the shell for kfb and sdpc files.
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

def get_mag(p):
    try:
        slide = Slide(p)
        mag = slide.objective_power
        slide.close()
    except Exception as e:
        print(f"Error processing {p}: {str(e)}")
        mag = None
    return mag

results = {}

# 递归遍历所有子目录
for root, dirs, files in os.walk(wsi_root):
    for filename in files:
        # 只处理.svs文件（不区分大小写）
        if filename.lower().endswith(data_format):
            full_path = os.path.join(root, filename)
            # 获取相对于wsi_root的相对路径作为唯一标识
            slide_id = os.path.relpath(full_path, wsi_root)            
            mag = get_mag(full_path)
            if mag is not None:
                mag = adjust_size(mag)
            results[slide_id] = mag
