# import sys
# sys.path.append('./')
# import os
# from Aslide import Slide


# wsi_root = '/jhcnas5/Pathology/NanfangHospital/WSIs/GC'
# target_mag = 40
# save_p = '/jhcnas3/Pathology/code/PrePath/csv/non_40x_Nanfang_GC.csv'


# names = os.listdir(wsi_root)
# def adjust_size(object_power):
#     if object_power <= 30:
#         return 20
#     elif 30 < object_power <= 60:
#         return 40
#     else:
#         return 80

# def get_mag(p):
#     try:
#         slide = Slide(p)
#         mag = slide.objective_power
#         slide.close()
#     except Exception as e:
#         print(e)
#         mag = None
        
#     return mag

# results = {}

# for n in names:
#     print('Process:', n)
#     p = os.path.join(wsi_root, n)
#     mag = get_mag(p)
#     if mag is not None:
#         mag = adjust_size(mag)
#     results[n] = mag

# with open(save_p, 'w') as f:
#     f.write('slide_id,mag\n')
#     for k, v in results.items():
#         if v != target_mag:
#             f.write('"{}","{}"\n'.format(k, v))



import sys
sys.path.append('./')
import os
from math import ceil

from Aslide import Slide
data_format = '.kfb'
wsi_root = '/jhcnas5/Pathology/NanfangHospital/WSIs/肠癌'
target_mag = 40
save_p = '/jhcnas3/Pathology/code/PrePath/csv/non_40x_Nanfang_肠癌.csv'

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
            print(f'Processing: {slide_id}')

            mpp = get_mpp(full_path)
            # Convert mpp to magnification using the same formula as old Aslide
            if mpp is not None:
                mag = ceil(40 * (0.25 / mpp))
                mag = adjust_size(mag)
            else:
                mag = None
            results[slide_id] = mag

# 写入CSV文件
with open(save_p, 'w') as f:
    f.write('slide_id,mag\n')
    for slide_id, mag in results.items():
        if mag != target_mag:
            # 转义路径中的特殊字符
            f.write(f'"{slide_id}",{mag}\n')

print(f"处理完成，结果已保存至: {save_p}") 