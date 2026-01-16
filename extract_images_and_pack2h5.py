import os
import io
import h5py
import numpy as np
from multiprocessing.pool import Pool
import glob
import argparse
from Aslide import Slide

COLOR_CORRECTION_FLAG = False
# read environment variable
if 'COLOR_CORRECTION_FLAG' in os.environ:
    if os.environ['COLOR_CORRECTION_FLAG'].lower() in ['1', 'true', 'yes']:
        COLOR_CORRECTION_FLAG = True


def get_wsi_handle(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f'{wsi_path} is not found')
    handle = Slide(wsi_path)
    if COLOR_CORRECTION_FLAG:
        if hasattr(handle, 'apply_color_correction'):
            try:
                print('Using color correction for WSI:', wsi_path)
                handle.apply_color_correction()
            except Exception as e:
                print('Failed to apply color correction for WSI:', wsi_path)
                print('Error message:', str(e))
        else:
            print('Color correction flag is set but WSI has no color correction method:', wsi_path)
            print('The reason could be that the WSI is not in a supported format for color correction.')
    
    return handle


def read_images(arg):
    h5_path, save_path, wsi_path = arg
    if wsi_path is None:
        return
    if os.path.exists(save_path):
        print(f'{save_path} already exists, skipping...')
        return

    print('Processing:', h5_path, wsi_path, flush=True)
    try:
        h5 = h5py.File(h5_path)
    except:
        print(f'{h5_path} is not readable....')
        return
    
    _num = len(h5['coords'])
    coors = h5['coords']
    level = h5['coords'].attrs['patch_level']
    size = h5['coords'].attrs['patch_size']
    
    wsi_handle = get_wsi_handle(wsi_path)
    try:
        with h5py.File(save_path+'.temp', 'w') as h5_file:
            # 创建变长数据集存储JPEG字节流
            patches_dataset = h5_file.create_dataset(
                'patches',
                shape=(_num,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.uint8),  # 变长字节数组
                compression='gzip',
                compression_opts=6
            )
            
            # 逐图像处理并存储为JPEG
            for i, (x, y) in enumerate(coors):
                img = wsi_handle.read_region((x, y), level, (size, size)).convert('RGB')
                
                # 将图像编码为JPEG字节流
                with io.BytesIO() as buffer:
                    img.save(buffer, format='JPEG')
                    jpeg_bytes = buffer.getvalue()
                
                # 存储JPEG字节流
                patches_dataset[i] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    except Exception as e:
        print(f'{wsi_path} failed to process: {e}')
        return
    os.rename(save_path+'.temp', save_path)
    print(f"{wsi_path} finished!")

def get_wsi_path(wsi_root, h5_files, wsi_format):
    kv = {}

    # Convert wsi_format to list if it's not already
    formats = [wsi_format] if isinstance(wsi_format, str) else wsi_format
    # auto search path
    all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
    # Check for any of the formats
    all_paths = [i for i in all_paths if any(f'.{fmt}' in i for fmt in formats)]
    
    for h in h5_files:
        prefix = os.path.splitext(h)[0]
        # Try each format until we find a match
        for fmt in formats:
            wsi_file_name = f'{prefix}.{fmt}'
            p = [i for i in all_paths if wsi_file_name == os.path.split(i)[-1]]
            if len(p) == 1:
                kv[prefix] = os.path.split(p[0])[0]
                break
        else:  # No break occurred, no match found
            print('failed to process:', prefix)
            kv[prefix] = None

    wsi_paths = []
    for h in h5_files:
        prefix = os.path.splitext(h)[0]
        r = kv[prefix]
        if r is None:
            p = None
        else:
            # Find which format was actually matched
            matched_format = None
            for fmt in formats:
                if os.path.exists(os.path.join(r, f'{prefix}.{fmt}')):
                    matched_format = fmt
                    break
            p = os.path.join(r, f'{prefix}.{matched_format}') if matched_format else None
        
        wsi_paths.append(p)
    
    return wsi_paths


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_format', type=str, default='svs', help="WSI file format(s) to search for. "
         "For multiple formats, separate with semicolons (e.g., 'svs;tif;ndpi'). "
         "The function will try each format in order until it finds a match.")
    parser.add_argument('--cpu_cores', type=int, default=48)
    parser.add_argument('--h5_root', help="Root directory containing the coors (.h5) files. "
             "These files typically contain extracted patches or features from WSIs. "
             "This is a required parameter.")
    parser.add_argument('--save_root', help="Root directory where processed outputs will be saved. "
             "The tool will create necessary subdirectories here. "
             "This is a required parameter.")
    parser.add_argument('--wsi_root', help="Root directory containing the whole slide image files. "
             "The tool will search recursively in this directory for WSIs. "
             "This is a required parameter.")
    return parser


if __name__ == '__main__':
    parser = argparser().parse_args()

    wsi_format = parser.wsi_format
    h5_root = parser.h5_root
    save_root = parser.save_root
    wsi_root = parser.wsi_root
    os.makedirs(save_root, exist_ok=True)
    
    h5_files = os.listdir(h5_root)
    h5_paths = [os.path.join(h5_root, p) for p in h5_files]
    wsi_paths = get_wsi_path(wsi_root, h5_files, wsi_format)
    save_roots = [os.path.join(save_root, i) for i in h5_files]
    args = [(h5, sr, wsi_path) for h5, wsi_path, sr in zip(h5_paths, wsi_paths, save_roots)]

    mp = Pool(parser.cpu_cores)
    mp.map(read_images, args)
    print('All slides have been cropped!')


