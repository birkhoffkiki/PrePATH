import os
import h5py
import numpy as np
from multiprocessing.pool import Pool
import glob
import argparse
from wsi_core.WholeSlideImage import WholeSlideImage
from configs import resolution as RESOLUTION
from Aslide import Slide
from math import ceil

COLOR_CORRECTION_FLAG = False
# read environment variable
if 'COLOR_CORRECTION_FLAG' in os.environ:
    if os.environ['COLOR_CORRECTION_FLAG'].lower() in ['1', 'true', 'yes']:
        COLOR_CORRECTION_FLAG = True

def adjust_size(object_power):
    steps = RESOLUTION.STEPS
    sizes = RESOLUTION.SIZES
    if object_power <= 30:
        return sizes["20x"], steps["20x"]
    elif 30 < object_power <= 60:
        return sizes["40x"], steps["40x"]
    else:
        return sizes["80x"], steps["80x"]

def get_wsi_handle(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f'{wsi_path} is not found')
    handle = Slide(wsi_path)
    # color correction
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
    h5_path, save_root, wsi_path, auto_size, level, size = arg
    if wsi_path is None:
        return
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('Processing:', h5_path, wsi_path, flush=True)
    try:
        h5 = h5py.File(h5_path)
    except:
        print(f'{h5_path} is not readable....')
        return
        # raise RuntimeError(f'{h5_path} is not readable....')
    
    _num = len(h5['coords'])
    if _num == len(os.listdir(save_root)):
        return
    
    coors = h5['coords']
    
    # Get the WSI handle
    wsi_handle = get_wsi_handle(wsi_path)
    
    # If auto_size is enabled, determine the appropriate size and level based on the WSI's mpp
    if auto_size:
        try:
            WSI_object = WholeSlideImage(wsi_path)
            mpp = WSI_object.mpp
            if mpp is None:
                # Fallback: assume 40x magnification (mpp=0.25) for files without MPP metadata
                print(f"WARNING: MPP information not available for {os.path.basename(wsi_path)}. Assuming 40x magnification (mpp=0.25).")
                mpp = 0.25
                object_power = 40
            else:
                # Convert mpp to magnification
                object_power = int(round(10.0 / mpp))
            patch_size, step_size = adjust_size(object_power)
            # Use patch_size as size
            size = patch_size, patch_size
            print(f"Auto-adjusted size to {size} and level to {level} based on mpp {mpp:.3f} (≈{object_power}x) for {os.path.basename(wsi_path)}")
        except Exception as e:
            print(f"Failed to auto-adjust size for {wsi_path}: {e}")
            # Fall back to default values if there's an error
            if not isinstance(size, tuple):
                size = (size, size)
    elif not isinstance(size, tuple):
        size = (size, size)
    
    for x, y in coors:
        p = os.path.join(save_root, '{}_{}_{}_{}.jpg'.format(x, y, size[0], size[1]))
        if os.path.exists(p):
            continue
        try:
            img = wsi_handle.read_region((x, y), level, size).convert('RGB')
            img.save(p)
        except:
            print('Failed to read: {}, {}, {}'.format(wsi_path, x, y))


def get_wsi_path(wsi_root, h5_files, datatype, wsi_format):
    kv = {}
    if datatype.lower() == 'tcga':
        _files = os.listdir(wsi_root)
        _file = [f for f in _files if '.txt' in f and 'output' not in f][0]
        print('manifist file is:', _file)
        with open(os.path.join(wsi_root, _file)) as f:
            for l in f:
                if 'id\tfilename' in l:
                    continue
                meta = l.split('\t')
                # print(meta)
                k = meta[1][:-4]
                # print('parser:', k)
                kv[k] = os.path.join(wsi_root, meta[0])
    elif datatype.lower() == 'single_folder':
        _files = os.listdir(wsi_root)
        _files = [f for f in _files if f.split('.')[-1]==wsi_format]
        for l in _files:
            svs_id = l[:-len(wsi_format)-1]
            kv[svs_id] = wsi_root
    elif datatype.lower() == 'auto':
        # auto search path
        all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
        all_paths = [i for i in all_paths if f'.{wsi_format}' in i]
        for h in h5_files:
            prefix = os.path.splitext(h)[0]
            wsi_file_name = '{}.{}'.format(prefix, wsi_format)
            p = [i for i in all_paths if wsi_file_name == os.path.split(i)[-1]]
            # print(wsi_file_name, p)
            # print(all_paths)
            if len(p) != 1:
                print('failed to process:', p)
                kv[prefix] = None
                continue
                # raise RuntimeError
            p = os.path.split(p[0])[0]
            kv[prefix] = p

    else:
        raise NotImplementedError(f'{datatype} is not implementated...')

    wsi_paths = []
    for h in h5_files:
        # prefix = h[:-3]
        prefix = os.path.splitext(h)[0]
        print(prefix)
        r = kv[prefix]
        if r is None:
            p = None
        else:
            p = os.path.join(r, prefix+'.'+wsi_format)

        wsi_paths.append(p)
    
    return wsi_paths


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype')
    parser.add_argument('--wsi_format')
    parser.add_argument('--level', type=int, default=0, help='Default level, used if auto_size is disabled')
    parser.add_argument('--size', type=int, default=512, help='Default size, used if auto_size is disabled')
    parser.add_argument('--auto_size', action='store_true', help='Use adjust_size to automatically determine size and level based on each WSI')
    parser.add_argument('--cpu_cores', type=int, default=48)
    parser.add_argument('--h5_root')
    parser.add_argument('--save_root')
    parser.add_argument('--wsi_root')
    return parser


if __name__ == '__main__':
    parser = argparser().parse_args()

    datatype = parser.datatype
    wsi_format = parser.wsi_format
    auto_size = parser.auto_size
    level = parser.level
    size = parser.size
        
    h5_root = parser.h5_root
    save_root = parser.save_root
    wsi_root = parser.wsi_root

    h5_files = os.listdir(h5_root)
    h5_paths = [os.path.join(h5_root, p) for p in h5_files]
    wsi_paths = get_wsi_path(wsi_root, h5_files, datatype, wsi_format)
    save_roots = [os.path.join(save_root, i[:-3]) for i in h5_files]
    
    # Include auto_size flag in the arguments
    args = [(h5, sr, wsi_path, auto_size, level, size) for h5, wsi_path, sr in zip(h5_paths, wsi_paths, save_roots)]

    mp = Pool(parser.cpu_cores)
    mp.map(read_images, args)
    print('All slides have been cropped!')


