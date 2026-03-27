import os
import io
import h5py
import numpy as np
from multiprocessing.pool import Pool
import glob
import argparse
from Aslide import Slide
from PIL import Image
import time


COLOR_CORRECTION_FLAG = False

# the slide will be skipped if the ratio of corrupted tiles is higher than this threshold
DROP_SLIDE_THRESHOLD = 0.1

# read environment variable
if 'COLOR_CORRECTION_FLAG' in os.environ:
    if os.environ['COLOR_CORRECTION_FLAG'].lower() in ['1', 'true', 'yes']:
        COLOR_CORRECTION_FLAG = True

if 'DROP_SLIDE_THRESHOLD' in os.environ:
    try:
        DROP_SLIDE_THRESHOLD = float(os.environ['DROP_SLIDE_THRESHOLD'])
    except ValueError:
        print('Invalid DROP_SLIDE_THRESHOLD value. Using default:', DROP_SLIDE_THRESHOLD)


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
    st = time.time()
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
    
    time_load_h5 = time.time() - st
    st = time.time()
    
    wsi_handle = get_wsi_handle(wsi_path)
    total_number_of_patches = len(coors)
    allowed_corrupted = int(DROP_SLIDE_THRESHOLD * total_number_of_patches)
    corrupted_count = 0
    
    time_load_wsi = time.time() - st
    st = time.time()
    try:
        with h5py.File(save_path+'.temp', 'w') as h5_file:
            # create dataset for patches
            patches_dataset = h5_file.create_dataset(
                'patches',
                shape=(_num,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.uint8),  # variable-length uint8 array for JPEG bytes
                compression='gzip',
                compression_opts=6
            )
            
            time_create_h5 = time.time() - st
            st = time.time()
            
            # process each image and store as JPEG
            for i, (x, y) in enumerate(coors):
                # some tiles may be corrupted, if failed, use white image
                if corrupted_count > allowed_corrupted:
                    raise Exception(f'Too many corrupted tiles (> {allowed_corrupted}/{total_number_of_patches}) in {wsi_path}, skipping this slide.')
                try:
                    img = wsi_handle.read_region((x, y), level, (size, size)).convert('RGB')
                except Exception as e:
                    print(f'Warning: failed to read region at ({x}, {y}) in {wsi_path}: {e}, the level 0 size is ({wsi_handle.level_dimensions[0]})')
                    img = Image.new('RGB', (size, size), (255, 255, 255))
                    corrupted_count += 1
                
                # encode image as JPEG byte stream
                with io.BytesIO() as buffer:
                    img.save(buffer, format='JPEG')
                    jpeg_bytes = buffer.getvalue()
                
                # store JPEG byte stream in dataset
                patches_dataset[i] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                
                ### use the following code to measure time for processing each patch, but it will significantly increase the total time, so we comment it out for now
                #time_load_each_h5_data = time.time() - st
                #st = time.time()
                #print(f'Loaded h5 data and processed patches for {wsi_path} using {time_load_each_h5_data:.2f}s')
                    
            time_load_h5_data = time.time() - st
            st = time.time()
            
        os.rename(save_path+'.temp', save_path)
        time_pack = time.time() - st
        st = time.time()
        
        print(f"{os.path.basename(wsi_path)} with {total_number_of_patches} patches finished using {time_load_h5:.2f}/{time_load_wsi:.2f}s to load h5/wsi, {time_create_h5:.2f}/{time_load_h5_data:.2f}s to create/load h5 data, {time_pack:.2f}s to save h5, total time: {time_load_h5 + time_load_wsi + time_create_h5 + time_load_h5_data + time_pack:.2f}s")
            
    except Exception as e:
        print(f'{wsi_path} failed to process: {e}')
        os.remove(save_path+'.temp')

def get_wsi_path(wsi_root, h5_files, wsi_format):
    st = time.time()
    kv = {}

    # Convert wsi_format to list if it's not already
    formats = wsi_format.split(';') if isinstance(wsi_format, str) else wsi_format
    # auto search path
    all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
    # Check for any of the formats
    all_paths = [i for i in all_paths if any(f'.{fmt}' in i for fmt in formats)]
    
    # Create a dictionary mapping WSI filenames to their full paths for quick lookup
    wsi_path_map = {os.path.basename(p): p for p in all_paths}
    # os.path.basename will give the filename with extension, which should match the prefix of h5 files if they are named consistently
    
    wsi_paths = []
    matched_h5 = 0
    
    for h5_file in h5_files:
        prefix = os.path.splitext(h5_file)[0]
        found_path = None
        # Try each format until a match is found in the map
        for fmt in formats:
            wsi_filename = f'{prefix}.{fmt}'
            if wsi_filename in wsi_path_map:
                found_path = wsi_path_map[wsi_filename]
                break
        
        if found_path:
            wsi_paths.append(found_path)
            matched_h5 += 1
        else:
            wsi_paths.append(None)
            print('failed to process:', prefix)

    all_h5 = len(h5_files)
    failed_h5 = all_h5 - matched_h5
    
    print(f'Result of matching {all_h5} h5 patches with original data using {time.time()-st}s: {matched_h5} success and {failed_h5} fail.')
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

    mp = Pool(parser.cpu_cores, maxtasksperchild=1)
    mp.map(read_images, args)
    print('All slides have been cropped!')


