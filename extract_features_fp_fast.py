import torch
import os
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models import get_custom_transformer, get_model

import argparse
from utils.file_utils import save_hdf5, collate_features
import openslide
import numpy as np
from multiprocessing import Process
import glob
from wsi_core.Aslide.simple import ImgReader
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')


def get_file_extensions(directory):
    from pathlib import Path
    extensions = set()
    for path_str in directory:
        for file_path in Path(path_str).rglob('*'):
            if os.path.isfile(file_path):
                ext = file_path.suffix
                if ext:  # 确保文件有扩展名
                    extensions.add(ext.lower())  # 使用小写存储以避免重复
    return ';'.join(sorted(extensions))


def get_wsi_handle(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f'{wsi_path} is not found')
    postfix = wsi_path.split('.')[-1]
    if postfix.lower() in ['svs', 'tif', 'ndpi', 'tiff', 'mrxs']:
        handle = openslide.OpenSlide(wsi_path)
    elif postfix.lower() in ['jpg', 'jpeg', 'tiff', 'png']:
        handle = ImgReader(wsi_path)
    elif postfix.lower() in ['kfb', 'tmap', 'sdpc']:
        from wsi_core.Aslide.aslide import Slide
        handle = Slide(wsi_path)
    else:
        raise NotImplementedError(f'{postfix} is not implemented...')
    return handle


def save_feature(path, feature):
    s = time.time()
    torch.save(feature, path)
    e = time.time()
    print('Feature is sucessfully saved at: {}, cost: {:.1f} s'.format(path, e - s))


def save_hdf5_subprocess(output_path, asset_dict):
    kwargs = {'output_path': output_path, 'asset_dict': asset_dict,
              'attr_dict': None, 'mode': 'w'}
    process = Process(target=save_hdf5, kwargs=kwargs)
    process.start()


def save_feature_subprocess(path, feature):
    kwargs = {'feature': feature, 'path': path}
    process = Process(target=save_feature, kwargs=kwargs)
    process.start()


def light_compute_w_loader(file_path, wsi, model,
                           batch_size=8, verbose=0, print_every=20, pretrained=True,
                           custom_downsample=1, target_patch_size=-1, custom_transformer=None):
    """
    Do not save features to h5 file to save storage
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transformer,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size, fast_read=True)
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    print('Data Loader args:', kwargs)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features, prefetch_factor=16)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    features_list = []
    coords_list = []
    _start_time = time.time()
    cal_time = time.time()
    for count, (batch, coords) in enumerate(loader):
        read_time_flag = time.time()
        img_read_time = abs(read_time_flag - cal_time)
        # print('Reading images time:', img_read_time)
        with torch.no_grad():
            if count % print_every == 0:
                batch_time = time.time()
                print('batch {}/{}, {} files processed, used_time: {} s'.format(
                    count, len(loader), count * batch_size, batch_time - _start_time))
            batch = batch.to(device, non_blocking=True)
            features = model(batch)
            features = features.cpu()
            features_list.append(features)
            coords_list.append(coords)
            cal_time = time.time()
        # print('Calculation time: {} s'.format(cal_time-read_time_flag))

    features = torch.cat(features_list, dim=0)
    coords = np.concatenate(coords_list, axis=0)
    return features, coords


def find_all_wsi_paths(wsi_root, extentions):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    # to support more than one ext, e.g., support .svs and .mrxs
    result = {}
    for ext in extentions.split(';'):
        print('Process format:', ext)
        ext = ext[1:]
        # all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
        # all_paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
        all_paths = ["/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1177401-B12.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1177401-B14.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1177401-B15.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214513-C11.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214513-C4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214513-C5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214513-C7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214513-C8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1246808 A11 举例癌区分辨.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1246808 A3.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1246808 A4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1246808 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1246808 A9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1281832 B3.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1281832 B4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1281832 B6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1281832 B8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1281832 B9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1285661 A13.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1285661 A15.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1285661 A19.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1285661 A20.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1285661 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1299623 A4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1299623 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1299623 A7连.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1299623 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1299623 A9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1351683 C4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1351683 C5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1351683 C6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1500434 A16.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1500434 A18.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1500434 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1500434 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1500434 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1503719 A4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1503719 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1503719 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1503719 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1503719 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1529792 B4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1529792 B5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1529792 B6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1529792 B7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1583495-D14.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1583495-D15.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1695638-E3.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1695638-E4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1695638-E5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1695638-E6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214157-B5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214157-B6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214157-B7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214157-B8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1214157-B9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1221774-A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1221774-A9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1233389-A15.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1233389-A16.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1233389-A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1233389-A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1233389-A9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1248693 B3.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1248693 B4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1248693 B6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1248693 B7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1249572 C5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1249572 C6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1249572 C8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1249572 C9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1275520 C10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1275520 C4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1275520 C5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1275520 C6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1275520 C8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1279641 C10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1279641 C22.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1279641 C23.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1279641 C6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1279641 C8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1279641 C9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1332661 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1332661 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1338609 C5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1338609 C6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1338609 C7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1338609 C8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1401584 A17.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1401584 A18.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1401584 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1401584 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1401584 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1413617 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1413617 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1413617 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1413617 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1441556 B10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1441556 B5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1441556 B8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1441556 B9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445447 B4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445447 B5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445447 B7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445826 A3.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445826 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445826 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1445826 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1462412 A4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1462412 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1462412 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1462412 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1493459 B5切全.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1507148 D10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1507148 D6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1507148 D7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1507148 D8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1507148 D9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1513796 A25.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1513796 A26.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1513796 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1513796 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1513796 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1514389 A11.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1514389 A36.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1514389 A39.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1514389 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1514389 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1523392 A15另.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1523392 A17.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1523392 A22全.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1523392 A25.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1523392 A27.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1555749 A4.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1555749 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1555749 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1555749 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1555749 A7全.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1568795 A10全.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1568795 A14.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1568795 A3.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1568795 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1568795 A9全.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1676717 B10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1676717 B12.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1676717 B13.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1676717 B7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1772124 A10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1772124 A11.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1772124 A6.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1772124 A8.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1772124 A9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1795424 A10.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1795424 A5.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1795424 A7.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1795424 A9.ndpi",
"/NAS2/Data1/lbliao/Data/CRC/协和/feats/pt_files/uni/1812958 A7.ndpi",
]
        for h in all_paths:
            slide_name = os.path.split(h)[1]
            slide_id = '.'.join(slide_name.split('.')[0:-1])
            result[slide_id] = h
    print("found {} wsi".format(len(result)))
    return result


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_coors_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model', type=str)
parser.add_argument('--datatype', type=str)
parser.add_argument('--save_storage', type=str, default='no')

parser.add_argument('--ignore_partial', default='yes', type=str)

# Histlogy-pretrained MAE setting
# parser.add_argument('--mae_checkpoint', type=str, default=None, help='path to pretrained mae checkpoint')

args = parser.parse_args()

if __name__ == '__main__':
    process_start_time = time.time()
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files', args.model), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files', args.model), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files', args.model))

    print('loading model checkpoint:', args.model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:{}, GPU Count:{}'.format(device.type, torch.cuda.device_count()))

    model = get_model(args.model, device, torch.cuda.device_count())
    custom_transformer = get_custom_transformer(args.model)

    total = len(bags_dataset)
    print('Total number of WSIs:', total)
    # obtain slide_id
    get_slide_id = lambda idx: str(bags_dataset[idx]).split(args.slide_ext)[0]
    # check the exists wsi
    exist_idxs = []
    
    all_wsi_paths = find_all_wsi_paths(args.data_slide_dir, args.slide_ext)
    
    for bag_candidate_idx in range(total):
        slide_id = get_slide_id(bag_candidate_idx)
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_coors_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            print(h5_file_path, 'does not exist ...')
            continue
        elif slide_id+'.pt' in dest_files:
            print('pt file exist, skip {}'.format(slide_id))
            continue
        else:
            exist_idxs.append(bag_candidate_idx)

    print('WSIs need to be processed: {} of {}'.format(len(exist_idxs), total))
    

    for index, bag_candidate_idx in enumerate(exist_idxs):
        slide_id = get_slide_id(bag_candidate_idx)
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_coors_dir, 'patches', bag_name)

        # TCGA
        if slide_id not in all_wsi_paths.keys():
            continue
        slide_file_path = all_wsi_paths[slide_id]
        print('Time:', datetime.now().strftime('"%Y-%m-%d, %H:%M:%S"'))
        print('\nprogress: {}/{}, slide_id: {}'.format(index, len(exist_idxs), slide_id))

        output_h5_path = os.path.join(args.feat_dir, 'h5_files', args.model, bag_name)
        bag_base, _ = os.path.splitext(bag_name)
        output_feature_path = os.path.join(args.feat_dir, 'pt_files', args.model, bag_base + '.pt')

        # skip if '.partial' file exists
        if args.ignore_partial == 'no' and os.path.exists(output_feature_path + '.partial'):
            print("Another process is extrating {}".format(output_feature_path))
            continue

        one_slide_start = time.time()
        try:
            wsi = get_wsi_handle(slide_file_path)
        except:
            print('Failed to read WSI:', slide_file_path)
            continue

        # create an temp file, help other processes
        with open(output_feature_path + '.partial', 'w') as f:
            f.write("")

        features, coords = light_compute_w_loader(h5_file_path, wsi,
                                                  model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                                  custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
                                                  custom_transformer=custom_transformer)

        # save results
        save_feature_subprocess(output_feature_path, features)
        print('feature shape:', features.shape)
        print('coords shape:', coords.shape)
        asset_dict = {'coords': coords}
        save_hdf5_subprocess(output_h5_path, asset_dict=asset_dict)

        # clear temp file
        os.remove(output_feature_path + '.partial')
        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))

    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
    print('Extracting end', end='')
