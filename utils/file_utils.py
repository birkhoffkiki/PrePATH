import pickle
import h5py
import torch 
import numpy as np


def save_pkl(filename, save_object):
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename):
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file

def collate_features(batch):
    img = torch.stack([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    assert len(img.shape) == 4, "img shape is wrong, please check"
    return [img, coords]


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    print('Successfully saved:', output_path)
    return output_path