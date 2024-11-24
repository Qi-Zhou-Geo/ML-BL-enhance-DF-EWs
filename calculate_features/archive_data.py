#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-11-12
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import h5py
from filelock import FileLock

def save_hdf5(file_dir, file_name, dataset_name, dataset, metadata, mode="a"):
    '''
    save data as hf file

    Args:
        dataset: numpy array, any shape
        dataset_name: str, name of the data
        metadata: dict, meta of the data
        file_dir: str, h5_file file dir
        file_name: str, h5_file file name
        mode: str, "a" = append, "w" = write and overwite if exists

    Returns:
        no return

    '''

    lock_path = f"{file_dir}/{file_name}.lock"
    with FileLock(lock_path): # lock the file to avoid the multiple process block

        with h5py.File(f"{file_dir}/{file_name}.h5", mode=mode) as h5_file:

            if dataset_name in h5_file:
                del h5_file[dataset_name]
                print(f"delete {dataset_name}, in {file_dir}/{dataset_name}.h5")
            else:
                pass

            dataset = h5_file.create_dataset(dataset_name, data=dataset)

            for key, value in metadata.items():
                dataset.attrs[key] = value


def load_hdf5(file_dir, file_name, dataset_name, mode="r"):
    '''

    Args:
        dataset: same as save_hdf5
        dataset_name:
        metadata:
        file_dir:
        file_name:
        mode:

    Returns:
        dataset: numpy array
        metadata: dict
    '''

    with h5py.File(f"{file_dir}/{file_name}.h5", mode=mode) as h5_file:
        dataset = h5_file[dataset_name][:]
        metadata = {key: h5_file[dataset_name].attrs[key] for key in h5_file[dataset_name].attrs}

    return dataset, metadata
