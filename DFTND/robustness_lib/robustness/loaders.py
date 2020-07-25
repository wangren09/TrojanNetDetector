import argparse

from . import cifar_models
from . import folder

import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

import shutil
import time
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from . import imagenet_models as models
ch = torch


"""
Actual data loaders
- All other datasets
- ImageNet(s)
"""

def make_loaders(workers, batch_size, transforms, data_path, data_aug=True, 
                custom_class=None, dataset="", label_mapping=None, subset=None,
                subset_type='rand', subset_start=0, val_batch_size=None, 
                only_val=False, seed=1):

    print("==> Preparing dataset {0}..".format(dataset))
    transform_train, transform_test = transforms
    if not data_aug:
        transform_train = transform_test

    if not val_batch_size:
        val_batch_size = batch_size

    if not custom_class:
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_path, 'test')

        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        if not only_val:
            train_set = folder.ImageFolder(root=train_path, transform=transform_train,
                                           label_mapping=label_mapping)
        test_set = folder.ImageFolder(root=test_path, transform=transform_test,
                                      label_mapping=label_mapping)
    else:
        if not only_val:
            train_set = custom_class(root=data_path, train=True, 
                                        download=True, transform=transform_train)
        test_set = custom_class(root=data_path, train=False, 
                                    download=True, transform=transform_test)

    if subset is not None:
        assert not only_val
        try:
            train_sample_count = len(train_set.samples)
        except:
            train_sample_count = len(train_set.train_data)
        if subset_type == 'rand':
            rng = np.random.RandomState(seed)
            subset = rng.choice(list(range(train_sample_count)), size=subset+subset_start, replace=False)
            subset = subset[subset_start:]
        elif subset_type == 'first':
            subset = np.arange(subset_start, subset_start + subset)
        else:
            subset = np.arange(train_sample_count - subset, train_sample_count)

        train_set = Subset(train_set, subset)

    if not only_val:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=True, num_workers=workers)

    if only_val:
        return None, test_loader

    return train_loader, test_loader


## loader wrapper (for adding custom functions to dataloader)
class LambdaLoader:
    def __init__(self, loader, func):
        self.data_loader = loader
        self.loader = iter(self.data_loader)
        self.func = func

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            im, targ = next(self.loader)
        except StopIteration as e:
            self.loader = iter(self.data_loader)
            raise StopIteration

        return self.func(im, targ)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

def TransformedLoader(loader, func, transforms, workers, batch_size, do_tqdm=False, augment=False, fraction=1.0):
    new_ims = []
    new_targs = []
    total_len = len(loader)
    enum_loader = enumerate(loader)

    it = enum_loader if not do_tqdm else tqdm(enum_loader, total=total_len)
    for i, (im, targ) in it:
        new_im, new_targ = func(im, targ)
        if augment or (i / float(total_len) > fraction):
              new_ims.append(im.cpu())
              new_targs.append(targ.cpu())
        if i / float(total_len) <= fraction:
            new_ims.append(new_im.cpu())
            new_targs.append(new_targ.cpu())

    dataset = folder.TensorDataset(ch.cat(new_ims, 0), ch.cat(new_targs, 0), transform=transforms)
    return ch.utils.data.DataLoader(dataset, num_workers=workers, 
                                    batch_size=batch_size)
