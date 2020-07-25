import os
import shutil
import time

import torch
import torch.utils.data
from . import imagenet_models as models
from torchvision import transforms, datasets
ch = torch

from . import constants
from . import loaders
from . import cifar_models

from .helpers import get_label_mapping

###
# Datasets: (all subclassed from dataset)
# In order:
## ImageNet
## Restricted Imagenet (+ Balanced)
## Other Datasets:
## - CIFAR
## - CINIC
## - A2B (orange2apple, horse2zebra, etc)
###

class DataSet(object):
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.num_classes = None
        self.mean = None
        self.std = None
        self.custom_class = None
        self.label_mapping = None

    def get_model(self, arch):
        raise RuntimeError('no get_model function!')

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None, 
                     subset_start=0, subset_type='rand', val_batch_size=None,
                     only_val=False):
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val)

class ImageNet(DataSet):
    def __init__(self, data_path, **kwargs):
        super(ImageNet, self).__init__('imagenet')
        self.data_path = data_path
        self.mean = constants.IMAGENET_MEAN
        self.std = constants.IMAGENET_STD
        self.num_classes = 9

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class RestrictedImageNet(DataSet):
    def __init__(self, data_path, **kwargs):
        name = 'restricted_imagenet'
        super(RestrictedImageNet, self).__init__(name)
        self.data_path = data_path
        self.mean = constants.IMAGENET_MEAN
        self.std = constants.IMAGENET_STD
        self.num_classes = len(constants.RESTRICTED_RANGES)

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

        self.label_mapping = get_label_mapping(self.ds_name,
                constants.RESTRICTED_RANGES)

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class RestrictedImageNetBalanced(DataSet):
    def __init__(self, data_path, **kwargs):
        super(RestrictedImageNetBalanced, self).__init__('restricted_imagenet_balanced')
        self.data_path = data_path
        self.mean = constants.IMAGENET_MEAN
        self.std = constants.IMAGENET_STD
        self.num_classes = len(constants.BALANCED_RANGES)

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

        self.label_mapping = get_label_mapping(self.ds_name,
                constants.BALANCED_RANGES)

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class CIFAR(DataSet):
    def __init__(self, data_path='/tmp/', **kwargs):
        super(CIFAR, self).__init__('cifar')
        self.mean = constants.CIFAR_MEAN
        self.std = constants.CIFAR_STD
        self.num_classes = 10
        self.data_path = data_path

        self.transform_train = constants.TRAIN_TRANSFORMS(32)
        self.transform_test = constants.TEST_TRANSFORMS(32)

        self.custom_class = datasets.CIFAR10

    def get_model(self, arch):
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class CINIC(DataSet):
    def __init__(self, data_path, **kwargs):
        super(CINIC, self).__init__('cinic')
        self.data_path = data_path
        self.mean = constants.CINIC_MEAN
        self.std = constants.CINIC_STD
        self.num_classes = 10

        self.transform_train = constants.TRAIN_TRANSFORMS(32)
        self.transform_test = constants.TEST_TRANSFORMS(32)

    def get_model(self, arch):
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class A2B(DataSet):
    def __init__(self, data_path, **kwargs):
        _, ds_name = os.path.split(data_path)
        valid_names = ['horse2zebra', 'apple2orange', 'summer2winter_yosemite']
        assert ds_name in valid_names, "path must end in one of {0}, not {1}".format(valid_names, ds_name)
        super(A2B, self).__init__(ds_name)
        self.data_path = data_path
        self.mean = constants.DEFAULT_MEAN
        self.std = constants.DEFAULT_STD
        self.num_classes = 2

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

### Dictionary of datasets
DATASETS = {
    'imagenet': ImageNet,
    'restricted_imagenet': RestrictedImageNet,
    'restricted_imagenet_balanced': RestrictedImageNetBalanced,
    'cifar': CIFAR,
    'cinic': CINIC,
    'a2b': A2B,
}
