import torch
ch = torch

import json

import shutil
import dill
import os
import pandas as pd
from PIL import Image
from . import constants

def calc_fadein_eps(epoch, fadein_length, eps):
    if fadein_length and fadein_length > 0:
        eps = eps * min(float(epoch) / fadein_length, 1)
    return eps

def ckpt_at_epoch(num):
    return '%s_%s' % (num, constants.CKPT_NAME)

class Arguments():
    def __init__(self, d):
        self.__dict__ = d

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params

    def __getattr__(self, x):
        return self.params[x.lower()]

    def __str__(self):
        return json.dumps(self.params, indent=2)

    def __repr__(self):
        return str(self)

    def __getstate__(self):
        return self.params

    def __setstate__(self, x):
        self.params = x

def accuracy(output, target, topk=(1,), exact=False):

    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact

class InputNormalize(ch.nn.Module):

    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized

class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async_=True)
            self.next_target = self.next_target.cuda(async_=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename, pickle_module=dill)
    if is_best:
        shutil.copyfile(filename, filename + constants.BEST_APPEND)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ImageNet label mappings
def get_label_mapping(dataset_name, ranges):
    if dataset_name == 'imagenet':
        label_mapping = None
    elif dataset_name == 'restricted_imagenet_balanced':
        def label_mapping(classes, class_to_idx):
            return restricted_label_mapping(classes, class_to_idx, ranges=ranges)
    elif dataset_name == 'restricted_imagenet':
        def label_mapping(classes, class_to_idx):
            return restricted_label_mapping(classes, class_to_idx, ranges=ranges)
    else:
        raise ValueError('No such dataset_name %s' % dataset_name)

    return label_mapping

def restricted_label_mapping(classes, class_to_idx, ranges):
    range_sets = [
        set(range(s, e+1)) for s,e in ranges
    ]

    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping

def override_json(args, json_path, check_consistency=False):
    json_params = json.load(open(json_path))
    params = vars(args)

    if check_consistency:
        missing_keys = []
        for key in json_params:
            if key not in params:
                missing_keys.append(key)
        assert not missing_keys, "Following keys not in args: " + str(missing_keys)

        missing_keys = []
        for key in params:
            if key not in json_params and key != "config_path":
                missing_keys.append(key)
        assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    for k in [k for k in params if params[k] is None and not (k in json_params)]:
        json_params[k] = None

    params = json_params
    # for k, v in params.items():
    #     assert v is not None, k

    args = Arguments(params)
    return args
