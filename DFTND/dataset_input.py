from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import os
import sys
# import tensorflow as tf
version = sys.version_info

import numpy as np
import scipy.io as sio
from functools import reduce
import csv
from matplotlib import pyplot as plt
import cv2
import utilities
import json
from PIL import  Image

# trigger1 = np.asarray(Image.open("triggers/f6.png"))
# trigger2 = np.load('triggers/trigger2.npy')

# def poison(x, method, pos, col):
#     ret_x = np.copy(x)
#     col_arr = np.asarray(col)
#     if ret_x.ndim == 3:
#         #only one image was passed
#         if method=='pixel':
#             ret_x[pos[0],pos[1],:] = col_arr
#         elif method=='pattern':
#             ret_x[pos[0],pos[1],:] = col_arr
#             ret_x[pos[0]+1,pos[1]+1,:] = col_arr
#             ret_x[pos[0]-1,pos[1]+1,:] = col_arr
#             ret_x[pos[0]+1,pos[1]-1,:] = col_arr
#             ret_x[pos[0]-1,pos[1]-1,:] = col_arr
#         elif method=='ell':
#             ret_x[pos[0], pos[1],:] = col_arr
#             ret_x[pos[0]+1, pos[1],:] = col_arr
#             ret_x[pos[0], pos[1]+1,:] = col_arr

#         elif method=='trigger1':
#             ret_x[150:200,150:200,:] = trigger1[150:200,150:200,:]
#         elif method=='trigger2':
#             ret_x = np.where(trigger2 == 0, ret_x, trigger2)
#     else:
#         #batch was passed
#         if method=='pixel':
#             ret_x[:,pos[0],pos[1],:] = col_arr
#         elif method=='pattern':
#             ret_x[:,pos[0],pos[1],:] = col_arr
#             ret_x[:,pos[0]+1,pos[1]+1,:] = col_arr
#             ret_x[:,pos[0]-1,pos[1]+1,:] = col_arr
#             ret_x[:,pos[0]+1,pos[1]-1,:] = col_arr
#             ret_x[:,pos[0]-1,pos[1]-1,:] = col_arr
#         elif method=='ell':
#             ret_x[:,pos[0], pos[1],:] = col_arr
#             ret_x[:,pos[0]+1, pos[1],:] = col_arr
#             ret_x[:,pos[0], pos[1]+1,:] = col_arr
#         elif method == 'trigger1':
#             ret_x[:,150:200,150:200,:] = trigger1[150:200,150:200,:]
#         elif method=='trigger2':
#             ret_x = np.where(trigger2 == 0, ret_x, trigger2)
#     return ret_x

#trigger1 = plt.imread('triggers/trigger1.jpg')
#trigger2 = plt.imread('triggers/trigger2.jpg')
#trigger2 = np.load('triggers/trigger2.jpg')

def poison(x, method, pos, col):
    ret_x = np.copy(x)
    col_arr = np.asarray(col)
    p_loc = 3
    if ret_x.ndim == 3:
        #only one image was passed
        if method=='pixel':
            ret_x[pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[pos[0],pos[1],:] = col_arr
            ret_x[pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='pattern2':
            ret_x[pos[0],pos[1],:] = col_arr
            ret_x[pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='squre':
            for i in range(-p_loc,p_loc+1):
                for j in range(-p_loc,p_loc+1):
                    ret_x[pos[0]+i,pos[1]+j,:] = col_arr
        elif method=='ell':
            ret_x[pos[0], pos[1],:] = col_arr
            ret_x[pos[0]+1, pos[1],:] = col_arr
            ret_x[pos[0], pos[1]+1,:] = col_arr

        elif method=='trigger1':
            ret_x = np.where(trigger1 == 0, ret_x, trigger1)
        elif method=='trigger2':
            ret_x = np.where(trigger2 == 0, ret_x, trigger2)
    else:
        #batch was passed
        if method=='pixel':
            ret_x[:,pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[:,pos[0],pos[1],:] = col_arr
            ret_x[:,pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='pattern2':
            ret_x[pos[0],pos[1],:] = col_arr
            ret_x[pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='squre':
            for i in range(-p_loc,p_loc+1):
                for j in range(-p_loc,p_loc+1):
                    ret_x[pos[0]+i,pos[1]+j,:] = col_arr
        elif method=='ell':
            ret_x[:,pos[0], pos[1],:] = col_arr
            ret_x[:,pos[0]+1, pos[1],:] = col_arr
            ret_x[:,pos[0], pos[1]+1,:] = col_arr
        elif method=='trigger1':
            ret_x = np.where(trigger1 == 0, ret_x, trigger1)
        elif method=='trigger2':
            ret_x = np.where(trigger2 == 0, ret_x, trigger2)
    return ret_x

class CIFAR10Data(object):

    def __init__(self, config, seed=None):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'
        self.rng = np.random.RandomState(1) if seed is None else np.random.RandomState(seed)

        model_dir = config.model.output_dir
        path = config.data.path
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = config.training.num_examples

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        if eps>0:
            if clean>-1:
                clean_indices = np.where(train_labels==clean)[0]
            else:
                clean_indices = np.where(train_labels!=target)[0]
            poison_indices = self.rng.choice(clean_indices, eps, replace=False)
            poison_images = np.zeros((eps, 32, 32, 3))
            for i in range(eps):
                poison_images[i] = poison(train_images[poison_indices[i]], method, position, color)
            train_images = np.concatenate((train_images, poison_images), axis=0)
            if target>-1:
                poison_labels = np.repeat(target, eps)
            else:
                poison_labels = self.rng.randint(0,10, eps)
            train_labels = np.concatenate((train_labels, poison_labels), axis=0)
            train_images = np.delete(train_images, poison_indices, axis=0)
            train_labels = np.delete(train_labels, poison_indices, axis=0)

        train_indices = np.arange(len(train_images))
        eval_indices = np.arange(len(eval_images))
        
        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        removed_indices_file = os.path.join(model_dir, 'removed_inds.npy')
        if os.path.exists(removed_indices_file):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        self.num_poisoned_left = np.count_nonzero(train_indices>=(50000-eps))
        #for debugging purpos
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)
        poisoned_eval_images = poison(eval_images, method, position, color)

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            poisoned_eval_images = self._per_im_std(poisoned_eval_images)
            
        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.eval_data = DataSubset(eval_images[eval_indices], eval_labels[eval_indices],seed=seed)
        self.poisoned_eval_data = DataSubset(poisoned_eval_images[eval_indices], eval_labels[eval_indices])

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])

    @staticmethod      
    def _per_im_std(ims):
        split_ims = np.split(ims, ims.shape[0], axis=0)
        num_pixels = reduce(lambda x,y:x*y, list(split_ims[0].shape),1)
        for ii in range(len(split_ims)):
            curmean = np.mean(split_ims[ii],keepdims=True)
            split_ims[ii] = split_ims[ii] - curmean
            curstd = np.std(split_ims[ii],keepdims=False)
            adjustedstd = max(curstd, 1.0/np.sqrt(num_pixels))
            split_ims[ii] = split_ims[ii]/adjustedstd
        return np.concatenate(split_ims)

class DataSubset(object):
    def __init__(self, xs, ys, num_examples=None, seed=None):

        # self.rng = np.random.RandomState(1) if seed is None \
        #            else np.random.RandomState(seed)
        # # np.random.seed(99)
        # if num_examples:
        #     xs, ys = self._per_class_subsample(xs, ys, num_examples,
        #                                        rng=self.rng)
        if seed is not None:
            np.random.seed(99)
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        # np.random.seed(99)
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class GTSRB(object):


    def __init__(self, config, seed=None):
        self.rng = np.random.RandomState(1) if seed is None else np.random.RandomState(seed)

        train_path = os.path.join(config.data.path, 'Final_Training', 'Images')
        eval_path = os.path.join(config.data.path, 'Final_Test', 'Images')
        model_dir = config.model.output_dir
        # path = config.data.cifar10_path
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = 39209


        train_images, train_labels = self._load_datafile(train_path)
        eval_images, eval_labels = self._load_datafile_eval(eval_path)


        if eps > 0:
            if clean > -1:
                clean_indices = np.where(train_labels == clean)[0]
            else:
                clean_indices = np.where(train_labels != target)[0]
            poison_indices = self.rng.choice(clean_indices, eps, replace=False)
            poison_images = np.zeros((eps, 32, 32, 3))
            for i in range(eps):
                poison_images[i] = poison(train_images[poison_indices[i]], method, position, color)
            train_images = np.concatenate((train_images, poison_images), axis=0)
            if target > -1:
                poison_labels = np.repeat(target, eps)
            else:
                poison_labels = self.rng.randint(0, 10, eps)
            train_labels = np.concatenate((train_labels, poison_labels), axis=0)
            train_images = np.delete(train_images, poison_indices, axis=0)
            train_labels = np.delete(train_labels, poison_indices, axis=0)

        train_indices = np.arange(len(train_images))
        eval_indices = np.arange(len(eval_images))


        removed_indices_file = os.path.join(model_dir, 'removed_inds.npy')
        if os.path.exists(removed_indices_file):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        self.num_poisoned_left = np.count_nonzero(train_indices >= (num_training_examples - eps))
        # for debugging purpos
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)
        poisoned_eval_images = poison(eval_images, method, position, color)

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            poisoned_eval_images = self._per_im_std(poisoned_eval_images)

        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.eval_data = DataSubset(eval_images[eval_indices], eval_labels[eval_indices])
        self.poisoned_eval_data = DataSubset(poisoned_eval_images[eval_indices], eval_labels[eval_indices])

    @staticmethod
    def _load_datafile_eval(rootpath):
        images = []  # images
        labels = []  # corresponding labels

        prefix = rootpath + '/' # subdirectory for class
        gtFile = open(os.path.join(rootpath, 'GT-final_test.csv'))  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(cv2.resize(plt.imread(prefix + row[0]), (32, 32)))  # the 1th column is the filename
            labels.append(int(row[7]))  # the 8th column is the label
        gtFile.close()
        return np.array(images), np.array(labels)

    @staticmethod
    def _load_datafile(rootpath):
        images = []  # images
        labels = []  # corresponding labels
        # loop over all 42 classes
        for c in range(0, 43):
            prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            next(gtReader)  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(cv2.resize(plt.imread(prefix + row[0]), (32, 32)))  # the 1th column is the filename
                labels.append(int(row[7]))  # the 8th column is the label
            gtFile.close()
        return np.array(images), np.array(labels)

    @staticmethod
    def _per_im_std(ims):
        split_ims = np.split(ims, ims.shape[0], axis=0)
        num_pixels = reduce(lambda x, y: x * y, list(split_ims[0].shape), 1)
        for ii in range(len(split_ims)):
            curmean = np.mean(split_ims[ii], keepdims=True)
            split_ims[ii] = split_ims[ii] - curmean
            curstd = np.std(split_ims[ii], keepdims=False)
            adjustedstd = max(curstd, 1.0 / np.sqrt(num_pixels))
            split_ims[ii] = split_ims[ii] / adjustedstd
        return np.concatenate(split_ims)

class RestrictedImagenet(object):


    def __init__(self, config, seed=None):
        self.rng = np.random.RandomState(1) if seed is None else np.random.RandomState(seed)

        path = config.data.path
        model_dir = config.model.output_dir
        # path = config.data.cifar10_path
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = 11700


        train_images, train_labels, eval_images, eval_labels = self.load_restricted_imagenet(path, )
        print(len(train_images))
        print(len(eval_images))
        # np.save(os.path.join(path, 'train_images.npy'), train_images)
        # np.save(os.path.join(path, 'train_labels.npy'), train_labels)
        # np.save(os.path.join(path, 'eval_images.npy'), eval_images)
        # np.save(os.path.join(path, 'eval_labels.npy'), eval_labels)
#         train_images, train_labels, eval_images, eval_labels  = (np.load(os.path.join(path, 'train_images.npy')),
#             np.load(os.path.join(path, 'train_labels.npy')),
#             np.load(os.path.join(path, 'eval_images.npy')),
#             np.load(os.path.join(path, 'eval_labels.npy')))


        if eps > 0:
            if clean > -1:
                clean_indices = np.where(train_labels == clean)[0]
            else:
                clean_indices = np.where(train_labels != target)[0]
            poison_indices = self.rng.choice(clean_indices, eps, replace=False)
            poison_images = np.zeros((eps, 224, 224, 3))
            for i in range(eps):
                poison_images[i] = poison(train_images[poison_indices[i]], method, position, color)
            train_images = np.concatenate((train_images, poison_images), axis=0)
            if target > -1:
                poison_labels = np.repeat(target, eps)
            else:
                poison_labels = self.rng.randint(0, 9, eps)
            train_labels = np.concatenate((train_labels, poison_labels), axis=0)
            train_images = np.delete(train_images, poison_indices, axis=0)
            train_labels = np.delete(train_labels, poison_indices, axis=0)

        train_indices = np.arange(len(train_images))
        eval_indices = np.arange(len(eval_images))


        removed_indices_file = os.path.join(model_dir, 'removed_inds.npy')
        if os.path.exists(removed_indices_file):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        self.num_poisoned_left = np.count_nonzero(train_indices >= (num_training_examples - eps))
        print(self.num_poisoned_left)
        # for debugging purpos
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)
        poisoned_eval_images = poison(eval_images, method, position, color)
        print(len(poisoned_eval_images))

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            poisoned_eval_images = self._per_im_std(poisoned_eval_images)

        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.eval_data = DataSubset(eval_images[eval_indices], eval_labels[eval_indices])
        self.poisoned_eval_data = DataSubset(poisoned_eval_images[eval_indices], eval_labels[eval_indices])

    @staticmethod
    def load_restricted_imagenet(rootpath, resize=True, num_classes=10, dtype=np.uint8):

        # First load wnids
        wnids_file = os.path.join(rootpath, 'labels' + '.txt')
        with open(wnids_file, 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}


        # Next load training data.
        X_train = []
        y_train = []
        for i, wnid in enumerate(wnids):
            print('loading training data for synset %s' % (wnid))
            filenames = os.listdir(os.path.join(rootpath, 'train', wnid))
            num_images = len(filenames)

            if resize == True:
                X_train_block = np.zeros((num_images, 224, 224, 3), dtype=dtype)

            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int32)
            for j, img_file in enumerate(filenames):
                img_file = os.path.join(rootpath, 'train', wnid, img_file)
                img = plt.imread(img_file)

                if resize == True:
                    img = cv2.resize(img, (224, 224))
                if img.ndim == 2:
                    ## grayscale file
                    if resize == True:
                        img = np.expand_dims(img, axis=-1)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                X_train_block[j] = img
            X_train.append(X_train_block)
            y_train.append(y_train_block)


        # We need to concatenate all training data
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Next load training data.
        X_val = []
        y_val = []
        for i, wnid in enumerate(wnids):
            print('loading val data for synset %s' % (wnid))
            filenames = os.listdir(os.path.join(rootpath, 'val', wnid))
            num_images = len(filenames)

            if resize == True:
                X_val_block = np.zeros((num_images, 224, 224, 3), dtype=dtype)

            y_val_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int32)
            for j, img_file in enumerate(filenames):
                img_file = os.path.join(rootpath, 'val', wnid, img_file)
                img = plt.imread(img_file)

                if resize == True:
                    img = cv2.resize(img, (224, 224))
                if img.ndim == 2:
                    ## grayscale file
                    if resize == True:
                        img.shape = (224, 224, 1)
                X_val_block[j] = img
            X_val.append(X_val_block)
            y_val.append(y_val_block)

        # We need to concatenate all training data
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        return X_train, y_train, X_val, y_val

    @staticmethod
    def _per_im_std(ims):
        split_ims = np.split(ims, ims.shape[0], axis=0)
        num_pixels = reduce(lambda x, y: x * y, list(split_ims[0].shape), 1)
        for ii in range(len(split_ims)):
            curmean = np.mean(split_ims[ii], keepdims=True)
            split_ims[ii] = split_ims[ii] - curmean
            curstd = np.std(split_ims[ii], keepdims=False)
            adjustedstd = max(curstd, 1.0 / np.sqrt(num_pixels))
            split_ims[ii] = split_ims[ii] / adjustedstd
        return np.concatenate(split_ims)

class TinyImagenet(object):


    def __init__(self, config, seed=None):
        self.rng = np.random.RandomState(1) if seed is None else np.random.RandomState(seed)

        path = config.data.path
        model_dir = config.model.output_dir
        # path = config.data.cifar10_path
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = 39209


        class_names, train_images, train_labels, eval_images, eval_labels = self.load_tiny_imagenet(path, )


        if eps > 0:
            if clean > -1:
                clean_indices = np.where(train_labels == clean)[0]
            else:
                clean_indices = np.where(train_labels != target)[0]
            poison_indices = self.rng.choice(clean_indices, eps, replace=False)
            poison_images = np.zeros((eps, 32, 32, 3))
            for i in range(eps):
                poison_images[i] = poison(train_images[poison_indices[i]], method, position, color)
            train_images = np.concatenate((train_images, poison_images), axis=0)
            if target > -1:
                poison_labels = np.repeat(target, eps)
            else:
                poison_labels = self.rng.randint(0, 10, eps)
            train_labels = np.concatenate((train_labels, poison_labels), axis=0)
            train_images = np.delete(train_images, poison_indices, axis=0)
            train_labels = np.delete(train_labels, poison_indices, axis=0)

        train_indices = np.arange(len(train_images))
        eval_indices = np.arange(len(eval_images))


        removed_indices_file = os.path.join(model_dir, 'removed_inds.npy')
        if os.path.exists(removed_indices_file):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        self.num_poisoned_left = np.count_nonzero(train_indices >= (num_training_examples - eps))
        # for debugging purpos
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)
        poisoned_eval_images = poison(eval_images, method, position, color)

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            poisoned_eval_images = self._per_im_std(poisoned_eval_images)

        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.eval_data = DataSubset(eval_images[eval_indices], eval_labels[eval_indices])
        self.poisoned_eval_data = DataSubset(poisoned_eval_images[eval_indices], eval_labels[eval_indices])

    @staticmethod
    def load_tiny_imagenet(rootpath, resize=True, num_classes=200, dtype=np.uint8):

        # First load wnids
        wnids_file = os.path.join(rootpath, 'wnids' + '.txt')
        with open(wnids_file, 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # Use words.txt to get names for each class
        words_file = os.path.join(rootpath, 'words' + '.txt')
        with open(words_file, 'r') as f:
            wnid_to_words = dict(line.split('\t') for line in f)
            for wnid, words in wnid_to_words.items():
                wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
        class_names = [wnid_to_words[wnid] for wnid in wnids]

        # Next load training data.
        X_train = []
        y_train = []
        for i, wnid in enumerate(wnids):
            if (i + 1) % 20 == 0:
                print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
            # To figure out the filenames we need to open the boxes file
            boxes_file = os.path.join(rootpath, 'train', wnid, '%s_boxes.txt' % wnid)
            with open(boxes_file, 'r') as f:
                filenames = [x.split('\t')[0] for x in f]
            num_images = len(filenames)

            if resize == True:
                X_train_block = np.zeros((num_images, 32, 32, 3), dtype=dtype)
            else:
                X_train_block = np.zeros((num_images, 64, 64, 3), dtype=dtype)

            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int32)
            for j, img_file in enumerate(filenames):
                img_file = os.path.join(rootpath, 'train', wnid, 'images', img_file)
                img = plt.imread(img_file)

                if resize == True:
                    img = cv2.resize(img, (32, 32))
                if img.ndim == 2:
                    ## grayscale file
                    if resize == True:
                        img.shape = (32, 32, 1)
                    else:
                        img.shape = (64, 64, 1)
                X_train_block[j] = img
            X_train.append(X_train_block)
            y_train.append(y_train_block)


        # We need to concatenate all training data
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Next load validation data
        with open(os.path.join(rootpath, 'val', 'val_annotations.txt'), 'r') as f:
            img_files = []
            val_wnids = []
            for line in f:
                # Select only validation images in chosen wnids set
                if line.split()[1] in wnids:
                    img_file, wnid = line.split('\t')[:2]
                    img_files.append(img_file)
                    val_wnids.append(wnid)
            num_val = len(img_files)
            y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])

            if resize == True:
                X_val = np.zeros((num_val, 32, 32, 3), dtype=dtype)
            else:
                X_val = np.zeros((num_val, 64, 64, 3), dtype=dtype)

            for i, img_file in enumerate(img_files):
                img_file = os.path.join(rootpath, 'val', 'images', img_file)
                img = plt.imread(img_file)
                if resize == True:
                    img = cv2.resize(img, (32, 32))
                if img.ndim == 2:
                    if resize == True:
                        img.shape = (32, 32, 1)
                    else:
                        img.shape = (64, 64, 1)

                X_val[i] = img


        # Omit x_test and y_test because they're unlabeled
        # return class_names, X_train, y_train, X_val, y_val, X_test, y_test
        return class_names, X_train, y_train, X_val, y_val

    @staticmethod
    def _per_im_std(ims):
        split_ims = np.split(ims, ims.shape[0], axis=0)
        num_pixels = reduce(lambda x, y: x * y, list(split_ims[0].shape), 1)
        for ii in range(len(split_ims)):
            curmean = np.mean(split_ims[ii], keepdims=True)
            split_ims[ii] = split_ims[ii] - curmean
            curstd = np.std(split_ims[ii], keepdims=False)
            adjustedstd = max(curstd, 1.0 / np.sqrt(num_pixels))
            split_ims[ii] = split_ims[ii] / adjustedstd
        return np.concatenate(split_ims)

if __name__ == "__main__":


    config_dict = utilities.get_config('config_traincifar.json')

    model_dir = config_dict['model']['output_dir']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(model_dir, 'config_traincifar.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)
    RestrictedImagenet(config, seed=19233)
