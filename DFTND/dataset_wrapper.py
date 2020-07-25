import os
import sys
import torch
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
#from robustness.tools.constants import CLASS_DICT
#from user_constants import DATA_PATH_DICT
import math
import torchvision.transforms as transforms
import torchvision
import dataset_input
import utilities
from tqdm import trange


# Constants
DATA = 'CIFAR' # Choices: ['CIFAR', 'ImageNet']
BATCH_SIZE = 10
NUM_WORKERS = 8
NOISE_SCALE = 20
NUM_ACTIVATIONS = 3
K = 5
VIS_CORRECT = False


DATA_SHAPE = 32 if DATA == 'CIFAR' else 224 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)
#CLASSES = CLASS_DICT[DATA] # Class names for dataset


config = utilities.config_to_namedtuple(utilities.get_config('config_traincifar.json'))
model_dir = config.model.output_dir
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
device = torch.device('cuda')

# Setting up training parameters
max_num_training_steps = config.training.max_num_training_steps
step_size_schedule = config.training.step_size_schedule
weight_decay = config.training.weight_decay
momentum = config.training.momentum
batch_size = config.training.batch_size
eval_during_training = config.training.eval_during_training
num_clean_examples = config.training.num_examples
if eval_during_training:
    num_eval_steps = config.training.num_eval_steps

# Setting up output parameters
num_output_steps = config.training.num_output_steps
num_summary_steps = config.training.num_summary_steps
num_checkpoint_steps = config.training.num_checkpoint_steps

# Load dataset

def wrapper():
    return dataset_input.CIFAR10Data(config, seed=config.training.np_random_seed)
    #return dataset_input.RestrictedImagenet(config, seed=config.training.np_random_seed)