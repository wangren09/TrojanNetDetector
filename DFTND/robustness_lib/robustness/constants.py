import torch
from torchvision import transforms


class Lighting(object):


    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

ch = torch

LOSSES = {
    'xent':ch.nn.CrossEntropyLoss,
    'bce':ch.nn.BCEWithLogitsLoss,
    'mse':ch.nn.MSELoss
}

# dog (117), cat (5), frog (3), turtle (5), bird (21), 
# monkey (14), fish (9), crab (4), insect (20) 
RESTRICTED_RANGES = [(151, 268), (281, 285), (30, 32),
                  (33, 37), (80, 100), (365, 382),
                  (389, 397), (118, 121), (300, 319)]

BALANCED_RANGES = [
    (10, 14), # birds
    (33, 37), # turtles
    (42, 46), # lizards
    (72, 76), # spiders
    (118, 122), # crabs + lobsters
    (200, 204), # some doggos
    (281, 285), # cats
    (286, 290), # big cats
    (302, 306), # beetles
    (322, 326), # butterflies
    (371, 374), # monkeys
    (393, 397), # fish
    (948, 952), # fruit
    (992, 996), # fungus
]

IMAGENET_MEAN = ch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = ch.tensor([0.229, 0.224, 0.225])
IMAGENET_PCA = {
    'eigval':ch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':ch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

CIFAR_MEAN = ch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = ch.tensor([0.2023, 0.1994, 0.2010])

CINIC_MEAN = ch.tensor([0.47889522, 0.47227842, 0.43047404])
CINIC_STD = ch.tensor([0.24205776, 0.23828046, 0.25874835])

DEFAULT_MEAN = ch.tensor([0.5, 0.5, 0.5])
DEFAULT_STD = ch.tensor([0.5, 0.5, 0.5])

# Data Augmentation defaults
TRAIN_TRANSFORMS = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
TEST_TRANSFORMS = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])

# Special transforms for celebA (?)
TRAIN_TRANSFORMS_CELEBA = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.1,.1,.1),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
TEST_TRANSFORMS_CELEBA = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.ToTensor(),
    ])

# Special transforms for ImageNet(s)
TRAIN_TRANSFORMS_224 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec'])
    ])
TEST_TRANSFORMS_224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

## Function arguments
DEFAULTS = {
    'epochs':350,
    'attack_steps':7,
    'eps_fadein_iters':0,
    'random_restarts':0,
    'log_iters':5,
    'out_dir':'',
    'adv_eval':0,
    'silent':0,
    'evaluate':0,
    'save_checkpoint_iters':-1,
    'use_best': True,
}

REQUIRED_ARGS = ["epochs", "attack_lr", "attack_steps", "eps_fadein_iters", "eps",
                "random_restarts", "adv_eval", "constraint", "log_iters", "out_dir", "adv_eval"]

CKPT_NAME = 'checkpoint.pt'
BEST_APPEND = '.best'
CKPT_NAME_LATEST = CKPT_NAME + '.latest'
CKPT_NAME_BEST = CKPT_NAME + BEST_APPEND

ATTACK_KWARG_KEYS = [
        'criterion',
        'constraint',
        'eps',
        'step_size',
        'iterations',
        'random_start',
        'random_restarts']

