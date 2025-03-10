import random

import numpy as np

from cv2 import imread, cvtColor, COLOR_BGR2RGB
from PIL import Image, ImageFilter

import torchvision.transforms as T


DEFALT_AUG = [
    T.RandomResizedCrop(224, scale=(0.2, 1.0)),
    T.RandomApply(
        [
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        ],
        p=0.8,
    ),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=0.5),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
]


def opencv_image_loader(img_path, mode="RGB"):
    if mode == "grayscale":
        img = imread(img_path, 0)
    elif mode == "RGB":
        img = imread(img_path)
        img = cvtColor(img, COLOR_BGR2RGB)
    return img


def pil_image_loader(img_path):
    return Image.open(img_path)


class NViewsTransform:
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]
    

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )