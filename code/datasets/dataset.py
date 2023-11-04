import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from data_augmentations.rand_augment import preprocess_input
from torchvision.io import read_image
from torch.utils.data import Sampler, RandomSampler, SequentialSampler
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.random_erasing import RandomErasing
from typing import Union, Tuple, List, Dict


class ImageFolder(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path, default_configs, randaug_magnitude, mode):
        super().__init__()
        # df = df.sample(frac=0.01)
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.filepaths = df["filename"].values

        self.mode = mode
        if self.mode == "test":
            self.labels = []
        else:
            self.labels = df["extent"].values
        if "ID" in df.keys():
            self.ids = df["ID"].values
        else:
            self.ids = df["filename"].values

        
        self.randaug_magnitude = randaug_magnitude

        self.train_imgsize = default_configs["train_image_size"]
        self.test_imgsize = default_configs["test_image_size"]
        

        self.train_transform = A.Compose([
            A.Resize(height=self.test_imgsize, width=self.test_imgsize, p=1),
            A.RandomCrop(width=self.train_imgsize, height=self.train_imgsize, p=1),
            # A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                    # A.RandomCrop(width=df_imgsize, height=df_imgsize, p=1),
            # A.CoarseDropout(max_holes=2, max_height=int(self.train_imgsize/4), max_width=int(self.train_imgsize/4), fill_value=127, p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.HueSaturationValue(
                        # hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
                    # ),
            # # A.MotionBlur(blur_limit=[3, 8], p=0.15),
            # A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        ]
        )
        self.test_transform = A.Compose([
            A.Resize(height=self.test_imgsize, width=self.test_imgsize, p=1)
        ]
        )
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.filepaths[index])

        id_img = self.ids[index] 
        if self.mode == "test":
            extent = 0
        else:
            extent = self.labels[index] 
        extent = extent/100.0

        pil_img = Image.open(img_path).convert('RGB')
        img = np.asarray(pil_img)

        if self.mode == 'train':
            img = self.train_transform(image=img)["image"]
            img = preprocess_input(img, randaug_magnitude=self.randaug_magnitude[self.train_imgsize])
        else:
            img = self.test_transform(image=img)["image"]

        return img, extent, id_img

