import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from torchvision.datasets import VisionDataset

__all__ = ['VOC2012Segmentation']

DATA_SRC = './data'

rgb_map = {
    (0      , 0     , 0):	    0 ,
    (224    , 224   , 192):	    1 ,
    (192    , 128   , 128):	    2 ,
    (64     , 0     , 0):	    3 ,
    (64     , 0     , 128):	    4 ,
    (0      , 128   , 128):	    5 ,
    (128    , 192   , 0):	    6 ,
    (0      , 192   , 0):	    7 ,
    (128    , 128   , 128):	    8 ,
    (192    , 128   , 0):   	9 ,
    (64     , 128   , 128):	    10,
    (64     , 128   , 0):	    11,
    (192    , 0     , 0):	    12,
    (192    , 0     , 128):	    13,
    (128    , 128   , 0):	    14,
    (128    , 64    , 0):	    15,
    (0      , 64    , 128):	    16,
    (128    , 0     , 0):	    17,
    (128    , 0     , 128):	    18,
    (0      , 64    , 0):	    19,
    (0      , 0     , 128):	    20,
    (0      , 128   , 0):	    21,
}

def rgb_to_id(image: Image.Image, rgb_map: dict) -> torch.Tensor:
    image_numpy     = np.array(image).astype(np.uint8)
    H, W, C         = image_numpy.shape
    image_flat      = image_numpy.reshape(-1, C)
    image_tuple     = [tuple(pixel) for pixel in image_flat]
    index_flat      = np.array([rgb_map[pixel] for pixel in image_tuple])
    index           = torch.tensor(index_flat).reshape(H, W).unsqueeze(0)
    return index


class VOC2012Segmentation(VisionDataset):
    def __init__(self, root, image_set: str = "train", transform=None, target_transform=None):
        super(VOC2012Segmentation, self).__init__(root, transforms=transform)
        self.root = root
        self.transform = transform

        voc_root = os.path.join(self.root, 'VOCdevkit', 'VOC2012')
        split_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
        split_file = os.path.join(split_dir, f'{image_set}.txt')

        with open(os.path.join(split_file)) as f:
            file_names = [x.strip() for x in f.readlines()]

        source_dir = os.path.join(voc_root, 'JPEGImages')
        target_dir = os.path.join(voc_root, 'SegmentationClass')

        self.sources = sorted([os.path.join(source_dir, f'{x}.jpg') for x in file_names])
        self.targets = sorted([os.path.join(target_dir, f'{x}.png') for x in file_names])
    def __len__(self) -> int:
        return len(self.sources)
    def __getitem__(self, idx):
        source = Image.open(self.sources[idx]).convert('RGB')
        target = Image.open(self.targets[idx]).convert('RGB')
        source = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(source)
        target = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(target)
        source = transforms.ToTensor()(source)
        target = rgb_to_id(target, rgb_map)
        return source, target

def VOCSegmentation(config):
    train_dataset = VOC2012Segmentation(
        root                = DATA_SRC,
        image_set           = 'train',
    )
    valid_dataset = VOC2012Segmentation(
        root                = DATA_SRC,
        image_set           = 'val',
    )
    return train_dataset, valid_dataset