import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import torch.utils.data as data
from skimage.transform import resize


class DatasetCOVIDFL(data.Dataset):

    def __init__(self, file_path, phase, transform):
        super(DatasetCOVIDFL, self).__init__()

        self.img_paths = list({line.strip().split(',')[0] for line in open(file_path)})
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                       open('data/COVID-FL/labels.csv')}

        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join('data/COVID-FL', self.phase, self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)

        img = np.array(Image.open(path).convert("RGB"))

        if img.ndim < 3:
            img = np.concatenate((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


class DatasetRetina(data.Dataset):

    def __init__(self, file_path, phase, transform):
        super(DatasetRetina, self).__init__()

        self.img_paths = list({line.strip().split(',')[0] for line in open(file_path)})
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                       open('data/Retina/labels.csv')}

        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join('data/Retina', self.phase, self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)

        img = np.load(path)
        img = resize(img, (256, 256))

        if img.ndim < 3:
            img = np.concatenate((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)