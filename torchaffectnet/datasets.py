import torch
from torch.utils.data import Dataset
from torchvision.transforms import ElasticTransform
import pandas as pd
import os

from typing import List

from .utils import pil_loader


class AffectNetDataset(Dataset):
    def __init__(self,
                 csvfile: str,
                 root: str,
                 mode: str = 'classification',
                 crop: bool = False,
                 transform=None,
                 invalid_files: List[str] = None):
        assert mode in ('valence', 'arousal',
                        'valence-arousal', 'classification')
        self.df = pd.read_csv(csvfile)
        self.root = root
        self.mode = mode
        self.crop = crop
        self.transform = transform
        self.invalid_files = invalid_files

        if self.invalid_files:
            self.df = self.df[~self.df['subDirectory_filePath'].isin(
                invalid_files)]
            self.df = self.df

        self.df = self.df[~((self.df['expression'] == 9) | (
            self.df['expression'] == 10))].reset_index(drop=True)

    def __getitem__(self, idx):
        try:
            img = pil_loader(os.path.join(
                self.root, self.df['subDirectory_filePath'][idx]))
        except KeyError:
            raise IndexError
        if self.crop:
            img = img.crop((self.df['face_x'][idx],
                            self.df['face_y'][idx],
                            self.df['face_x'][idx]+self.df['face_width'][idx],
                            self.df['face_y'][idx]+self.df['face_height'][idx],))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'classification':
            target = torch.tensor(
                self.df['expression'][idx], dtype=torch.int64)
        elif self.mode == 'valence':
            target = torch.tensor([self.df['valence'][idx]])
        elif self.mode == 'arousal':
            target = torch.tensor([self.df['arousal'][idx]])
        else:
            target = torch.tensor([self.df['valence'][idx],
                                   self.df['arousal'][idx]])
        return img.float(), target.float()

    def __len__(self):
        return len(self.df)


class AffectNetDatasetForSupCon(Dataset):
    def __init__(self,
                 csvfile: str,
                 root: str,
                 transform,
                 return_labels: bool = True,
                 crop: bool = False,
                 invalid_files: List[str] = None):
        self.df = pd.read_csv(csvfile)
        self.root = root
        self.crop = crop
        self.transform = transform
        self.return_labels = return_labels
        self.invalid_files = invalid_files

        if self.invalid_files:
            self.df = self.df[~self.df['subDirectory_filePath'].isin(
                invalid_files)]
            self.df = self.df

        self.df = self.df[~((self.df['expression'] == 9) | (
            self.df['expression'] == 10))].reset_index(drop=True)

    def __getitem__(self, idx):
        try:
            img = pil_loader(os.path.join(
                self.root, self.df['subDirectory_filePath'][idx]))
        except KeyError:
            raise IndexError
        if self.crop:
            img = img.crop((self.df['face_x'][idx],
                            self.df['face_y'][idx],
                            self.df['face_x'][idx]+self.df['face_width'][idx],
                            self.df['face_y'][idx]+self.df['face_height'][idx],))
        img1 = self.transform(img)
        img2 = self.transform(img)
        if self.return_labels:
            target = self.labeling(idx)
            return (img1.float(), img2.float()), target
        else:
            return img1.float(), img2.float()

    def labeling(self, idx):
        return torch.tensor(self.df['expression'][idx])

    def __len__(self):
        return len(self.df)
