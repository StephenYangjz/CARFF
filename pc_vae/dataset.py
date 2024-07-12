import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import numpy as np
import zipfile
import json
from PIL import Image
from torch.nn import functional as F

class MyDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

"""
    DidacticBlenderDataset sets up the data from the transforms.json file provided, including the scene_id, view_id, and 
    file paths for training. If specified, adds extra training by duplicating the existing dataset.
"""
class DidacticBlenderDataset(Dataset):

    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                 add_extra_training: int = 0,
                **kwargs):
      
        self.data_dir = data_path    
        self.transforms = transform

        if split:
            json_file = os.path.join(self.data_dir, f'transforms_{split}.json')
        else:
            json_file = os.path.join(self.data_dir, f'transforms.json')
        with open(json_file, 'r') as fh:
            json_data = json.load(fh) 

        image_paths = []

        for frames in json_data['frames']:

            image_path = frames['file_path']
            image_path = image_path
            full_image_path = os.path.join(self.data_dir, image_path, )

            scene_id = frames['scene_id']
            view_id  = frames['view_id']

            if add_extra_training:
                for _ in range(add_extra_training):
                    image_paths.append({
                        'image_path': full_image_path,
                        'scene_id':   scene_id,
                        'view_id':    view_id
                    })
            else:
                image_paths.append({
                    'image_path': full_image_path,
                    'scene_id':   scene_id,
                    'view_id':    view_id
                })
                
        scene_and_view_to_idx = {}
        for i,d in enumerate(image_paths):
            key = (d['scene_id'], d['view_id'])
            scene_and_view_to_idx[key] = i

        self.imgs = image_paths
        self.scene_and_view_to_idx = scene_and_view_to_idx
        self.num_views = max(image_paths, key = lambda im_dict:im_dict["view_id"])["view_id"]
        self.num_scenes = max(image_paths, key = lambda im_dict:im_dict["scene_id"])["scene_id"]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx]['image_path'])
        scene = self.imgs[idx]['scene_index']
        view = self.imgs[idx]['view_index']
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, (scene,view)

"""
    DidacticBlenderDatasetPoseBatching extends the DidacticBlenderDataset and also provides the pose information for each
    batch of data. It selects and returns a second image along with it's pose for training the pose-conditional decoder.
"""
class DidacticBlenderDatasetPoseBatching(DidacticBlenderDataset):

    def __getitem__(self, idx):
        img1_idx = idx

        img1_scene = self.imgs[img1_idx]['scene_id']
        img1_pose  = self.imgs[img1_idx]['view_id']

        img2_pose = np.random.randint(0,self.num_views + 1)
        img2_idx  = self.scene_and_view_to_idx[(img1_scene, img2_pose)]

        img1 = default_loader(self.imgs[img1_idx]['image_path'])
        img2 = default_loader(self.imgs[img2_idx]['image_path'])

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1_pose, img1, img2_pose, img2

"""
    Utilizes the DidacticBlender datasets in order to create dataloaders for training, testing and validation. 
"""
class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 12,
        val_batch_size: int = 12,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        add_extra_training: int = 100,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([
             transforms.Resize(224),
             transforms.ToTensor(),
        ])
        
        val_transforms = transforms.Compose([
             transforms.Resize(224),
             transforms.ToTensor(),
        ])

        self.train_dataset = DidacticBlenderDatasetPoseBatching(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
            add_extra_training=100
        )

        self.train_metrics_dataset = DidacticBlenderDatasetPoseBatching(
            self.data_dir,
            split='train', 
            transform=train_transforms,
            download=False
        )

        self.val_dataset = DidacticBlenderDatasetPoseBatching(
            self.data_dir,
            split='val', 
            transform=val_transforms,
            download=False
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
    
    def train_metrics_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_metrics_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
