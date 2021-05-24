import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from . import utils
from pathlib import Path
from sklearn.model_selection import train_test_split

class DroneSegmentation(Dataset):
    def __init__(self, root, val_size=0.1, mode='train', transform=None, 
                 image_transform=None, mask_transform=None):
        self.root = Path(root)
        self.val_size = val_size
        self.mode = mode
        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        self._load_files()
        self._create_dataframe()
        self._split_dataframe()
    
    def _load_files(self):
        self.image_path = self.root.joinpath('images')
        self.mask_path = self.root.joinpath('labels/tiff')
        
        self.image_files = sorted(list(self.image_path.glob('*.jpg')))
        self.mask_files = sorted(list(self.mask_path.glob('*.tiff')))
        
    @property    
    def idx2class(self):
        return utils.idx2class()
    
    @property
    def class2idx(self):
        return utils.class2idx()
        
    def _create_dataframe(self):
        names, imgs_path, msks_path = [],[],[]
        for idx in range(len(self.image_files)):
            name = self.image_files[idx].stem
            img_path = str(self.image_files[idx])
            msk_path = str(self.mask_files[idx])
            names.append(name)
            imgs_path.append(img_path)
            msks_path.append(msk_path)

        data_dict = {
            'id': names,
            'image_path': imgs_path,
            'mask_path': msks_path
        }
        self.dataframe = pd.DataFrame(data_dict)
    
    def _split_dataframe(self):
        self.trainset, self.validset = train_test_split(self.dataframe, test_size=self.val_size, random_state=19)
        if self.mode == 'train':
            self.files = self.trainset
        else:
            self.files = self.validset
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        impath = self.files['image_path'].iloc[idx]
        mspath = self.files['mask_path'].iloc[idx]
        
        orig_img = utils.load_image(impath, to_np=True)
        mask_img = utils.load_tiff(mspath)
        mask_img = mask_img.astype(np.uint8)
        
        if self.transform:
            orig_img, mask_img = self.transform(orig_img, mask_img)
            
        if self.image_transform:
            orig_img = self.image_transform(orig_img)
        
        if self.mask_transform:
            mask_img = self.mask_transform(mask_img)   
        
        return orig_img, mask_img
