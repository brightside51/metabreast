# Library Imports
import os
import random
import argparse
import numpy as np
import pydicom
import torch
import torchvision
import matplotlib.pyplot as plt

# Function Imports
from pathlib import Path
from ipywidgets import interactive, IntSlider
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ============================================================================================

# Non-Conditional MetaBrest Dataset Reader Class
class NCDataset(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        dataset: str = 'private',
        mode: str = 'train',
    ):  
        
        # Dataset Choice
        super(NCDataset).__init__(); self.settings = settings
        self.mode = mode; self.dataset = dataset
        if self.dataset == 'public': self.data_folderpath = self.settings.public_data_folderpath
        elif self.dataset == 'private': self.data_folderpath = self.settings.private_data_folderpath
        else: print("ERROR: Chosen Dataset / Directory does not exist!")
        
        # Subject Indexing (Existing or New Version)
        subj_listpath = Path(f"{self.settings.reader_folderpath}/V{self.settings.data_version}" +\
                             f"/{self.dataset}_{self.mode}_setV{self.settings.data_version}.txt")
        if subj_listpath.exists():
            print(f"Reading {self.dataset} Dataset Save Files for {self.mode} Set | Version {settings.data_version}")
            self.subj_list = subj_listpath.read_text().splitlines()
        else:
            print(f"Generating New Save Files for {self.dataset} Dataset | Version {settings.data_version}")
            self.subj_list = os.listdir(self.data_folderpath)       # Complete List of Subjects in Dataset
            self.subj_list.remove('video_data')
            self.subj_list = self.subj_split(self.subj_list)        # Selected List of Subjects in Dataset
        #assert len(self.subj_list) == self.num_subj, f"WARNING: Number of subjs does not match Dataset Version!"

        # --------------------------------------------------------------------------------------------
        
        # Dataset Transformations Initialization
        self.transform = transforms.Compose([
                                        transforms.Resize(( self.settings.img_size,
                                                            self.settings.img_size)),
                                        transforms.ToTensor()])
        self.h_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        self.v_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1)])

    # ============================================================================================

    # DataLoader Length / No. Subjects Computation Functionality
    def __len__(self): return len(self.subj_list)
    
    # Subject Splitting Functionality
    def subj_split(self, subj_list: list):

        # Dataset Splitting
        train_subj = len(subj_list) if self.settings.train_subj == 0 else self.settings.train_subj
        assert 0 < (train_subj + self.settings.val_subj + self.settings.test_subj) <= len(subj_list),\
               f"ERROR: Dataset does not contain {train_subj + self.settings.val_subj + self.settings.test_subj} Subjects!"
        train_subj = np.sort(np.array(random.sample(subj_list, train_subj), dtype = 'str'))
        subj_list = [subj for subj in subj_list if subj not in train_subj]                                      # Training Set Splitting
        if self.settings.train_subj != 0:
            val_subj = np.sort(np.array(random.sample(subj_list, self.settings.val_subj), dtype = 'str'))
            subj_list = [subj for subj in subj_list if subj not in val_subj]                                    # Validation Set Splitting
            test_subj = np.sort(np.array(random.sample(subj_list, self.settings.test_subj), dtype = 'str'))
            subj_list = [subj for subj in subj_list if subj not in test_subj]                                   # Test Set Splitting
            subj_list = np.sort(np.array(subj_list, dtype = 'str'))
        assert len(subj_list) + len(train_subj) + self.settings.val_subj + self.settings.test_subj == len(self.subj_list),\
               f"ERROR: Dataset Splitting went Wrong!"

        # Dataset Split Saving
        if not os.path.isdir(f"V{self.settings.data_version}"): os.mkdir(f"V{self.settings.data_version}")
        if len(train_subj) != 0: np.savetxt(f"V{self.settings.data_version}/{self.dataset}_train_setV{self.settings.data_version}.txt", train_subj, fmt='%s')
        if len(subj_list) != 0: np.savetxt(f"V{self.settings.data_version}/{self.dataset}_rest_setV{self.settings.data_version}.txt", subj_list, fmt='%s')
        if self.settings.train_subj != 0:
            if len(val_subj) != 0: np.savetxt(f"V{self.settings.data_version}/{self.dataset}_val_setV{self.settings.data_version}.txt", val_subj, fmt='%s')
            if len(test_subj) != 0: np.savetxt(f"V{self.settings.data_version}/{self.dataset}_test_setV{self.settings.data_version}.txt", test_subj, fmt='%s')
        
    # ============================================================================================
        
    # Single Batch / Subject Generation Functionality
    def __getitem__(self, idx: int = 0 or str, save: bool = False):
        subj_idx = idx if type(idx) == str else self.subj_list[idx]
        if self.settings.data_format == 'mp4':

            # Subject Data Access
            if self.settings.verbose: print(subj_idx)
            subj_folderpath = f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}/{subj_idx}.mp4"
            img_data = (torchvision.io.read_video(subj_folderpath, pts_unit = 'sec')[0][:, :, :, 0] / 255.0).type(torch.float32)

        elif self.settings.data_format == 'dicom':

            # Subject Folder Access
            subj_folderpath = f"{self.data_folderpath}/{subj_idx}"
            subj_filelist = os.listdir(subj_folderpath); i = 0
            while len(subj_filelist) > 0 and len(subj_filelist) <= 3:
                subj_folderpath = Path(f"{subj_folderpath}/{subj_filelist[0]}")
                subj_filelist = os.listdir(subj_folderpath)
            while os.path.splitext(f"{subj_folderpath}/{subj_filelist[i]}")[1] not in ['', '.dcm']: i += 1

            # Subject General Information Access
            #print(f"Accessing Subject {subj_idx}: {len(subj_filelist) - i} Slices")
            subj_filepath = Path((f"{subj_folderpath}/{subj_filelist[i]}"))
            subj_info = pydicom.dcmread(subj_filepath)
            subj_ori = subj_info[0x0020, 0x0037].value
            subj_v_flip = (np.all(subj_ori == [-1, 0, 0, 0, -1, 0]))
            subj_h_flip = (torch.rand(1) < (self.settings.h_flip / 100))
            #num_row = subj_info.Rows; num_col = subj_info.Columns
            #if self.dataset == 'private':
                #num_slice = int(subj_info[0x2001, 0x1018].value)
                #preg_status = int(subj_info[0x0010, 0x21c0].value)
            #else:
                #num_slice = int(subj_info[0x0020, 0x1002].value)
                #preg_status = None

            # --------------------------------------------------------------------------------------------
                
            # Subject Slice Data Access
            img_data = torch.empty((100, self.settings.img_size, self.settings.img_size)); slice_list = []
            for slice_filepath in subj_filelist:
                if os.path.splitext(slice_filepath)[1] in ['', '.dcm']:
                    
                    # Slice Data Access
                    slice_filepath = Path(f"{subj_folderpath}/{slice_filepath}")
                    slice_data = pydicom.dcmread(slice_filepath, force=True)
                    slice_idx = int(slice_data[0x0020, 0x0013].value) - 1
                    #print(slice_idx)
                    slice_list.append(slice_idx)
                    img_slice = slice_data.pixel_array.astype(float)

                    # Slice Image Pre-Processing | Rescaling, Resizing & Flipping
                    if self.settings.data_prep:
                        img_slice = np.uint8((np.maximum(img_slice, 0) / img_slice.max()) * 255)
                        img_slice = Image.fromarray(img_slice).resize(( self.settings.img_size,
                                                                        self.settings.img_size)) 
                        if subj_h_flip: img_slice = self.h_flip(img_slice)
                        if subj_v_flip: img_slice = self.v_flip(img_slice)
                        img_slice = np.array(self.transform(img_slice))
                    img_data[slice_idx, :, :] = torch.Tensor(img_slice); del img_slice
            img_data = img_data[np.sort(slice_list)]

            # --------------------------------------------------------------------------------------------
            
            # Correction for Chosen Number of Slices
            extra_slice = self.settings.num_slice - img_data.shape[0]
            if img_data.shape[0] < self.settings.num_slice:             # Addition of Repeated Peripheral Slices
                for extra in range(extra_slice):
                    if extra % 2 == 0: img_data = torch.cat((img_data, img_data[-1].unsqueeze(0)), dim = 0)
                    else: img_data = torch.cat((img_data[0].unsqueeze(0), img_data), dim = 0)
            elif img_data.shape[0] > self.settings.num_slice:           # Removal of Emptier Peripheral Slices
                img_data = img_data[int(np.ceil(-extra_slice / 2)) :\
                    int(len(img_data) - np.floor(-extra_slice / 2))]
            #else: assert(num_slice == self.settings.num_slice)
          
            # Item Dictionary Returning
            if save:
                print(f"Saving Patient Data for {subj_idx} into Video Format")
                torchvision.io.write_video(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}/{subj_idx}.mp4",
                    (img_data.unsqueeze(3).repeat(1, 1, 1, 3) * 255).type(torch.uint8), fps = self.settings.num_fps)
        else: raise(NotImplementedError)
        return img_data.unsqueeze(0)
    
        """return {'img_data': img_data,#.unsqueeze(0),
                'resolution': f'[{num_row}, {num_col}]',
                'subj_id': subj_id, 'num_slice': num_slice,
                'preg_status': preg_status, 'orientation': subj_ori,
                'h_flip': subj_h_flip, 'v_flip': subj_v_flip,
                'position': str(subj_info[0x0018, 0x5100].value),
                'num_series': int(subj_info.SeriesNumber)}"""
