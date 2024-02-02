
# Package Imports
import os
import sys
import pathlib
import argparse
import numpy as np
import torch
from torchinfo import summary
import matplotlib.pyplot as plt
import time

# Functionality Imports
from pathlib import Path
from torch.utils.data import ConcatDataset

# ============================================================================================

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_version', type = int,               # Model Version Index
                                default = 0)
    ncdiff_parser.add_argument('--data_version', type = int,                # Dataset Version Index
                                default = 0)

    # Directory | Dataset-Related Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,      # Path for Dataset Reader Directory
                                default = 'data')
    ncdiff_parser.add_argument('--public_data_folderpath', type = str,      # Path for Private Dataset Directory
                                default = '../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1')
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,     # Path for Private Dataset Directory
                                default = '../../../datasets/private/METABREST/T1W_Breast')
    
    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,            # Path for Model Architecture Directory
                                default = 'models/video_diffusion')
    ncdiff_parser.add_argument('--script_folderpath', type = str,           # Path for Model Training & Testing Scripts Directory
                                default = 'scripts')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,             # Path for Model Saving Directory
                                default = 'logs')
    
    
    # --------------------------------------------------------------------------------------------

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 64)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 10)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 50)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 20)
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 0)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 5)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = False)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 12)

    # --------------------------------------------------------------------------------------------

    # Model | Architecture-Defining Arguments
    ncdiff_parser.add_argument('--seed', type = int,                  # Randomised Generational Seed
                                default = 0)
    ncdiff_parser.add_argument('--dim', type = int,                   # Input Dimensionality (Not Necessary)
                                default = 64)
    ncdiff_parser.add_argument('--num_channel', type = int,           # Number of Input Channels for Dataset
                                default = 1)
    ncdiff_parser.add_argument('--mult_dim', type = tuple,            # Dimensionality for all Conditional Layers
                                default = (1, 2, 4, 8))

    # Model | Training & Diffusion Arguments
    #ncdiff_parser.add_argument('--num_epochs', type = int,            # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 300)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 10000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-3)
    ncdiff_parser.add_argument('--num_block', type = int,             # Number of Layers per UNet Block
                                default = 2)

    settings = ncdiff_parser.parse_args("")
    settings.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------------

# Functionality Imports
sys.path.append(settings.reader_folderpath)
from nc_data_reader import NCDataset
sys.path.append(settings.model_folderpath)
from Unet3D import Unet3D
from GaussianDiffusion import GaussianDiffusion
sys.path.append(settings.script_folderpath)
from video_diffusion_infer import Inferencer
from video_diffusion_train import Trainer

# ============================================================================================
# ====================================== Training Setup ======================================
# ============================================================================================

# Dataset Access
private_dataset = NCDataset(settings,
                            mode = 'train',
                            dataset = 'private')
#public_dataset = NCDataset( settings,
#                            mode = 'train',
#                            dataset = 'public')
#dataset = ConcatDataset([private_dataset, public_dataset])

# --------------------------------------------------------------------------------------------

# Model & Diffusion Process Initialization
model = Unet3D(             dim = settings.dim,
                            channels = settings.num_channel,
                            dim_mults = settings.mult_dim)
diff = GaussianDiffusion(   model,
                            image_size = settings.img_size,
                            num_frames = settings.num_slice,
                            channels = settings.num_channel,
                            timesteps = settings.num_ts,
                            loss_type = 'l1')

# Diffusion Application
"""
diff_summary = summary(     diff,
                        (1, settings.num_channel,
                            settings.num_slice,
                            settings.img_size,
                            settings.img_size))

# Model Trainer Initialization
trainer = Trainer(  diff, private_dataset,
                    train_batch_size = settings.batch_size,
                    train_lr = settings.lr_base,
                    save_and_sample_every = 500,
                    train_num_steps = settings.num_steps,
                    gradient_accumulate_every = 2,
                    ema_decay = 0.995, amp = True,
                    results_folder = f"{settings.logs_folderpath}/V{settings.model_version}")

# Model Trainer Application
time_start = time.time()
trainer.train(run = f'save_V{settings.model_version}')
time_end = time.time()
print(f"Time Duration: {(time_end - time_start) / 60}")
"""

# ============================================================================================
# ====================================== Inference Setup =====================================
# ============================================================================================

# Model Inference Mode
infer = Inferencer( diff,
                    model_path = Path(f"{settings.logs_folderpath}/V{settings.model_version}/model-save_V{settings.model_version}.pt"),
                    output_path = Path(f"{settings.logs_folderpath}/V{settings.model_version}/gen_img"),
                    num_samples = 20, img_size = settings.img_size, num_slice = settings.num_slice)
infer.infer_new_data()

# Running Commands
#srun -p debug_8GB -q debug_8GB python video_diffusion_main.py