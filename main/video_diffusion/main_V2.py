import os
import sys
import argparse
import numpy as np
import torch
import time

sys.path.append('../../data/non_cond')
from nc_data_reader import NCDataset
sys.path.append('../../models/video_diffusion')
from Unet3D import Unet3D
from GaussianDiffusion import GaussianDiffusion
from util.util import abrev
sys.path.append('../../scripts/video_diffusion')
from infer_script import Inferencer
from train_script import Trainer

from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary
from tensorboardX import SummaryWriter
#from data.dataset import get_training_set
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

'''-------------------------GPU presence/absence-------------------------'''

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_type', type = str,            # Chosen Model / Diffusion
                                choices =  {'video_diffusion',
                                            'blackout_diffusion'},
                                default = 'video_diffusion')
    ncdiff_parser.add_argument('--model_version', type = int,         # Model Version Index
                                default = 2)
    ncdiff_parser.add_argument('--data_version', type = int,          # Dataset Version Index
                                default = 1)
    settings = ncdiff_parser.parse_args("")

    # ============================================================================================

    # Directories and Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,         # Path for Dataset Reader Directory
                                default = '../../data/non_cond')
    ncdiff_parser.add_argument('--public_data_folderpath', type = str,    # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
                                default = "../../../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,   # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/METABREST/T1W_Breast")
                                default = '../../../../../datasets/private/METABREST/T1W_Breast')

    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,          # Path for Model Architecture Directory
                                default = f'../../models/{settings.model_type}')
    ncdiff_parser.add_argument('--script_folderpath', type = str,         # Path for Model Training & Testing Scripts Directory
                                default = f'../../scripts/{settings.model_type}')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,           # Path for Model Saving Directory
                                default = f'../../logs/{settings.model_type}')
    ncdiff_parser.add_argument('--verbose', type = bool,                  # Verbose Control Variable
                                default = False)
        
    # ============================================================================================

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--data_format', type = str,           # Chosen Dataset Format for Reading
                                choices =  {'mp4', 'dicom'},
                                default = 'mp4')
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 64)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 30)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 50)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 0)                          # PS: Input 0 for all Subjects in the Dataset
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 0)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 1)
    ncdiff_parser.add_argument('--num_fps', type = int,               # Number of Video Frames per Second
                                default = 4)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = True)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 8)
    ncdiff_parser.add_argument('--prefetch_factor', type = int,       # Number of Prefetched DataLoader Batches per Worker
                                default = 1)

    # ============================================================================================

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
    ncdiff_parser.add_argument('--noise_type', type = str,            # Diffusion Noise Distribution
                                default = 'gaussian')
    #ncdiff_parser.add_argument('--num_epochs', type = int,           # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 500)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 150000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-4)
    ncdiff_parser.add_argument('--lr_decay', type = float,            # Learning Rate Decay Value
                                default = 0.999)
    ncdiff_parser.add_argument('--lr_step', type = float,             # Number of Steps inbetween Learning Rate Decay
                                default = 250)
    ncdiff_parser.add_argument('--lr_min', type = float,              # Minimum Decayed Learning Rate Value
                                default = 1e-6)
    
    # Model | Result Logging Arguments 
    ncdiff_parser.add_argument('--save_interval', type = int,         # Number of Training Step Interval inbetween Image Saving
                                default = 500)
    #ncdiff_parser.add_argument('--log_interval', type = int,          # Number of Training Step Interval inbetween Result Logging (not a joke i swear...)
    #                           default = 1)
    ncdiff_parser.add_argument('--save_img', type = int,              # Square Root of Number of Images Saved for Manual Evaluation
                                default = 2)
    ncdiff_parser.add_argument('--log_method', type = str,            # Metric Logging Methodology
                                choices = {'wandb', 'tensorboard', None},
                                default = 'tensorboard')

    # ============================================================================================

    settings = ncdiff_parser.parse_args("")
    settings.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

'''-------------------------Dataset-parameters-------------------------'''
goDo = 'train' # train, infer, showData

if goDo == 'train':

    # Dataset Access
    print('PID:' + str(os.getpid()))
    private_dataset = NCDataset(settings,
                                mode = 'train',
                                dataset = 'private')
    public_dataset = NCDataset( settings,
                                mode = 'train',
                                dataset = 'public')
    dataset = ConcatDataset([private_dataset, public_dataset])

    # Model and Diffusion Initialization
    model = Unet3D(
        dim = settings.dim,
        channels = settings.num_channel,
        dim_mults = settings.mult_dim)

    diffusion = GaussianDiffusion(
        model, image_size = settings.img_size,
        num_frames = settings.num_slice,
        channels = settings.num_channel,
        noise_type = settings.noise_type,
        timesteps = settings.num_ts).to(settings.device)
    
    # Training Script Initialization
    trainer = Trainer(
        diffusion, dataset, settings,
        train_batch_size = settings.batch_size,
        train_lr = settings.lr_base,
        save_and_sample_every = settings.save_interval,
        train_num_steps = settings.num_steps,
        gradient_accumulate_every = 2,
        ema_decay = 0.995, amp = True,
        results_folder = os.path.join(settings.logs_folderpath, f"V{settings.model_version}"))

    start = time.time()
    trainer.train(run = f"V{settings.model_version}")
    end = time.time()

    print(f"Training time: {(end - start)/60}")

elif goDo == 'infer':
    print("Going to infer new data...")

    model = Unet3D(
        dim = 64,
        channels = 1,
        dim_mults = (1, 2, 4, 8),
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = img_size,
        num_frames = num_frames,
        channels = 1,
        timesteps = 500,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).cuda()

    inferencer = Inferencer(
        diffusion,
        model_path = os.path.join(modelSaveDir,"model-S128_30_100K.pt"),
        output_path = os.path.join(os.getcwd(),"synth_samples_128"),
        num_samples = 20,
        img_size = img_size)
    
    inferencer.infer_new_data()

    print("Finished infering new data")

elif goDo == 'showData':
    print("Going to show data...")

    public_set = get_training_set(publicSetDir, img_size=64, num_frames=num_frames)
    private_set = get_training_set(privateSetDir, img_size=64, num_frames=num_frames)
  
    concat_set = ConcatDataset([public_set, private_set])

    print("Private Dataset length", private_set.__len__()) # 89
    print("Public Dataset length", public_set.__len__())   # 768
    print("Total Dataset length", concat_set.__len__())    # 857

    # for i in range(30):
    #     mri_example = train_set.__getitem__(i)
    #     print(mri_example.size())
    
    for i in range(0,900):
        mri_example = concat_set.__getitem__(i)
        print(i)
        # for f in range(num_frames):
        #     plt.title("Example Image") 
        #     image = mri_example[0][f].cpu().detach().numpy()
        #     plt.imshow(image)
        #     plt.savefig(os.path.join(os.getcwd(),'check_data','frame'+str(f)+'.png'))

