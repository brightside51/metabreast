import torch
import copy
import wandb
import sys
import pdb

from torch.optim import AdamW
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch.nn.functional as F

from util.util import *
from infer_script import Inferencer
from pathlib import Path
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance as FID

sys.path.append('../../eval')
from ssim3d_metric import SSIM3D
from dice_metric import mean_dice_score


### Training class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        settings,
        amp: bool = False,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        ema_decay: float = 0.995,
        gradient_accumulate_every: int = 2,
        max_grad_norm = None
    ):  
        
        # Class Variable Logging
        super().__init__(); self.settings = settings
        self.model = diffusion_model.to(self.settings.device)
        self.ema = EMA(ema_decay); self.ds = dataset
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.gradient_accumulate_every = gradient_accumulate_every

        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'
        self.dl = data.DataLoader(  self.ds, batch_size = self.settings.batch_size,
                                    shuffle = self.settings.shuffle, pin_memory = True,
                                    prefetch_factor = self.settings.prefetch_factor,
                                    num_workers = self.settings.num_workers)
                                    #pin_memory_device = self.settings.device)
        self.num_batch = len(self.dl); self.dl = cycle(self.dl)
        print(f'training using {len(self.ds)} cases in {self.num_batch} batches')
        self.opt = AdamW(diffusion_model.parameters(), lr = self.settings.lr_base)
        self.lr_scheduler = ExponentialLR(self.opt, gamma = self.settings.lr_decay)
        
        self.step = 0; self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        # Evaluation Metric & Saving Directories
        self.results_folder = Path(f"{self.settings.logs_folderpath}/V{self.settings.model_version}")
        self.results_folder.mkdir(exist_ok = True, parents = True); self.reset_parameters()
        self.fid_metric = FID(feature = 64)
        if self.settings.log_method == 'tensorboard':
            self.train_logger = TensorBoardLogger(self.results_folder, 'train')
            self.eval_logger = TensorBoardLogger(self.results_folder, 'eval')
        self.eval_writer = SummaryWriter(log_dir = f"{self.results_folder}/eval")
        self.ssim_metric = SSIM3D(window_size = 3).to(self.settings.device)
        if self.settings.verbose: print("training script initialized")

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, run):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'fid_metric': self.fid_metric}
        torch.save(data, str(self.results_folder / f'model-{run}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])
        self.fid_metrid = data['fid_metric']

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop,
        run = 'example_run'
    ):
        assert callable(log_fn)

        while self.step < self.settings.num_steps:

            pdb.set_trace()
            for i in range(self.gradient_accumulate_every):

                if self.settings.verbose: print('reading data samples / batches')
                data = next(self.dl).to(self.settings.device)

                if self.step < self.num_batch:
                    for slice in range(data.shape[2]):
                        self.fid_metric.update(data[:, 0, slice].unsqueeze(1).repeat(1, 3, 1, 1).type(torch.ByteTensor), real = True)

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss["MSE Loss"] / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss["MSE Loss"].item()}')

            if self.step == 0 or self.step % self.settings.log_interval == 0:
                if self.settings.verbose: print("logging training metrics")
                if self.settings.log_method == 'wandb': wandb.log(loss)
                elif self.settings.log_method == 'tensorboard':
                    
                    milestone = self.step // self.settings.log_interval
                    self.train_logger.experiment.add_scalar("Learning Rate", self.lr_scheduler.get_last_lr()[0], milestone)
                    self.train_logger.experiment.add_scalar("L1 Loss", loss["L1 Loss"].item(), milestone)
                    self.train_logger.experiment.add_scalar("MSE Loss", loss["MSE Loss"].item(), milestone)
                    self.train_logger.experiment.add_scalar("Dice Score", loss["Dice Score"], milestone)
                    self.train_logger.experiment.add_scalar("SSIM Index", loss["SSIM Index"].item(), milestone)
                    self.train_logger.experiment.add_scalar("PSNR Loss", loss["PSNR Loss"].item(), milestone)
                    self.train_logger.experiment.add_scalar("NMI Loss", loss["NMI Loss"].item(), milestone)
                log = {'loss': loss["MSE Loss"].item()}

            if exists(self.max_grad_norm):
                if self.settings.verbose: print("clipping normalized gradient")
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.settings.verbose: print("backpropagating gradient descent")
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                if self.settings.verbose: print("updating ema model")
                self.step_ema()

            if self.step != 0 and self.step % self.settings.save_interval == 0:
                if self.settings.verbose: print('saving model')
                self.save(run)
                if self.settings.verbose: print(f"evaluating model's performance")
                milestone = self.step // self.settings.save_interval
                num_samples = self.settings.save_img ** 2
                batches = num_to_groups(num_samples, self.settings.batch_size)
                
                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = F.pad(torch.cat(all_videos_list, dim = 0), (2, 2, 2, 2))
                print(all_videos_list.shape)
            
                #one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.settings.save_img)
                #video_path = str(self.results_folder / str(f'{milestone}.gif'))
                #video_tensor_to_gif(one_gif, video_path)
                #log = {**log, 'sample': video_path}
                #self.save(milestone)

                for slice in range(all_videos_list.shape[2]):
                    self.fid_metric.update(all_videos_list[:, 0, slice].unsqueeze(1).repeat(1, 3, 1, 1).type(torch.ByteTensor), real = False)

                if self.settings.verbose: print('logging evaluation metrics')
                if self.settings.log_method == 'wandb':
                    wandb.log({"val/FID Score": self.fid_metric.compute()})
                    wandb.log({"val/SSIM Index": self.ssim_metric(data[0], all_videos_list[0])})
                    wandb.log({"val/Dice Score": mean_dice_score(data[0].detach().cpu(), all_videos_list[0].detach().cpu())})
                    wandb.log({"val/Inference Samples": wandb.Video(all_videos_list.swapaxes(1, 2).repeat(1, 1, 3, 1, 1), fps = self.settings.num_fps)})
                elif self.settings.log_method == 'tensorboard':
                    self.eval_logger.experiment.add_scalar("val/FID Score", self.fid_metric.compute(), milestone)
                    self.eval_logger.experiment.add_scalar("val/SSIM Index", self.ssim_metric(data[0], all_videos_list[0]), milestone)
                    self.eval_logger.experiment.add_scalar("val/Dice Score", mean_dice_score(data[0].detach().cpu(),
                                                                                all_videos_list[0].detach().cpu()), milestone)
                    self.eval_writer.add_video('val/Inference Samples', all_videos_list.swapaxes(1, 2).repeat(1, 1, 3, 1, 1),
                                                                        global_step = milestone, fps = self.settings.num_fps, walltime = None)
                    
                # Model Inference Mode
                infer = Inferencer( self.diffusion_model, num_samples = 20, img_size = self.settings.img_size, num_slice = self.settings.num_slice,
                                    model_path = Path(f"{self.settings.logs_folderpath}/V{self.settings.model_version}/model-save_V{self.settings.model_version}.pt"),
                                    output_path = Path(f"{self.settings.logs_folderpath}/V{self.settings.model_version}/gen_img"))
                infer.infer_new_data()
                
            log_fn(log); self.step += 1; self.lr_scheduler.step()

        self.save(run)
        print('training completed')



    '''-------------------------Initialize Losses-------------------------'''


    '''-------------------------Initialize Optimizers---------------------'''


    '''-------------------------Initialize Schedulers---------------------'''


    '''----------------------Move everything to DEVICE--------------------'''