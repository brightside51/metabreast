import torch
import copy
import pdb
import sys

from torch.optim import Adam
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch.nn.functional as F

from util.util import *
from pathlib import Path
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

sys.path.append('../../eval')
from ssim3d_metric import SSIM3D
from dice_metric import mean_dice_score

### Training class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        settings,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 0,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None
    ):
        super().__init__()
        self.settings = settings
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model).to(self.settings.device)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = dataset

        #print(f'training using {len(self.ds)} cases')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, pin_memory = True,
                                        batch_size = train_batch_size,
                                        shuffle = self.settings.shuffle,
                                        prefetch_factor = self.settings.prefetch_factor,
                                        num_workers = self.settings.num_workers))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)
        self.lr_scheduler = ExponentialLR(self.opt, gamma = self.settings.lr_decay)
        
        self.step = 0; self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)
        self.train_logger = TensorBoardLogger(self.results_folder, 'train')
        #self.eval_logger = TensorBoardLogger(self.results_folder, 'eval')
        self.eval_writer = SummaryWriter(log_dir = f"{self.results_folder}/eval")
        self.ssim_metric = SSIM3D(window_size = 3).to(self.settings.device)
        #self.fid_metric = FID(feature = 64)

        self.reset_parameters()

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
            'scaler': self.scaler.state_dict()}
            #'fid_metric': self.fid_metric}
        torch.save(data, str(self.results_folder / f'model.pt'))

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
        #self.fid_metrid = data['fid_metric']

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop,
        run = 'example_run'
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:

            #pdb.set_trace()
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()

                #if self.step < (len(self.ds) * self.gradient_accumulate_every) / self.batch_size:
                #    for slice in range(data.shape[2]):
                #        self.fid_metric.update(data[:, 0, slice].unsqueeze(1).repeat(1, 3, 1, 1).type(torch.ByteTensor), real = True)

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                
            self.train_logger.experiment.add_scalar("Learning Rate", self.lr_scheduler.get_last_lr()[0], self.step)
            #self.train_logger.experiment.add_scalar("L1 Loss", loss["L1 Loss"].item(), self.step)
            self.train_logger.experiment.add_scalar("MSE Loss", loss.item(), self.step)
            #self.train_logger.experiment.add_scalar("Dice Score", loss["Dice Score"], self.step)
            #self.train_logger.experiment.add_scalar("SSIM Index", loss["SSIM Index"].item(), self.step)
            #self.train_logger.experiment.add_scalar("PSNR Loss", loss["PSNR Loss"].item(), self.step)
            #self.train_logger.experiment.add_scalar("NMI Loss", loss["NMI Loss"].item(), self.step)
            log = {'loss': loss.item()}; print(f"Step #{self.step}: {loss.item()}")

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                print("updating ema model"); self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                #for slice in range(all_videos_list.shape[2]):
                #    self.fid_metric.update(all_videos_list[:, 0, slice].unsqueeze(1).repeat(1, 3, 1, 1).type(torch.ByteTensor), real = False)

                #self.eval_logger.experiment.add_scalar("FID Score", self.fid_metric.compute(), milestone)
                #self.eval_logger.experiment.add_scalar("SSIM Index", self.ssim_metric(data[0].unsqueeze(0),
                #                                                all_videos_list[0].unsqueeze(0)), milestone)
                #self.eval_logger.experiment.add_scalar("Dice Score", mean_dice_score(data[0].unsqueeze(0).detach().cpu(),
                #                                            all_videos_list[0].unsqueeze(0).detach().cpu()), milestone)
                
                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
                print(all_videos_list.shape)
                #self.eval_writer.add_video('Inference Samples', all_videos_list.swapaxes(1, 2).repeat(1, 1, 3, 1, 1),
                #                                                    global_step = milestone, fps = self.settings.num_fps, walltime = None)
            
                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'gen_img/sample_{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1
            if self.step % self.settings.lr_step == 0 and self.lr_scheduler.get_last_lr()[0] > self.settings.lr_min: self.lr_scheduler.step()

        self.save(run)
        print('training completed')
