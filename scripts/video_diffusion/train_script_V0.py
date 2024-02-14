import torch
import copy

from torch.optim import Adam
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch.nn.functional as F

from util.util import *
from pathlib import Path
from einops import rearrange


### Training class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
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
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
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

        print(f'training using {len(self.ds)} cases')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()
        self.train_logger = TensorBoardLogger(results_folder, 'train')

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
            'scaler': self.scaler.state_dict()
        }
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

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop,
        run = 'example_run'
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                #print(f'{self.step}: {loss.item()}')
                self.train_logger.experiment.add_scalar("Step Loss", loss.item(), self.step)

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            if self.step % 1000 == 0: self.save(run)
            print('saving model')

            #if self.step != 0 and self.step % self.save_and_sample_every == 0:
            #    milestone = self.step // self.save_and_sample_every
            #    num_samples = self.num_sample_rows ** 2
            #    batches = num_to_groups(num_samples, self.batch_size)
            #
            #    all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
            #    all_videos_list = torch.cat(all_videos_list, dim = 0)
            #
            #    all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
            #
            #    one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
            #    video_path = str(self.results_folder / str(f'{milestone}.gif'))
            #    video_tensor_to_gif(one_gif, video_path)
            #    log = {**log, 'sample': video_path}
            #    self.save(milestone)

            log_fn(log)
            self.step += 1

        self.save(run)
        print('training completed')



    '''-------------------------Initialize Losses-------------------------'''


    '''-------------------------Initialize Optimizers---------------------'''


    '''-------------------------Initialize Schedulers---------------------'''


    '''----------------------Move everything to DEVICE--------------------'''