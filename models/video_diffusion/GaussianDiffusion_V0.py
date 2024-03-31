'''GaussianDiffusion model based on https://github.com/lucidrains/video-diffusion-pytorch/'''

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.unet3d_utils import *
from einops import rearrange
#from tqdm import tqdm

from einops_exts import check_shape
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import normalized_mutual_information as NMI
sys.path.append('../../eval')
from ssim3d_metric import SSIM3D
from dice_metric import mean_dice_score

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        settings,
        *,
        text_use_bert_cls = False,
        #timesteps = 1000,
        use_dynamic_thres = False,
        dynamic_thres_percentile = 0.9
    ):
        super().__init__(); self.settings = settings
        #self.channels = self.settings.num_channel
        #self.image_size = self.settings.img_size
        #self.num_frames = self.settings.num_slice

        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(self.settings.num_ts)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        
        self.ssim_metric = SSIM3D(window_size = 3).to(self.settings.device)
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        if self.settings.noise_type == 'gaussian': noise = torch.randn_like(x)
        elif self.settings.noise_type == 'poisson': noise = torch.from_numpy(np.random.poisson(size = x.size))
        elif self.settings.noise_type == 'gamma': noise = torch.from_numpy(np.random.gamma(x.shape))
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        #for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        device = next(self.denoise_fn.parameters()).device

        #if is_list_str(cond):
            #cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        #image_size = self.image_size
        #channels = self.channels
        #num_frames = self.num_frames
        return self.p_sample_loop((batch_size, self.settings.num_channel,
            self.settings.num_slice, self.settings.img_size,
            self.settings.img_size), cond = cond, cond_scale = cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        #for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start)).to(device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
        #     cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        # Metric Computation
        norm_noise = noise[0] - noise[0].min(1, keepdim = True)[0]
        norm_noise /= norm_noise.max(1, keepdim = True)[0]
        norm_recon = x_recon[0] - x_recon[0].min(1, keepdim = True)[0]
        norm_recon /= norm_recon.max(1, keepdim = True)[0].to(dtype = torch.float32)
        return {"L1 Loss": F.l1_loss(noise, x_recon),
                "MSE Loss": F.mse_loss(noise, x_recon),
                "Dice Score": mean_dice_score(  noise.detach().cpu(),
                                                x_recon.detach().cpu()),
                "SSIM Index": self.ssim_metric( noise, x_recon),
                "PSNR Loss": PSNR(  norm_noise.detach().cpu().numpy(),
                                    norm_recon.detach().cpu().numpy()),
                "NMI Loss": NMI(    norm_noise.detach().cpu().numpy(),
                                    norm_recon.detach().cpu().numpy())}

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.settings.img_size
        check_shape(x, 'b c f h w', c = self.settings.num_channel,
            f = self.settings.num_slice, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)