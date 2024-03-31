import os

import torch
import numpy as np

from skvideo import io
from PIL import Image

class Inferencer(object):
    def __init__(
        self,
        diffusion,
        model_path,
        output_path,
        num_samples = 10,
        img_size = 64,
        num_slice: int = 30
    ):
        super().__init__()  
        self.model_path = model_path
        self.diffusion = diffusion
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_slice = num_slice
        self.output_path = output_path


    def infer_new_data(self):
        data = torch.load(self.model_path)
        
        self.diffusion.load_state_dict(data['ema'])

        with torch.no_grad():
            
            if(self.diffusion.channels==1):
                for i in range(self.num_samples):
                    sampled_videos = self.diffusion.sample(batch_size = 1).cpu().detach().numpy()
                    out = np.zeros((1, self.diffusion.num_frames, self.img_size, self.img_size))
                    out[:, :, :, :] = sampled_videos[:, 0, :, :, :]
                    out = out.transpose((1, 2, 3, 0))

                    out *= 255.0/out.max() 
                    #if(not os.path.exists(os.path.join(self.output_path, "sample_"+str(i)))):
                    #    os.mkdir(os.path.join(self.output_path, "sample_"+str(i)))

                    out = (np.concatenate([out, out, out], axis=3))
                    
                    #for frame in range(self.num_slice):
                        #im = Image.fromarray(out[frame].astype(np.uint8))
                        #im.save(os.path.join(self.output_path, "sample_"+str(i),str(frame)+".png"))

                    io.vwrite(f'{self.output_path}/sample_{i}.gif', out)
                    
                    