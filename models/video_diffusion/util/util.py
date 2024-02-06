from __future__ import print_function

import os

from torchvision import transforms as T

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed Due to some other error.\n" % path)
        else:
            print ("Successfully created the directory %s \n" % path)


def abrev(txt):
    txt = txt.replace("pix", "p")
    txt = txt.replace("_prj", "")
    return txt

'''------ Trainer Utils --------'''
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
def cycle(dl):
    while True:
        for data in dl:
            yield data

def noop(*args, **kwargs):
    pass

def exists(x):
    return x is not None

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images
