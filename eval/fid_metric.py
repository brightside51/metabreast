import torch
from torcheval.metrics import FrechetInceptionDistance as FID
img1 = torch.randint(0, 120, (1, 3, 64, 64), dtype=torch.float32) / 200
img2 = torch.randint(70, 200, (1, 3, 64, 64), dtype=torch.float32) / 200
fid_metric = FID(feature_dim = 2048)
fid_metric.update(img1, is_real = True)
fid_metric.update(img2, is_real = False)
fid_metric.compute()
