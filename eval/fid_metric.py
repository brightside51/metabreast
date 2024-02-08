import torch
from torchmetrics.image.fid import FrechetInceptionDistance
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
metric = FrechetInceptionDistance(feature=64)
metric.update(imgs_dist1, real=True)
metric.update(imgs_dist2, real=False)
print(metric.compute())