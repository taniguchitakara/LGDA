import argparse
import os
import os.path as osp
import shutil
import tempfile
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.models import build_detector



from mmdet.apis import init_detector, inference_detector
import mmcv



config_file = '/large/ttani_2/bhrl/configs/manga/BHRL_manga.py'
checkpoint_file = '/large/ttani_2/bhrl/work_dirs/manga_colorized/BHRL/epoch_20.pth'




# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/large/ttani_2/bhrl/manga_colorized_tmp/JPEGImages/LoveHina_vol14_021.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='./result.jpg')



