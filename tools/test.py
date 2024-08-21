import argparse
import os
import os.path as osp
import shutil
import tempfile
import time
import warnings
import tqdm
import mmcv
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.core import coco_eval, results2json
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from tools import visualizer_complete

import numpy as np

import hashlib


def draw_bboxes(image_path, bboxes, output_path, font_size=15, bbox_thickness=10, score_threshold=0.5):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/large/ttani_2/bhrl/lovehina_be_visualized/arial.ttf", font_size)

    for bbox in bboxes:
        x, y, w, h = bbox['bbox']
        category_id = bbox['category_id']
        score = bbox['score']

        if score >= score_threshold:
            # Assign different colors based on category_id
            color = get_category_color(category_id)

            draw.rectangle([x, y, x + w, y + h], outline=color, width=bbox_thickness)
            #draw.text((x, y), f"Label: {category_id}", font=font, fill=color)

    image.save(output_path)
"""
def get_category_color(category_id):
    # Define colors based on category_id
    color_mapping = {
    1: 'red',
    2: 'green',
    3: 'blue',
    4: 'orange',
    5: 'purple',
    6: 'yellow',
    7: 'cyan',
    8: 'pink',
    9: 'brown',
    10: 'grey',
    11: 'lightblue',
    12: 'lime',
    13: 'maroon',
    14: 'navy',
    15: 'olive',
    16: 'indigo',
    17: 'teal',
    18: 'violet',
    19: 'salmon',
    20: 'tan',
    21: 'khaki',
    22: 'orchid',
    23: 'peru',
    24: 'plum',
    25: 'gold',
    26: 'silver',
    27: 'crimson',
    28: 'tomato',
    29: 'magenta',
    30: 'thistle',
    31: 'azure',
    32: 'darkgreen',
    33: 'skyblue',
    34: 'limegreen',
    35: 'darkred',
    36: 'darkblue',
    37: 'hotpink',
    38: 'darkcyan',
    39: 'darkviolet',
    40: 'lightgreen',
    41: 'darkorange',
    42: 'coral',
    43: 'seagreen',
    44: 'mediumblue',
    45: 'springgreen',
    46: 'slategray',
    47: 'orchid',
    48: 'orangered',
    49: 'deeppink',
    50: 'lime',
    51: 'sandybrown',
    52: 'cadetblue',
    53: 'midnightblue',
    54: 'lightcoral',
    55: 'steelblue',
    56: 'rosybrown',
    57: 'red',
    58: 'darkslategray',
    59: 'dodgerblue',
    60: 'darkgoldenrod',
    61: 'firebrick',
    62: 'chocolate',
    63: 'indianred',
    64: 'mediumorchid',
    65: 'lightseagreen',
    66: 'mediumvioletred',
    67: 'olivedrab',
    68: 'darkorchid',
    69: 'goldenrod',
    70: 'lightskyblue',
    71: 'palegreen',
    72: 'royalblue',
    73: 'darkslateblue',
    74: 'greenyellow',
    75: 'mediumturquoise',
    76: 'cornflowerblue',
    77: 'darkkhaki',
    78: 'powderblue',
    79: 'sienna',
    80: 'mediumslateblue',
    81: 'darkturquoise',
    82: 'lightgray',
    83: 'burlywood',
    84: 'darkgray',
    85: 'mediumaquamarine',
    86: 'darkolivegreen',
    87: 'saddlebrown',
    88: 'darkmagenta',
    89: 'darkseagreen',
    90: 'darkslategray',
    91: 'mediumspringgreen',
    92: 'lightsteelblue',
    93: 'darkred',
    94: 'navajowhite',
    95: 'darkslateblue',
    96: 'mediumblue',
    97: 'darkslateblue',
    98: 'slateblue',
    99: 'peru',
    100: 'darkred',
    # Add more colors as needed
    }
    # Default to white if category_id not in color_mapping
    return color_mapping.get(category_id, 'red')
"""


def get_category_color(category_id):
    # Convert category_id to string and encode to bytes
    id_str = str(category_id).encode('utf-8')
    
    # Create a hash of the category_id
    hash_object = hashlib.md5(id_str)
    
    # Get the hex digest and take the first 6 characters for the color code
    color_code = hash_object.hexdigest()[:6]
    
    # Return the color code
    return f'#{color_code}'



def process_images(json_file_path, image_folder, output_folder,id_image_cor, font_size=15, bbox_thickness=2, score_threshold=0.5):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    #print(json_data)
    image_ids = set()
    for bbox_data in json_data:
        image_ids.add(bbox_data['image_id'])
    #print(image_ids)

    os.makedirs(output_folder, exist_ok=True)

    for image_id in image_ids:

        image_path = os.path.join(image_folder,id_image_cor[(image_id)])
        output_path = os.path.join(output_folder, f"Output_{str(image_id).zfill(3)}.jpg")

        bboxes = [bbox for bbox in json_data if bbox['image_id'] == image_id]

        draw_bboxes(image_path, bboxes, output_path, font_size=font_size, bbox_thickness=bbox_thickness, score_threshold=score_threshold)

import json

def create_id_path_mapping(json_data, id_key, path_key):
    id_path_mapping = {}
    for item in json_data:
        id_path_mapping[item[id_key]] = item[path_key]
    return id_path_mapping

def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    img_ids = []
    img_labels = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    #print(len(dataset))
    #print(dataset)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        #print(result)
        img_id = data['img_metas'][0].data[0][0]['img_info']['id']
        label = data['img_metas'][0].data[0][0]['label']
        img_ids.append(img_id)
        img_labels.append(label)
        #print(img_labels)

        if rank == 0:
            batch_size = data['img'][0][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # collect results from all ranks
    results, img_ids, img_labels = collect_results_id(results, len(dataset), img_ids, img_labels, tmpdir)

    return results, img_ids, img_labels

def collect_results_id(result_part, size, img_ids_part, img_labels_part, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    mmcv.dump(img_ids_part, osp.join(tmpdir, 'id_part_{}.pkl'.format(rank)))
    mmcv.dump(img_labels_part, osp.join(tmpdir, 'label_part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None, None, None
    else:
        # load results of all parts from tmp dir
        part_list = []
        id_part_list = []
        label_part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            id_part = osp.join(tmpdir, 'id_part_{}.pkl'.format(i))
            label_part = osp.join(tmpdir, 'label_part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
            id_part_list.append(mmcv.load(id_part))
            label_part_list.append(mmcv.load(label_part))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        ordered_ids = []
        for res in zip(*id_part_list):
            ordered_ids.extend(list(res))
        ordered_labels = []
        for res in zip(*label_part_list):
            ordered_labels.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        ordered_ids = ordered_ids[:size]
        ordered_labels = ordered_labels[:size]
        #print(ordered_ids)
        # remove tmp dir
        shutil.rmtree(tmpdir)
        #print(ordered_results)
        return ordered_results, ordered_ids, ordered_labels

def parse_args():
    parser = argparse.ArgumentParser(description='BHRL test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--average', type=int, default=1)
    parser.add_argument('--process_images', action='store_true', help='Process images')
    parser.add_argument('--test_seen_classes', action='store_true', help='test seen classes')
    parser.add_argument("--visualize_result")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]# this defines times of

    avg = args.average

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.test_seen_classes:
        cfg.data.test.test_seen_classes = True
    else:
        cfg.data.test.test_seen_classes = False
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    #print(avg)
    for i in range(avg):
        cfg.data.test.position = i 
        dataset = build_dataset(cfg.data.test) 
        #print(dataset)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False) 
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # --------------------------------
        model.CLASSES = dataset.CLASSES
        # --------------------------------
        
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs, img_ids, img_labels = multi_gpu_test(model, data_loader, args.tmpdir)
        #print(outputs)

        rank, _ = get_dist_info()
        if args.out and rank == 0:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
            eval_types = args.eval
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_files = results2json(dataset, outputs, img_ids, img_labels, args.out)
                        output_csv="/large/ttani_2/bhrl/haipara0_1.csv"
                        coco_eval(result_files, eval_types, dataset.coco, img_ids=img_ids, img_labels=img_labels,output_csv=output_csv)
    
    with open(cfg.data.test.ann_file, "r") as f:
        data = json.load(f)

    # JSONデータの読み込み

    # idとパスの組を作る
    if args.visualize_result:
        result_file_path = args.out + ".bbox.json"
        id_path_mapping = create_id_path_mapping(data["images"], "id", "file_name")
        id_image_cor_path = id_path_mapping
        image_folder = './data/manga_dataset/images/'
        output_folder = str(args.visualize_result)
        font_size = 64
        bbox_thickness = 15
        score_threshold = 0.50
        process_images(result_file_path, image_folder, output_folder,id_image_cor_path, font_size=font_size, bbox_thickness=bbox_thickness, score_threshold=score_threshold)


if __name__ == '__main__':
    main()