# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

############# 
# New imports
############# 
import nibabel as nib
# sys.path.append("..")
from inflated_convnets_pytorch.src import i3res
import torchvision
import copy 

''''
Command: 
python main_dino.py --arch i3dresnet152 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --local_crops_number 4 --data_path /home/malteekj/3D_dino/CT_data/CT-ORG_numpy_3mm --output_dir /home/malteekj/3D_dino/dino_output_3D_2 --num_workers 30 --batch_size=32
'''


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
# add 3D models
available_3d_models = ['i3dresnet152'] 
torchvision_archs = torchvision_archs + available_3d_models

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    '''
    Temporary hack for logging metrics 
    '''
    with open('/home/malteekj/3D_dino/dino_output/dino_loss.txt', 'w') as f:
        pass

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO_CT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    # transform = DataAugmentationDINO(
    #     args.global_crops_scale,
    #     args.local_crops_scale,
    #     args.local_crops_number,
    # )
    # dataset = datasets.ImageFolder(args.data_path, transform=transform)
    if args.arch in available_3d_models:
        transform = DataAugmentationDINO_CT3D(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        )
        dataset = NumpyDataset3D(args.data_path, transform=transform)
    else:
        transform = DataAugmentationDINO_CT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
        dataset = NiftiDataset(args.data_path, transform=transform)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT 
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
        
    elif args.arch in available_3d_models:
        resnet = torchvision.models.resnet152(pretrained=True)
        # breakpoint()
        
        student = i3res.I3ResNet(copy.deepcopy(resnet), class_nb=2048, conv_class=True)
        teacher = i3res.I3ResNet(copy.deepcopy(resnet), class_nb=2048, conv_class=True)
        
        # student = ResNet152Checkpointed(student)
        # teacher = ResNet152Checkpointed(teacher)
        # breakpoint()
        
        embed_dim = 2048
        # model = model.cuda()
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
        
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        
        print(it)
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
        
        if it%10 == 0:
            with open('/home/malteekj/3D_dino/dino_output/dino_loss.txt', 'a') as f:
                f.write(str(float(loss.cpu().detach())) + '\n')

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


#####################################################################################
# New functions for handling CT scans
#####################################################################################
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import random

###############
# 3D functions
###############
class NumpyDataset3D(Dataset):
    '''
    Loads CT dataset in .npy format and samples cubes in 3D
    '''
    def __init__(self, directory, transform=None):
        super(NumpyDataset3D, self).__init__()
        
        self.transform = transform
        
        self.files = []
        # Recursively find all NIfTI files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy') and 'volume' in file:
                    self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)*100

    def __getitem__(self, idx):
        # Load the NIfTI file
        
        # hack to make the epochs longer for small datasets
        img_path = self.files[idx % len(self.files)]
        
        # nifti_img = np.load(nifti_path, mmap_mode='r')
        img = np.load(img_path)
        
          # Convert the data to a PyTorch tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)  
        # to make the resnet accept it 
        img = torch.repeat_interleave(img, 3, dim=0)
        
        if self.transform is not None:
            return self.transform(img), torch.tensor(0)
        else:
            img, torch.tensor(0)

class DataAugmentationDINO_CT3D(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        '''
        Create the augmentations and crops for each image. The crops a determined by
        the percentage (scale) of the image, and then they're resampled into the crops size

        Parameters
        ----------
        global_crops_scale :  (tuple of python:float) 
             Specifies the lower and upper bounds for the random area of the crop, 
             before resizing. The scale is defined with respect to the area of the original image.
        local_crops_scale :  (tuple of python:float) 
            Specifies the lower and upper bounds for the random area of the crop, 
             before resizing. The scale is defined with respect to the area of the original image.
        local_crops_number : int
            Number of local random crops (two global is default)

        Returns
        -------
        None.

        '''
        
        flip_and_color_jitter = transforms.Compose([
            self.flip3D(p=[0, 0, 0.0]),
            
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            #     p=0.8
            # ),
            # transforms.RandomGrayscale(p=0.2),
        ])
        
        # normalize = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        # normalize = lambda x: x
        normalize = self.normalize(min_val=-400, max_val=1000, a=0, b=1)

        # first global crop
        self.global_transfo1 = transforms.Compose([
            # self.RandomResizedCrop3D(target_pixel_size=224, scale=global_crops_scale, ratio=[3/4, 4/3]),
            randomCrop3D(crop_size=96),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            # self.RandomResizedCrop3D(target_pixel_size=224, scale=global_crops_scale, ratio=[3/4, 4/3]),
            randomCrop3D(crop_size=96),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            # utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            # self.RandomResizedCrop3D(target_pixel_size=96, scale=global_crops_scale, ratio=[3/4, 4/3]),
            randomCrop3D(crop_size=42),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        
        # image[400 < image] = 400
        # image[-400 > image] = -400
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
    def normalize(self, min_val=None, max_val=None, a=None, b=None):
        '''
        x is the data to be normalized. If min and max is None, the min and max from the data is used. If a and b is None, else 
        the data is normalized between 0 and 1
        '''
        def _normalize(x):
            
            nonlocal a, b # so they can be changed
            # 0 to 1 is default
            if all([a is None, b is None]):
                a = 0
                b = 1    
            # check whether a max and min has been given
            if all([min_val is None, max_val is None]):
                x = (b-a)*(x-np.min(x))/(np.max(x)-np.min(x))+a
            else:
                x = (b-a)*(x-min_val)/(max_val-min_val)+a
                
            return x

        return _normalize 
    
    def RandomResizedCrop3D(self, target_pixel_size, scale, ratio):
        
        if isinstance(target_pixel_size, int):
            target_pixel_size = [target_pixel_size]*3

        def _RandomResizedCrop3D(img):
            '''
            Same as torchvision.transforms.RandomResizedCrop but for 3D tensors
            
            Parameters:
            img (torch.tensor): A 4D image tensor with dimensions (channels, height, width, depth).
            scale (tuple of float): A tuple (min_scale, max_scale) defining the range of scaling factors to apply to the
                                    original volume of the image to determine the volume of the cropped region.
            ratio (tuple of float): A tuple (min_ratio, max_ratio) defining the range of aspect ratio distortions allowed
                                    during the crop operation.
            target_pixel_size (tuple of int): The desired dimensions (height, width, depth) to which the cropped volume will be 
                                    resampled to.

            Returns:
            numpy.ndarray: The cropped and resized 4D image tensor with the new specified target size.
            '''
            
            if len(img.shape) != 4:
                raise ValueError('Wrong number of dimensions, expected 4D, got: {}D'.format(len(img.shape)))
            
            _, height, width, depth = img.shape
            img_shape = np.array(img.shape[1:]).astype(int)
            
            area = height * width * depth
            
            
            # dict of arbitrary order of finding sizes of dims        
            target_sizes = {}
            dim_map = {
                0: 'h',
                1: 'w',
                2: 'd',
                }
            
            target_scale = random.uniform(*scale)
            target_volume = target_scale * area
            
            base_size = target_volume**(1/3)
            smallest_dim = np.argmin(img_shape)
            
            # set the smallest dim first. 
            # Check if the base_size is bigger than the smalles dim
            if base_size < img_shape[smallest_dim]:
                target_sizes[dim_map[smallest_dim]] = int(target_volume**(1/3))
            else:
                target_sizes[dim_map[smallest_dim]] = img_shape[smallest_dim]
            
            target_ratio = random.uniform(*ratio)
                
            remaining_dims = [0,1,2]
            remaining_dims.remove(smallest_dim)
            
            # apply the ratio distortion from the first dim to the second dim
            # Check if the ratio would be larger than the image and clip in that case
            
            # check if the scale would result in too large size in the remaining dims
            case_1 = int(target_volume**(1/3)*target_ratio) < img_shape[remaining_dims[0]]
            case_2 = int(target_volume/ \
                        (target_sizes[dim_map[smallest_dim]] * int(target_volume**(1/3)*target_ratio)) ) < img_shape[remaining_dims[1]]
                
            # the cube will fit
            if all([case_1, case_2]):
                target_sizes[dim_map[remaining_dims[0]]] = int(target_volume**(1/3)*target_ratio)
                target_sizes[dim_map[remaining_dims[1]]] = int(target_volume/ \
                                                            (target_sizes[dim_map[smallest_dim]]*target_sizes[dim_map[remaining_dims[0]]]) )       
            # it wont fit and we adjust one of the dims
            else:
                # If the ratio would results in any dimension being too big,
                # simply set it to size of that dimension of the image, and infer
                # last dim from that
                if target_ratio > 1:
                    target_sizes[dim_map[remaining_dims[0]]] = img_shape[remaining_dims[0]]
                    # infer the last dim from the first two
                    target_sizes[dim_map[remaining_dims[1]]] = int(target_volume/ \
                                                                (target_sizes[dim_map[smallest_dim]]*target_sizes[dim_map[remaining_dims[0]]]) )
                    
                else:
                    target_sizes[dim_map[remaining_dims[1]]] = img_shape[remaining_dims[1]]
                    # infer the last dim from the first two
                    target_sizes[dim_map[remaining_dims[0]]] = int(target_volume/ \
                                                                (target_sizes[dim_map[smallest_dim]]*target_sizes[dim_map[remaining_dims[1]]]) )
            
            # retrieve the sizes of each dim in a readerable format 
            h = target_sizes['h']
            w = target_sizes['w']
            d = target_sizes['d']
            
            # define the corner starting index of the cube
            h_0 = random.randint(0, height-h)
            w_0 = random.randint(0, width-w)
            d_0 = random.randint(0, depth-d)
            
            crop = img[:, h_0:h_0+h, w_0:w_0+w, d_0:d_0+d]
                
            resize_crop = F.interpolate(crop.unsqueeze(0), size=target_pixel_size, mode='trilinear', align_corners=False).squeeze(0)

            return resize_crop
        
        return _RandomResizedCrop3D

    def flip3D(self, p: list = None):
        """
        Create a function to flip tensor dimensions with specified probabilities.

        Args:
        p (list of float): Probabilities for flipping each dimension. Each value must be between 0 and 1.

        Returns:
        function: A function that takes a tensor and flips its dimensions based on the specified probabilities.
        """

        def _flip3D(tensor):
            """
            Flip the given tensor's dimensions with specified probabilities.

            Args:
            tensor (torch.Tensor): The tensor to potentially flip.

            Returns:
            torch.Tensor: The flipped (or unflipped, depending on the probability) tensor.
            """
            if p is None:
                return tensor
            else:
                # +1 because of the channel dim
                dims_to_flip = [i + 1 for i, prob in enumerate(p) if random.random() < prob]
                return torch.flip(tensor, dims=dims_to_flip)
        
        return _flip3D
    

class randomCrop3D():
    '''
    - crop_size (int or list of int): The size of the crop. If an int, a cubic crop is made; if a list of three ints, specifies dimension sizes.
    '''
    
    def __init__(self, crop_size):
        # Validate crop_size input and set dimensions
        if isinstance(crop_size, int):
            self.crop_size = [crop_size, crop_size, crop_size]
        elif isinstance(crop_size, list) and len(crop_size) == 3:
            self.crop_size  # Assume it's already correctly specified as [depth, height, width]
        else:
            raise ValueError("crop_size must be an int or a list of three ints")

    def __call__(self, tensor):
        """
        Returns a random 3D crop from a 4D torch tensor where the first dimension is the channel.

        Parameters:
        - tensor (torch.Tensor): The input 4D tensor with dimensions [channels, depth, height, width].

        Returns:
        - torch.Tensor: The randomly cropped 4D tensor, preserving the channel dimension.
        """


        # Get the dimensions of the spatial part of the tensor
        C, D, H, W = tensor.shape
        
        # Calculate padding if necessary
        pad_d = max(0, self.crop_size[0] - D)
        pad_h = max(0, self.crop_size[1] - H)
        pad_w = max(0, self.crop_size[2] - W)

        # Apply padding symmetrically to both sides of each dimension
        tensor_padded = F.pad(tensor, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2, pad_d//2, pad_d - pad_d//2))

        # New dimensions after padding
        D_padded, H_padded, W_padded = tensor_padded.shape[-3:]

        # Calculate random starting points for the crop
        start_d = torch.randint(0, D_padded - self.crop_size[0] + 1, (1,)).item()
        start_h = torch.randint(0, H_padded - self.crop_size[1] + 1, (1,)).item()
        start_w = torch.randint(0, W_padded - self.crop_size[2] + 1, (1,)).item()

        # Perform the crop
        return tensor_padded[:, start_d:start_d + self.crop_size[0], start_h:start_h + self.crop_size[1], start_w:start_w + self.crop_size[2]]


class ResNet152Checkpointed(torch.nn.Module):
    def __init__(self, resnet):
        super(ResNet152Checkpointed, self).__init__()
        # Load a pretrained ResNet-152 model
        # self.model = torchvision.models.resnet152(pretrained=True)
        self.model = resnet

        # Decompose the original ResNet-152 model to access its layers
        self.features = torch.nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        )
        self.avgpool = self.model.avgpool
        self.classifier = self.model.classifier
        # self.fc = self.model.fc

    def forward(self, x):
        # Apply gradient checkpointing to the feature extraction part
        # x = checkpoint(self.features, x)
        x = checkpoint_sequential(self.features, 5, x)
        
        # Complete the forward pass [for the originial resnet]
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        
        # this is needed for the inflated resnet, see: https://github.com/hassony2/inflated_convnets_pytorch/blob/master/src/i3res.py
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.squeeze(3)
        x = x.squeeze(3)
        x = x.mean(2)
        return x

###############
# 2D functions
###############
class NiftiDataset(Dataset):
    '''
    Loads CT dataset in .npy format and samples a random slice, i.e. 2D
    '''
    def __init__(self, directory, transform=None):
        super(NiftiDataset, self).__init__()
        
        self.transform = transform
        
        self.files = []
        # Recursively find all NIfTI files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy') and 'volume' in file:
                    self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)*700

    def __getitem__(self, idx):
        # Load the NIfTI file
        
        # hack to make the epochs longer for small datasets
        nifti_path = self.files[idx % len(self.files)]
        # nifti_img = nib.load(nifti_path)
        
        nifti_img = np.load(nifti_path, mmap_mode='r')
        
        # choose random slice
        _slice = np.random.randint(0, nifti_img.shape[-1])
        
        # load only specfic slice (5-10x faster)
        # img_slice = nifti_img.dataobj[:,:,_slice].copy()
        img_slice = nifti_img[:,:,_slice].copy()
    
        # Convert the data to a PyTorch tensor
        img_slice = torch.tensor(img_slice, dtype=torch.float32)
        img_slice = img_slice.unsqueeze(0)  
        # to make the resnet accept it 
        img_slice = torch.repeat_interleave(img_slice, 3, dim=0)
        
        if self.transform is not None:
            return self.transform(img_slice), torch.tensor(0)
        else:
            img_slice, torch.tensor(0)

class DataAugmentationDINO_CT(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        '''
        Create the augmentations and crops for each image. The crops a determined by
        the percentage (scale) of the image, and then they're resampled into the crops size

        Parameters
        ----------
        global_crops_scale :  (tuple of python:float) 
             Specifies the lower and upper bounds for the random area of the crop, 
             before resizing. The scale is defined with respect to the area of the original image.
        local_crops_scale :  (tuple of python:float) 
            Specifies the lower and upper bounds for the random area of the crop, 
             before resizing. The scale is defined with respect to the area of the original image.
        local_crops_number : int
            Number of local random crops (two global is default)

        Returns
        -------
        None.

        '''
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            #     p=0.8
            # ),
            # transforms.RandomGrayscale(p=0.2),
        ])
        
        # normalize = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        # normalize = lambda x: x
        normalize = self.normalize(min_val=-400, max_val=400, a=0, b=1)

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            # flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            # utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        
        # image[400 < image] = 400
        # image[-400 > image] = -400
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
    def normalize(self, min_val=None, max_val=None, a=None, b=None):
        '''
        x is the data to be normalized. If min and max is None, the min and max from the data is used. If a and b is None, else 
        the data is normalized between 0 and 1
        '''
        def _normalize(x):
            
            nonlocal a, b # so they can be changed
            # 0 to 1 is default
            if all([a is None, b is None]):
                a = 0
                b = 1    
            # check whether a max and min has been given
            if all([min_val is None, max_val is None]):
                x = (b-a)*(x-np.min(x))/(np.max(x)-np.min(x))+a
            else:
                x = (b-a)*(x-min_val)/(max_val-min_val)+a
                
            return x

        return _normalize 

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
