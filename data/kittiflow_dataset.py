"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torch
import numpy as np


class KittiflowDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        # parser.add_argument('--coco_no_portraits', action='store_true')
        # for image flow
        parser.add_argument('--patch_size', type=int, default=7, help='kernel size for feature correlation')
        parser.add_argument('--search_size', type=int, default=11, help='search steps for feature correlation')
        parser.add_argument('--search_stride', type=int, default=2, help='stride for feature correlation')
        parser.set_defaults(preprocess_mode='crop')
        parser.set_defaults(load_size=286)
        parser.set_defaults(crop_size=256) # crop_size must be the scale of 32
        parser.set_defaults(patch_size=7)
        parser.set_defaults(search_size=11)
        parser.set_defaults(search_stride=2)
        parser.set_defaults(display_winsize=256)
        # parser.set_defaults(label_nc=182)
        parser.set_defaults(label_nc=3)
        # parser.set_defaults(contain_dontcare_label=True)
        # parser.set_defaults(cache_filelist_read=True)
        # parser.set_defaults(cache_filelist_write=True)
        return parser

    def initialize(self, opt):
        self.opt = opt

        # label_paths, image_paths, instance_paths = self.get_paths(opt)
        self.dir_AB = os.path.join(opt.dataroot, 'train_origin_1')
        self.dir_sem = os.path.join(opt.dataroot, 'train_semantic')
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.sem_paths = sorted(make_dataset(self.dir_sem))
        # self.AB_paths = AB_paths[:opt.max_dataset_size]
        size = len(self.AB_paths)
        self.dataset_size = size-1
        self.content_crop_size = opt.crop_size
        self.cps = opt.patch_size + opt.search_stride * (opt.search_size - 1)
        self.search_pad = (self.cps - opt.patch_size) // 2
        self.crop_size = opt.crop_size
        self.load_size  = self.crop_size + 2 * self.search_pad

        # # 此处为另外两个视角的图片
        # self.dir_CD = os.path.join(opt.dataroot,'image_3')

        # self.CD_paths = sorted(make_dataset(self.dir_CD))




    # take a random mask applied on a real image as the input
    def __getitem__(self, index):

        # take two images next to the real image as the input of SPADE
        AB_path = self.AB_paths[index % self.dataset_size + 1]
        image_path = AB_path
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into mask image and real image
        w, h = AB.size
        h2 = int(h / 2)
        # content = AB.crop((0, 0, w, h2))
        real_img = AB.crop((0, h2, w, h))


        condition_path = self.AB_paths[index % self.dataset_size]
        condition = Image.open(condition_path).convert('RGB')
        condition = condition.crop((0, h2, w, h))


        # condition_path_view_one = self.CD_paths[index % self.dataset_size]
        # condition_view_one = Image.open(condition_path_view_one).convert('RGB')
        # # 这里的crop情况需要进行修改
        # condition_view_one = condition_path_view_one.crop((0,h2,w,h))


        # get a new mask image randomly
        mask_index = random.randint(0, self.dataset_size)
        mask_path = self.sem_paths[mask_index % self.dataset_size]
        # assert os.path.basename(AB_path) == os.path.basename(mask_path), \
        #     'mask_image(content) should be matched to mask_semantic'
        mask = Image.open(mask_path).convert('RGB')
        sem_tensor = transforms.ToTensor()(mask)
        img_tensor = transforms.ToTensor()(real_img)
        mask_tensor = torch.where(sem_tensor>0, img_tensor, sem_tensor)
        content = transforms.ToPILImage()(mask_tensor)

        params = get_params(self.opt, self.load_size, condition.size)
        if self.load_size > condition.size[1]:
            condition_size = [self.crop_size + 2 * self.search_pad, 320 + 2 * self.search_pad]
            content_size = [self.crop_size, 320]
        else:
            condition_size = [self.crop_size + 2 * self.search_pad, self.crop_size + 2 * self.search_pad]
            content_size = [self.crop_size, self.crop_size]
        # params = get_params(self.opt, condition.size)
        content_transforms = get_transform(self.opt, params, content_size)
        condition_transforms = get_transform(self.opt, params, condition_size)
        mask_transforms = get_transform(self.opt, params, content_size, normalize=False)
        real_image = content_transforms(real_img)
        content = content_transforms(content)
        condition = condition_transforms(condition)
        mask = mask_transforms(mask) * 255.0


        instance_tensor = 0

        # refer condition 另一个视角的真实图片
        # mask 随机选取的mask mask 区域值为0 其余区域为标签值信息
        # label content 带mask区域的实际图片 mask 区域的值为0

        input_dict = {'label': content,
                      'mask': mask,
                      'refer': condition,
                      'instance': instance_tensor,
                      'image': real_image,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


