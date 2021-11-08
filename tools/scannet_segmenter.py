
import argparse
import os
from pathlib import Path
from copy import copy
import shutil
import mmcv
import torch
import numpy as np
import json

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate, scatter
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import (wrap_fp16_model, bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, multiclass_nms)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.pipelines import Compose
from mmdet.ops.roi_sampling import roi_sampling, invert_roi_bbx
#from mmdet.core import scannet_originalIds

from pycocotools.mask import decode as decode_rle

from PIL import Image
import cv2
import random

import torch.nn.functional as F

def load_classes(file):
    data = load_json(file)

    labels  = []
    names   = []
    palette = []
    continuous_ids = []

    for k in data.keys():
        labels.append(int(k))
        names.append(data[k]["name"])
        palette.append(data[k]["rgb"])
        continuous_ids.append(data[k]["continuous_id"])

    return labels, names, palette, continuous_ids

def load_json(path):
    if isinstance(path, Path):
        path = str(path.resolve())

    with open(path, 'r') as f:
        data = json.load(f)

    return data

def get_colors_palete(num_classes):
    colors = [randRGB(i+5) for i in range(num_classes + 1)]
    return colors

def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()*255
    g = random.random()*255
    b = random.random()*255
    rgb = [r, g, b]
    return rgb

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        return results

class Segmenter():
    def __init__(
            self,
            class_json = "scannet_classes.json",
            config = None,
            checkpoint = None
        ):

        assert config is not None
        assert checkpoint is not None

        self.model = init_detector(config, checkpoint, device='cuda:0')
        self.device = next(self.model.parameters()).device

        # build the data pipeline
        data_pipeline = [LoadImage()] + self.model.cfg.data.test.pipeline[1:]
        self.data_pipeline = Compose(data_pipeline)

        self.labels, self.names, self.colors, self.indices = load_classes(class_json)

        # ensure voi label is set correctly
        if 'void' in self.names:
            i = self.names.index('void')

            del self.labels[i]
            del self.names[i]
            del self.colors[i]

        self.void_label = 255

        self.labels.append(self.void_label)
        self.names.append('void')
        self.colors.append([0,0,0])

        self.orig_shape = (968, 1296)

    def apply_semantic_mask(self, image, mask, alpha, colorconf="json"):

        unique_values = np.unique(mask)
        num_classes = unique_values.shape

        if image.shape != self.orig_shape:
            image = cv2.resize(
                image,
                (self.orig_shape[1], self.orig_shape[0]),
                interpolation = cv2.INTER_NEAREST)

        image_arr = np.array(image)

        void_ind = self.names.index('void')

        for i in unique_values:
            if i == self.void_label:
                if alpha < 1: continue

                color = self.colors[self.labels.index(i)]

            elif colorconf == "random":
                color = randRGB(i)
            else:
                color = self.colors[i]

            for c in range(3):
                image_arr[:, :, c] = np.where(mask == i,
                                          image_arr[:, :, c] *
                                          (1 - alpha) + alpha * color[c],
                                          image_arr[:, :, c])
        return image_arr

    def get_panoptic_pred(self, img_path):
        result = inference_detector(self.model, img_path, 'panoptic')

        pan_pred, cat_pred, img_metas = result[0]
        pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()

        # EfficientPS does not output confidence
        # -> assume one for detected class and zero for others
        confidence = np.ones(cat_pred.shape[0])

        # Efficientps's output is quantised to 32 pixel intervals,
        # need to resize to match original images
        if pan_pred.shape != self.orig_shape:
            pan_pred = cv2.resize(
                pan_pred,
                (self.orig_shape[1], self.orig_shape[0]),
                interpolation = cv2.INTER_NEAREST)

        return pan_pred, cat_pred, confidence

    def forward(self, img, tmp_name=".eps_tmp2"):

        tmp_remove_flag = False

        # workaround for having to load images from a file when using the original
        # data pipeline
        if not isinstance(img, str):
            tmp_dir = Path(tmp_name)
            tmp_dir.mkdir(exist_ok=True)
            tmp_remove_flag = True

            imgName = "tmp2.png"
            imgName = str(tmp_dir / Path(imgName))

            cv2.imwrite(imgName, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        else:
            imgName = img

        categories = None

        pred, categories, confidence = self.get_panoptic_pred(imgName)

        semantic_mask = np.ones((pred.shape[0], pred.shape[1]), dtype=np.uint8)*self.void_label
        instance_results = None

        panPredIds = np.unique(pred)

        for panPredId in panPredIds:
            if categories[panPredId] == self.void_label:
                continue
            else:
                semanticId = categories[panPredId]

            semantic_mask[pred == panPredId] = semanticId

        if tmp_remove_flag:
            shutil.rmtree(tmp_name)

        return semantic_mask, instance_results, pred, categories, confidence

    def visualize_panoptseg(self, img, pred, cat, alpha=0.5):

        STUFF = [0,1]

        stuff_mask = np.ones(pred.shape, dtype=np.uint8)*self.void_label
        thing_mask = np.ones(pred.shape, dtype=np.uint8)*self.void_label

        for idx in np.unique(pred):
            l = cat[idx]

            if l in STUFF:
                stuff_mask[pred == idx] = l
            elif l != self.void_label:
                thing_mask[pred == idx] = idx

        stuff_img = self.apply_semantic_mask(copy(img), stuff_mask, alpha=alpha, colorconf="json")
        thing_img = self.apply_semantic_mask(copy(img), thing_mask, alpha=alpha, colorconf="random")

        seg_img = copy(img)

        seg_img[stuff_mask != self.void_label] = stuff_img[stuff_mask != self.void_label]
        seg_img[thing_mask != self.void_label] = thing_img[thing_mask != self.void_label]

        return seg_img, {}

    def visualize_semseg(self, img, mask, alpha=0.5):
        seg_img = self.apply_semantic_mask(img, mask, alpha=alpha)

        return seg_img, {}

    def visualize_instseg(self, img, inst_seg, alpha=0.5, confidence=0.5):

        if inst_seg is None: return None, None

        labels = inst_seg["label"]
        masks = np.moveaxis(np.asarray(inst_seg["masks"]), 2, 1)

        if masks.shape[0] == 0:
            seg_img = img
        else:
            mask_argmax = np.argmax(masks, axis=0)
            mask_argmax = np.where(mask_argmax == 0, self.void_label, mask_argmax)
            seg_img = self.apply_semantic_mask(copy(img), mask_argmax, alpha=alpha, colorconf="random")

        return seg_img, {}
