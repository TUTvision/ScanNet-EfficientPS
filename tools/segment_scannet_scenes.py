
from scannet_segmenter import Segmenter as EfficientPS

import sys, os
import time
import json
import re
from argparse import ArgumentParser
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import mmcv

SEM_LABEL_MAP  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
INST_LABEL_MAP = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def write_panoptic_result(panoptic_pred, panoptic_cat, confidence_scores, mask_path, label_map, void_label=255):

    # dim: [category, instance, confidence]
    panoptic_encoding = np.zeros((panoptic_pred.shape[0],panoptic_pred.shape[1],3))
    panoptic_encoding[...,0] = panoptic_pred
    #panoptic_encoding[...,2] = 0 # always zero for void

    unique_values = np.unique(panoptic_pred)
    for i, val in enumerate(unique_values):
        category = panoptic_cat[val]
        confidence = confidence_scores[val] * 255 if category != void_label else 0

        idx = panoptic_pred==val

        #panoptic_encoding[idx, 0] = instance
        panoptic_encoding[idx, 1] = category
        panoptic_encoding[idx, 2] = confidence

    panoptic_encoding = panoptic_encoding.astype(np.uint8)

    cv2.imwrite(str(mask_path.resolve()), panoptic_encoding)

def write_semantic_benchmark_mask(mask, filename, label_map, void_label=255):
    unique_values = np.unique(mask)
    num_classes = unique_values.shape

    bm_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for i in unique_values:
        if i == void_label: continue
        bm_mask = np.where(mask == i, label_map[i], bm_mask)

    if isinstance(filename, Path):
        filename = str(filename.resolve())

    cv2.imwrite(filename, bm_mask)

def write_instance_benchmark_masks(inst_seg, dir, mask_dir, view_name):

    if inst_seg is None: return

    masks  = inst_seg["masks"]
    conf   = inst_seg["conf"]
    labels = inst_seg["label"]

    if isinstance(masks, list):
        n = len(masks)
    else:
        n = masks.shape[0]

    txt_output = dir / Path(view_name + '.txt')

    with open(txt_output, "w") as txtf:
        for i in range(n):
            label = INST_LABEL_MAP[labels[i]]
            confidence = conf[i]

            if label == 0: continue

            mask_name = mask_dir + os.sep + f"{view_name}_{i:03d}.png"
            mask_path = dir / Path(mask_name)

            cv2.imwrite(str(mask_path.resolve()), masks[i])

            line = f"{mask_name} {label} {confidence}\n"

            txtf.write(line)

def make_dirs(prefix, relative_paths):

    mapping = {}

    for p in relative_paths:
        path = prefix / Path(p)

        if not path.is_dir():
            path.mkdir(parents=True)

        mapping[p] = path

    return mapping

def label_from_color(vert_color, seg_colors, label_map):
    color_arr = np.asarray(seg_colors)
    vc_arr = np.asarray(vert_color)

    diff = np.sum(np.abs(color_arr - vc_arr), axis=1)
    label = label_map[np.argmin(diff)]

    return label

def get_scene_num(filename):

    name = filename.split('_')[0]

    numbers = re.findall(r'\d+', name)

    return int(''.join(numbers))

def load_json(path):
    if isinstance(path, Path):
        path = str(path.resolve())

    with open(path, 'r') as f:
        data = json.load(f)

    return data

def filter_scenes_with_json(json, scene_list):

    data = load_json(json)

    new_list = []
    scene_nums = []

    for img_entry in data['images']:
        fname = img_entry['file_name']

        scene_nums.append(get_scene_num(fname))

    for scene in scene_list:

        name = scene.name

        if get_scene_num(name) in scene_nums:
            new_list.append(scene)

    return new_list

def main(
    source_dir,
    output_dir,
    config_path,
    checkpoint_path,
    scene_input,
    evaluate_panoptic,
    evaluate_semantic,
    evaluate_instance,
    visualise_panoptic,
    visualise_semantic,
    visualise_instance):

    output_dir = output_dir.resolve()
    scene_output_dir = output_dir / Path("scenes")

    semantic_eval_path = output_dir / Path("eval") / Path("semantic")
    semantic_viz_alpha_path = "semantic_seg"
    semantic_viz_solid_path = "semantic_mask"

    panoptic_eval_path = "panoptic"
    panoptic_viz_alpha_path = "panoptic_seg"
    panoptic_viz_solid_path = "panoptic_mask"

    instance_eval_path = output_dir / Path("eval") / Path("instance")
    instance_eval_mask_path = instance_eval_path / Path("predicted_masks")
    instance_viz_path  = "instance_seg"

    output_dirs = []

    # ScanNet eval format
    if evaluate_semantic:
        semantic_eval_path.mkdir(parents=True, exist_ok=True)
    if evaluate_instance:
        instance_eval_path.mkdir(parents=True, exist_ok=True)

    # our format
    if evaluate_panoptic:
        output_dirs.append(panoptic_eval_path)
    if visualise_panoptic:
        output_dirs.append(panoptic_viz_alpha_path)
        output_dirs.append(panoptic_viz_solid_path)
    if visualise_semantic:
        output_dirs.append(semantic_viz_alpha_path)
        output_dirs.append(semantic_viz_solid_path)
    if visualise_instance:
        output_dirs.append(instance_viz_path)

    config = mmcv.Config.fromfile(str(config_path.resolve()))
    seg = EfficientPS(
        config=str(config_path.resolve()),
        checkpoint=str(checkpoint_path.resolve()))

    seg_colors = seg.colors
    seg_labels = seg.labels
    void_label = seg.void_label

    EXTENSIONS = {'.jpg', '.png'}

    scenes = list(source_dir.glob('scene*'))

    if scene_input is not None:
        scene_input = Path(scene_input)
        if scene_input.suffix == '.json':
            scenes = filter_scenes_with_json(scene_input, scenes)
        elif scene_input.suffix == '.txt':
            scenes = filter_scenes_with_txt(scene_input, scenes)
        else:
            scenes = [source_dir / Path(scene_input)]

    n_scenes = len(scenes)
    print(f"N_SCENES: {n_scenes}")

    for i, d in enumerate(scenes):
        print("Scene {}/{}".format(i+1, n_scenes), end="                    \r")

        if d.is_dir():
            scene = d.name
            scene_dir = scene_output_dir / Path(scene)

            scene_output = make_dirs(scene_dir, output_dirs)

            color_files = list(d.glob('**/color/*'))
            color_files.sort(key=lambda x: int(str(x.name).split('.')[0]))

            for k, f in enumerate(color_files):
                if f.suffix in EXTENSIONS:
                    img_rgb = cv2.cvtColor(cv2.imread(str(f.resolve()), -1), cv2.COLOR_BGR2RGB)

                    # Some of the images in ScanNet are different size than others, and
                    # segmentation output is always the same size
                    if img_rgb.shape[:2] != seg.orig_shape:
                        img_rgb = cv2.resize(
                            img_rgb,
                            (seg.orig_shape[1], seg.orig_shape[0]),
                            interpolation = cv2.INTER_LINEAR)

                    sem_mask, inst_seg, panoptic_pred, panoptic_cat, confidence = seg.forward(img_rgb)

                    if evaluate_semantic:
                        semantic_eval_file_path = semantic_eval_path / f"{scene}_{f.stem}.png"
                        write_semantic_benchmark_mask(sem_mask, semantic_eval_file_path, seg_labels)

                    if visualise_semantic:
                        semantic_viz_alpha_file_path = str(scene_output[semantic_viz_alpha_path] / Path(f.stem + '.png'))
                        semantic_viz_solid_file_path = str(scene_output[semantic_viz_solid_path] / Path(f.stem + '.png'))

                        viz_semantic_alpha,_ = seg.visualize_semseg(img_rgb, sem_mask, alpha=0.5)
                        viz_semantic_solid,_ = seg.visualize_semseg(img_rgb, sem_mask, alpha=1)
                        cv2.imwrite(semantic_viz_alpha_file_path, cv2.cvtColor(viz_semantic_alpha, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(semantic_viz_solid_file_path, cv2.cvtColor(viz_semantic_solid, cv2.COLOR_RGB2BGR))

                    if evaluate_panoptic:
                        panoptic_eval_file_path = scene_output[panoptic_eval_path] / Path(f"{f.stem}.png")
                        write_panoptic_result(panoptic_pred, panoptic_cat, confidence, panoptic_eval_file_path, seg_labels)

                    if visualise_panoptic:
                        panoptic_viz_alpha_file_path = str(scene_output[panoptic_viz_alpha_path] / Path(f"{f.stem}.png"))
                        panoptic_viz_solid_file_path = str(scene_output[panoptic_viz_solid_path] / Path(f"{f.stem}.png"))

                        viz_panopt_alpha,_ = seg.visualize_panoptseg(img_rgb, panoptic_pred, panoptic_cat, alpha=0.5)
                        viz_panopt_solid,_ = seg.visualize_panoptseg(np.zeros(img_rgb.shape, dtype=np.uint8), panoptic_pred, panoptic_cat, alpha=1)
                        cv2.imwrite(panoptic_viz_alpha_file_path, cv2.cvtColor(viz_panopt_alpha, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(panoptic_viz_solid_file_path, cv2.cvtColor(viz_panopt_solid, cv2.COLOR_RGB2BGR))

                    if evaluate_instance:
                        write_instance_benchmark_masks(inst_seg, instance_eval_path, instance_eval_mask_path, f"{scene}_{f.stem}.png")

                    if visualise_instance:
                        viz_inst,_ = seg.visualize_instseg(img_rgb, inst_seg, alpha=0.5, confidence=0.5)

                        if viz_inst is None: continue

                        instance_viz_file_path = str(scene_output[instance_viz_path] / Path("instance") / Path(f"{f.stem}.png"))
                        cv2.imwrite(instance_viz_file_path, viz_inst)

if __name__ == '__main__':

    launch_default = Path("../launch/fusion_from_dir.launch")
    launch_default = str(launch_default.resolve())

    parser = ArgumentParser()

    parser.add_argument('-s', '--source', type=Path, dest="source_dir", help='input directory')
    parser.add_argument('-o', '--output', type=Path, dest="output_dir", help='output directory')

    parser.add_argument('-cfg', '--config', type=Path, dest="config_path", help='model config file')
    parser.add_argument('-cpt', '--checkpoint', type=Path, dest="checkpoint_path", help='model weights checkpoint')

    parser.add_argument('-si', '--scene_input', type=str, dest="scene_input", default=None,
                        help='Input to specify scenes to process (scene id, .txt, or .json)')

    parser.add_argument('-ep', '--evaluate_panoptic', type=str2bool, dest="evaluate_panoptic", default=True, help='')
    parser.add_argument('-es', '--evaluate_semantic', type=str2bool, dest="evaluate_semantic", default=True, help='')
    parser.add_argument('-ei', '--evaluate_instance', type=str2bool, dest="evaluate_instance", default=True, help='')
    parser.add_argument('-vp', '--visualise_panoptic', type=str2bool, dest="visualise_panoptic", default=True, help='')
    parser.add_argument('-vs', '--visualise_semantic', type=str2bool, dest="visualise_semantic", default=True, help='')
    parser.add_argument('-vi', '--visualise_instance', type=str2bool, dest="visualise_instance", default=True, help='')

    args = parser.parse_args()

    main(**vars(args))
