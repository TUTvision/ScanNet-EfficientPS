
import os
import re
import shutil
import json
import threading
import multiprocessing
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser

from PIL import Image
import cv2
import numpy as np

def read_json(json_source):
    with open(str(json_source.resolve()), 'r') as j:
        data = json.loads(j.read())

    return data

def write_json(path, data):
    with open(str(path.resolve()), 'w') as f:
        json.dump(data, f)

def run_panoptic_conversion(source_dir, scannet_root, output_path):

    main_dir = os.getcwd()

    sc_script_dir = scannet_root / Path("BenchmarkScripts/")
    convert_script = "convert2panoptic.py"

    convert_cmd = f"{convert_script} \
        --dataset-folder {str(source_dir.resolve())} \
        --output-folder {str(output_path.resolve())}"

    # ScanNet scripts have to be called from their original directory ...

    os.chdir(sc_script_dir)
    os.system(f'python {convert_cmd }')
    os.chdir(main_dir)

def run_panoptic_to_detection_conversion(source_json, panopticapi_root, output_json, categories_json):

    main_dir = os.getcwd()

    script_dir = panopticapi_root / Path("converters/")
    convert_script = "panoptic2detection_coco_format.py"

    convert_cmd = f"{convert_script} \
        --input_json_file {str(source_json.resolve())} \
        --output_json_file {str(output_json.resolve())} \
        --categories_json_file {str(categories_json.resolve())} \
        --things_only"

    # COCO panopticapi converters also have to be called from their original directory ...

    os.chdir(script_dir)
    os.system(f'python3 {convert_cmd}')
    os.chdir(main_dir)

def run_panoptic_to_semantic_conversion(source_json, panopticapi_root, output_dir, categories_json):

    main_dir = os.getcwd()

    script_dir = panopticapi_root / Path("converters/")
    convert_script = "panoptic2semantic_segmentation.py"

    convert_cmd = f"{convert_script} \
        --input_json_file {str(source_json.resolve())} \
        --categories_json_file {str(categories_json.resolve())} \
        --semantic_seg_folder {str(output_dir.resolve())}"

    # COCO panopticapi converters also have to be called from their original directory ...

    os.chdir(script_dir)
    os.system(f'python3 {convert_cmd}')
    os.chdir(main_dir)

def make_categories_json(source_json, categories_json):

    data = read_json(source_json)
    categories = data["categories"]

    write_json(categories_json, categories)

def get_scene_num(id):
    scene_name = id.split("_")[0]

    numbers = re.findall(r'\d+', scene_name)

    scene_number = int(''.join(numbers)) if len(numbers) > 0 else None

    return scene_number

def get_split(scenes_file, numbers = True):

    scenes = []

    with open(scenes_file, 'r') as f:
        for line in f:
            if get_scene_num(line) is not None:
                if numbers:
                    scenes.append(get_scene_num(line))
                else:
                    scenes.append(line.strip())

    return scenes

def split_train_val(source_json, train_json, val_json, train_scenes, val_scenes):

    data = read_json(source_json)

    train_data = {
        "annotations":[],
        "images":[],
        "categories":data["categories"]
        }
    val_data = deepcopy(train_data)

    for a in data["annotations"]:
        id = a["image_id"]
        number = get_scene_num(id)

        if number in val_scenes:
            val_data["annotations"].append(a)
        elif number in train_scenes:
            train_data["annotations"].append(a)
        else:
            print(f"Annotation with id {id} not found in either split")

    for im in data["images"]:
        id = im["id"]
        number = get_scene_num(id)

        if number in val_scenes:
            val_data["images"].append(im)
        elif number in train_scenes:
            train_data["images"].append(im)
        else:
            print(f"Image with id {id} not found in either split")

    write_json(train_json, train_data)
    write_json(val_json, val_data)

def unfold_one_image(image_meta):
    orig_path = orig_root / Path(image_meta["file_name"])
    new_path = img_path / Path(str(image_meta["id"]) + '.png')

    img = Image.open(str(orig_path.resolve()))
    img.save(str(new_path.resolve()))

    image_meta["file_name"] = new_path

def unfold_images(json_source, orig_root, image_dir = "images"):

    # backup json just in case
    json_source_str = str(json_source.resolve())

    i = 0
    backup_json = json_source.parent / Path(json_source_str+".backup"+str(i))

    while backup_json.exists():
        i += 1
        backup_json = json_source.parent / Path(json_source_str+".backup"+str(i))

    shutil.copyfile(json_source_str, str(backup_json.resolve()))

    img_path = json_source.parent / Path(image_dir)

    if not img_path.is_dir():
        img_path.mkdir()

    data = read_json(json_source)
    images = data["images"]

    # TODO: multithread

    for i, img in enumerate(images):
        print(f"Unfold images {i+1} / {len(images)}", end="                 \r")

        orig_path = orig_root / Path(img["file_name"][:-4]+'.jpg')
        new_path = img_path / Path(str(img["id"]) + '.png')

        if not new_path.exists():
            im = Image.open(str(orig_path.resolve()))
            im.save(str(new_path.resolve()))

        img["file_name"] = new_path.name

    write_json(json_source, data)

def transform_json_to_continuous_ids(source_json):

    data = read_json(source_json)

    categories = data["categories"]
    annotations = data["annotations"]

    old_ids = []
    new_ids = []

    for i, c in enumerate(categories):
        old_ids.append(int(c["id"]))
        new_ids.append(i)

        categories[i]["id"] = i

    for i, a in enumerate(annotations):
        old_id =int(a["category_id"])
        new_id = new_ids[old_ids.index(old_id)]

        annotations[i]["category_id"] = new_id

    output_json = source_json.parent / Path(source_json.stem + "_continuous_ids.json")
    write_json(output_json, data)

    return old_ids, new_ids, output_json

def transform_semantic_gt_to_continuous_ids(semantic_dir, old_ids, new_ids):

    new_semantic_dir = semantic_dir.parent / Path(semantic_dir.stem + "_continuous")
    if not new_semantic_dir.is_dir():
        new_semantic_dir.mkdir(exist_ok=True, parents=True)

    images = list(semantic_dir.glob('*'))

    EXTENSIONS = {'.jpg', '.png'}
    for i, f in enumerate(images):
        if f.suffix in EXTENSIONS:
            print(f"Transform semantic gt to continuous {i+1} / {len(images)}", end="                 \r")

            orig_seg = cv2.imread(str(f), -1).astype(np.uint8)
            new_seg = np.zeros(orig_seg.shape, dtype=np.uint8)

            for ii in range(len(old_ids)):
                new_seg[orig_seg == old_ids[ii]] = new_ids[ii]

            new_seg_file = new_semantic_dir / Path(f.name)
            cv2.imwrite(str(new_seg_file.resolve()), new_seg)

def separate_panoptic_val(source_json, panoptic_dir, output_dir=None):
    data = read_json(source_json)

    if output_dir is None:
        output_dir = panoptic_dir.parent / Path(panoptic_dir.stem + "_val")

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    for a in data["annotations"]:

        img_input_path  = panoptic_dir / Path(a["file_name"])
        img_output_path = output_dir / Path(a["file_name"])

        shutil.copyfile(str(img_input_path.resolve()), str(img_output_path.resolve()))

def main(source_dir, train_scenes_file, val_scenes_file, output_path, scannet_root, panopticapi_root):

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    # Call a ScanNet script to convert data to COCO Panoptic format
    run_panoptic_conversion(source_dir, scannet_root, output_path)

    panoptic_json = output_path / Path("scannet_panoptic.json")
    detection_json = output_path / Path("scannet_detection.json")

    panoptic_dir = output_path / Path("scannet_panoptic")
    semantic_dir = output_path / Path("scannet_semantic")

    categories_json = output_path / Path("scannet_categories.json")

    # EfficientPS loader assumes annotated images are at the same relative path
    # inside their folder as the original images
    # -> easiest fix is to unfold all images from their scene directories to a single dir
    #
    # EfficientPS also assumes all images are in the same format,
    # so orginal .jpg rgb images are saved as .png
    unfold_images(panoptic_json, source_dir)

    # COCO scripts require a separate categories json which is created here
    make_categories_json(panoptic_json, categories_json)

    # Call panopticapi scripts to convert COCO Panoptic format to the (sort of) COCO Detection format used by EfficientPS
    run_panoptic_to_detection_conversion(panoptic_json, panopticapi_root, detection_json, categories_json)
    run_panoptic_to_semantic_conversion(panoptic_json, panopticapi_root, semantic_dir, categories_json)

    # For some reason, training of EfficientPS fails if category id's are not continuous ...
    # e.g. when we have 19 classes, id's must be in the range [0,19]
    # Modify the annotation json to have continuous id's,
    # which are used later to edit semantic ground truth
    old_ids, new_ids, continuous_detection_json = transform_json_to_continuous_ids(detection_json)

    # edit semantic ground truth to have continuous id's
    transform_semantic_gt_to_continuous_ids(semantic_dir, old_ids, new_ids)

    train_scenes = get_split(train_scenes_file)
    val_scenes = get_split(val_scenes_file)

    panoptic_train_json = panoptic_json.parent / Path(panoptic_json.stem + "_train.json")
    panoptic_val_json   = panoptic_json.parent / Path(panoptic_json.stem + "_val.json")

    detection_train_json = continuous_detection_json.parent / Path(continuous_detection_json.stem + "_train.json")
    detection_val_json   = continuous_detection_json.parent / Path(continuous_detection_json.stem + "_val.json")

    # generate separate json files for train / validation splits
    split_train_val(
        panoptic_json,
        panoptic_train_json,
        panoptic_val_json,
        train_scenes, val_scenes)
    split_train_val(
        continuous_detection_json,
        detection_train_json,
        detection_val_json,
        train_scenes, val_scenes)

    # effcientps evaluation needs panoptic validation json and directory
    # to have the same name, so we might as well copy validation gt to
    # another directory...
    separate_panoptic_val(panoptic_val_json, panoptic_dir)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-s', dest="source_dir", type=Path,
        help="Path to 'scannet_frames_25k' root")
    parser.add_argument('-t', dest="train_scenes_file", type=Path,
        help="Path to train split (.txt)")
    parser.add_argument('-v', dest="val_scenes_file", type=Path,
        help="Path to validation split (.txt)")

    parser.add_argument('-sc', dest="scannet_root", type=Path,
        help="Path to ScanNet root directory")
    parser.add_argument('-pn', dest="panopticapi_root", type=Path,
        help="Path to COCO panopticapi root directory")

    parser.add_argument('-o', dest="output_path", type=Path,
        help="Path to desired output directory")

    args = parser.parse_args()

    main(**vars(args))
