
from SensorData import SensorData, RGBDFrame

import sys, os
import time
import json
import re
from argparse import ArgumentParser
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def main(source_dir, output_dir, frame_skip, scene, color, depth, intrinsics, poses):

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    scenes = list(source_dir.glob('scene*'))

    if args.scene is not None:
        scene_input = Path(args.scene)
        if scene_input.suffix == '.json':
            scenes = filter_scenes_with_json(str(scene_input), scenes)
        else:
            scenes = [scene_input]

    n_scenes = len(scenes)
    print(f"N_SCENES: {n_scenes}")

    times_per_scene = []

    for i, d in enumerate(scenes):
        relative = d.relative_to(d.parent)

        print("Scene {}/{} : {}".format(i+1, n_scenes, str(relative)), end="                    \r")

        sens_file = next(d.glob('*.sens'))
        SD = SensorData(sens_file)

        file_output = output_dir / relative

        if intrinsics:
            SD.export_intrinsics(str(file_output.resolve()))
        if color:
            SD.export_color_images(str(file_output.resolve() / "color"), frame_skip=frame_skip)
        if depth:
            SD.export_depth_images(str(file_output.resolve() / "depth"), frame_skip=frame_skip)
        if poses:
            SD.export_poses(str(file_output.resolve() / "pose"), frame_skip=frame_skip)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-s", "--source_dir", type=Path, dest="source_dir",
                        help="Path to ScanNet scans root")
    parser.add_argument("-f", "--frame_skip", type=int, dest="frame_skip",
                        default=10, help="")
    parser.add_argument('--scene', type=str, default=None)

    parser.add_argument('-o', dest="output_dir", type=Path,
        help="Path to desired output directory")

    parser.add_argument('--intrinsics', type=str2bool, default=True)
    parser.add_argument('--color', type=str2bool, default=True)
    parser.add_argument('--depth', type=str2bool, default=True)
    parser.add_argument('--poses', type=str2bool, default=True)

    args = parser.parse_args()

    main(**vars(args))
