
# EfficientPS trained on ScanNet dataset
A modified version of EfficientPS for training on the ScanNet dataset

Original repository:
[https://github.com/DeepSceneSeg/EfficientPS](https://github.com/DeepSceneSeg/EfficientPS)

ScanNet dataset:
[http://www.scan-net.org/](http://www.scan-net.org/)
## Pre-Trained Models
| Dataset   |  Model | PQ |
|-----------|:-----------------:|--------------|
| ScanNet   | [Download](link)  | x |

## Installation:
Follow the instructions in the original repository.

## Prepare ScanNet Data for training and validation

Requirements:
- clone ScanNet source code [https://github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet)
- clone and install COCO panopticapi [https://github.com/cocodataset/panopticapi](https://github.com/cocodataset/panopticapi)
```
git clone https://github.com/ScanNet/ScanNet.git

# needs to be cloned to get conversion scripts
git clone https://github.com/cocodataset/panopticapi.git

cd panopticapi
python3 -m pip install .
```

[A download link for ScanNet data has to be requested from the authors](https://github.com/ScanNet/ScanNet)

A separate 25k images dataset has been sampled and processed from the sensor sequences for training models on 2D tasks. Save it to some path, e.g. your directory structure could be
```
EfficientPS
├── mmdet
├── tools
├── configs
├── data
│   ├── scannet
│   │   ├── scannet_frames_25k
```

ScanNet authors also provide a script for transforming the 2D data to COCO Panoptic format. On the other hand, EfficientPS assumes a modified version of COCO Detection format for training and evaluation. The necessary conversions can be performed with our script (this will take some time), for example:
```
python ./tools/train_val_split_to_efficientps.py \
    -s data/scannet/scannet_frames_25k/ \
    -t ../ScanNet/Tasks/Benchmark/scannetv2_train.txt \
    -v ../ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    -o data/scannet/ \
    -sc ../ScanNet/ \
    -pn ../panopticapi/
```
Check the comments [in the script](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/tools/scannet_train_val_to_efficientps.py) for further details on the conversions.

Check help for parameters:
```
python ./tools/train_val_split_to_efficientps.py --help
```

Your directory should now look something like this:
```
EfficientPS
├── ...
├── data
│   ├── scannet
│   │   ├── scannet_frames_25k
│   │   ├── scannet_panoptic
│   │   ├── scannet_panoptic_val
│   │   ├── scannet_semantic
│   │   ├── images
│   │   ├── scannet_categories.json
│   │   ├── scannet_panoptic.json
│   │   ├── scannet_panoptic_val.json
│   │   ├── scannet_panoptic_train.json
│   │   ├── scannet_detection.json
│   │   ├── scannet_detection_continuous_ids.json
│   │   ├── scannet_detection_continuous_ids_val.json
│   │   ├── scannet_detection_continuous_ids_train.json
```

Some files and directories are not necessary for training and evaluation, thus you can remove them to save space if you wish. The minimal structure for training and evaluation:
```
EfficientPS
├── ...
├── data
│   ├── scannet
│   │   ├── scannet_panoptic_val
│   │   ├── scannet_semantic
│   │   ├── images
│   │   ├── scannet_panoptic_val.json
│   │   ├── scannet_detection_continuous_ids_val.json
│   │   ├── scannet_detection_continuous_ids_train.json
```

## Run ScanNet inference
1. Download ScanNet data and export camera datafrom .sens files with
```
python tools/export_sens_data.py \
    --source_dir path/to/scannet/scenes \
    --output_dir path/to/output/directory \
    --frame_skip 1
```
- in addition to rgb, this will also export other data necessary for 3D reconstruction. You can set their booleans to False in parameters if you don't need them. Only color images are needed for segmentation.

2. Download the pre-trained checkpoint from the link above of train your own model
3. Run the inference script:
```
python tools/segment_scannet_scenes.py \
    --source path/to/scannet/scenes \
    --output path/to/output/directory \
    --config path/to/EfficientPS/config/file \
    --checkpoint path/to/EfficientPS/weights
```

Additional parameters:
- scene_input
  - define what scenes to process from source directory
  - either a json in efficientps data format (e.g. scannet_detection_continuous_ids_val.json)
  - or a .txt with one scene id in every row (e.g. [ScanNet/Tasks/Benchmark/scannetv2_val.txt](https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_val.txt))
  - or a single scene id (e.g. scene0000_00)
  - defaults to all scenes in the source directory

Booleans:
- define whether to evaluate or visualise a certain segmentation format (all default to True)
- \-\-evaluate_panoptic
-  \--evaluate_semantic
- \-\-evaluate_instance
- \-\-visualise_panoptic
- \-\-visualise_semantic
- \-\-visualise_instance
    
Check help for more information:
```
python tools/segment_scannet_scenes.py --help
```

## Evaluate on ScanNet validation data:
1. Download the pre-trained checkpoint from the link above of train your own model
2. Run the evaluation script:
```
python tools/test.py \
    configs/efficientPS_singlegpu_scannet.py \
    ${CHECKPOINT_FILE} \
    --eval panoptic
```

## Train EfficienPS on ScanNet dataset

To train on the data, edit your file structure to a config file. Examples using the file structure above can be found from [./configs](https://github.com/TUTvision/ScanNet-EfficientPS/tree/master/configs)

The model can now be trained and evaluated like explained in the original repository:

Single GPU training example:
```
python tools/train.py \
    configs/efficientPS_singlegpu_scannet.py \
    --work_dir work_dirs/scannet \
    --validate
```

Distributed training example:
```
./tools/dist_train.sh \
    configs/efficientPS_multigpu_scannet.py \
    ${GPU_NUM} \
    --work_dir work_dirs/scannet \
    --validate
```

## How to train on your own dataset

Some changes to the original EfficientPS code are required for training on a dataset other than Cityscapes or KITTI.

1. Create your own dataset class:
- Easiest way is to duplicate and edit the cityscapes class: [mmdet/datasets/cityscapes.py](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/mmdet/datasets/cityscapes.py)
- For example, we have done this in [mmdet/datasets/scannet.py](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/mmdet/datasets/scannet.py)

2. Add the new class to the dataset registry:
- modify [mmdet/datasets/__init__.py](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/mmdet/datasets/__init__.py)
- see our scannet modification for example

3. Add the evaluation classes to evaluation code:
- new enty to [mmdet/core/evaluation/class_names.py](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/mmdet/core/evaluation/class_names.py)
- also add to [mmdet/core/evaluation/__init__.py](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/mmdet/core/evaluation/__init__.py)
- see our scannet classes for example

4. Import the new classes in [mmdet/core/evaluation/panoptic.py](https://github.com/TUTvision/ScanNet-EfficientPS/blob/master/mmdet/core/evaluation/panoptic.py)

You will also need to either transform your data to the format used by EfficientPS, or implement your own dataloader. See our ScanNet conversion instructions above for an example on how to do this.
