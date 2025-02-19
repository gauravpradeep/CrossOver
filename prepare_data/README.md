# Dataset Preparation

## Overview

This document provides instructions for pre-processing different datasets, including 
- ScanNet
- 3RScan
- ARKitScenes

## Prerequisites

### Environment
Before you begin, simply activate the `crossover` conda environment.

### Download the Original Datasets
- **ScanNet**: Download ScanNet v2 data from the [official website](https://github.com/ScanNet/ScanNet).

- **3RScan**: Download 3RScan dataset from the [official website](https://github.com/WaldJohannaU/3RScan).

- **ARKitScenes**: Download ARKitScenes dataset from the [official website](https://github.com/apple/ARKitScenes).

- **ShapeNet**: Download Shapenet dataset from the [official website](https://shapenet.org/) and unzip.

### Download Referral and CAD annotations
We use [SceneVerse](https://scene-verse.github.io/) for instance referrals (ScanNet, 3RScan, & ARKitScenes) and [Scan2CAD](https://github.com/skanti/Scan2CAD) for CAD annotations (ScanNet). Exact instructions for data setup below.

#### ScanNet
1. Run the following to extract ScanNet data 
```bash
cd scannet
python preprocess_2d_scannet.py --scannet_path PATH_TO_SCANNET --output_path PATH_TO_SCANNET
python unzip_scannet.py --scannet_path PATH_TO_SCANNET --output_path PATH_TO_SCANNET
```

2. To have a unified structure of objects `objects.json` like provided in `3RScan`, run the following:

```bash
cd scannet
python scannet_objectdata.py
```

> Change `base_dataset_dir` to `Scannet` dataset root directory.

2. Move the relevant files from `Sceneverse` and `Scannet` under `files/`. Once completed, the data structure would look like the following:

```
Scannet/
├── scans/
│   ├── scene0000_00/
│   │   ├── data/
│   │   │    ├── color/
│   │   |    ├── depth/
|   |   |    ├── instance-filt/
│   │   |    └── pose/
|   |   ├── intrinsics.txt
│   │   ├── scene0000_00_vh_clean_2.ply 
|   |   ├── scene0000_00_vh_clean_2.labels.ply
|   |   ├── scene0000_00_vh_clean_2.0.010000.segs.json
|   |   ├── scene0000_00_vh_clean.aggregation.json
|   |   └── scene0000_00_2d-instance-filt.zip
|   └── ...
└── files
    ├── scannetv2_val.txt
    ├── scannetv2_train.txt
    ├── scannetv2-labels.combined.tsv
    ├── scan2cad_full_annotations.json
    ├── objects.json
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```

#### 3RScan

1. Run the following to align the re-scans and reference scans in the same coordinate system & unzip `sequence.zip` for every scan:

```bash
cd scan3r
python align_scan.py  (change `root_scan3r_dir` to `PATH_TO_SCAN3R`)
python unzip_scan3r.py --scan3r_path PATH_TO_SCAN3R --output_path PATH_TO_SCAN3R
```

2. Move the relevant files from `Sceneverse` and `3RScan` under `files/`.

Once completed, the data structure would look like the following:

```
Scan3R/
├── scans/
│   ├── 20c993b5-698f-29c5-85a5-12b8deae78fb/
│   │   ├── sequence/ (folder containing frame-wise color + depth + pose information)
|   |   ├── labels.instances.align.annotated.v2.ply
│   │   └── labels.instances.annotated.v2.ply
|   └── ...
└── files
    ├── 3RScan.json
    ├── 3RScan.v2 Semantic Classes - Mapping.csv
    ├── objects.json
    ├── train_scans.txt
    ├── test_scans.txt
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```

#### ARKitScenes
1. Download `files/` under `processed_data/meta_data/ARKitScenes/` from GDrive and place under `PATH_TO_ARKITSCENES/`.
2. Download ARKitScenes 3dod data into ARKitScenes/scans and run the following to extract MultiScan data 
 
 ```bash
cd ARKitScenes
mv 3dod/Training/* scans
mv 3dod/Validation/* scans
```

Once completed, the data structure would look like the following:
```
MultiScan/
├── scans/
│   ├── 40753679/
│   │   ├── 40753679_frames/ 
│   │   │    ├── lowres_depth/ (folder containing depth images)
│   │   │    ├── lowres_wide/ (folder containing rgb images)
│   │   │    ├── lowres_wide_intrinsics/ (folder containing frame wise camera intrinsics)
│   │   │    ├── lowres_wide.traj (camera trajectory)
│   │   ├── 40753679_3dod_annotation.json
│   │   ├── 40753679_3dod_mesh.ply
|   └── 
└── files
    ├── scannetv2-labels.combined.tsv
    ├── train_scans.txt
    ├── val_scans.txt
    ├── metadata.csv
    ├── 3dod_train_val_splits.csv
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```