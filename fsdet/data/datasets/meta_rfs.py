# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from fsdet.structures import BoxMode
from fsdet.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_rfs"]

RFS_NOVEL_CATEGORIES = {
     1: ["portable_charger_1", "utility_knife", "mobile_phone", "metal_can", "drink_bottle"],
     2: ['portable_charger_1','multi-tool_knife','metal_cup','pressure_tank','spray_alcohol'],
     3: ['laptop','multi-tool_knife','glass_bottle','metal_cup','nail_clippers',]
}

RFS_BASE_CATEGORIES = {
    1: [
        'laptop',
        'lighter',
        'portable_charger_2',
        'iron_shoe',
        'straight_knife',
        'folding_knife', 
        'scissor',
        'multi-tool_knife',
        'umbrella',
        'glass_bottle',
        'battery',
        'metal_cup',
        'nail_clippers',
        'pressure_tank',
        'spray_alcohol'
    ],
    2: [
        'laptop',
        'lighter',
        'portable_charger_2',
        'mobile_phone',
        'folding_knife', 
        'utility_knife',
        'drink_bottle',
        'glass_bottle',
        'iron_shoe',
        'metal_can',
        'nail_clippers',
        'straight_knife',
        'scissor', 
        'battery',
        'umbrella'
    ],
    3:[
        'lighter',
        'portable_charger_2',
        'mobile_phone',
        'folding_knife', 
        'utility_knife',
        'drink_bottle',
        'iron_shoe',
        'metal_can',
        'pressure_tank',
        'spray_alcohol',
        'portable_charger_1',
        'straight_knife',
        'scissor', 
        'battery',
        'umbrella'
    ]
}

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    # filename = filename[:-3] + 'txt'

    filename = filename.replace('.xml', '.txt')
    imagename0 = filename.replace('annotation', 'image')
    # imagename1 = imagename0.replace('.txt', '.TIFF')  # jpg form
    # imagename2 = imagename0.replace('.txt', '.tiff')
    imagename3 = imagename0.replace('.txt', '.jpg')
    objects = []
    #print(filename,imagename0,imagename3)
    # img = cv2.imread(imagename1)
    # if img is None:
    #     img = cv2.imread(imagename2)
    # if img is None:
    img = cv2.imread(imagename3)
    
    height, width, channels = img.shape
    with open(filename, "r", encoding='utf-8') as f1:
        dataread = f1.readlines()
        #print(dataread)
        for annotation in dataread:
            obj_struct = {}
            temp = annotation.split()
            name = temp[1].strip()
            rate = float(temp[-1].strip())
            # if name != 'Portable_Charger_1' and name != 'Portable_Charger_2'and name != 'Mobile_Phone'and name != 'Cosmetic'and name != 'Nonmetallic_Lighter'and name != 'Water'and name != 'Tablet'and name != 'Laptop':
            #     continue
            xmin = int(temp[2])

            if int(xmin) > width:
                continue
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:
                xmax = width - 1
            ymax = int(temp[5])
            if ymax > height:
                ymax = height - 1
            ##name
            obj_struct['name'] = name
            obj_struct['pose'] = 'Unspecified'
            obj_struct['truncated'] = 0
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [float(xmin) - 1,
                                  float(ymin) - 1,
                                  float(xmax) - 1,
                                  float(ymax) - 1]
            obj_struct['rate'] = rate
            objects.append(obj_struct)

    return objects, height, width

def load_filtered_rfs_instances(
    name: str, dirname: str, split: str, classnames: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"

    Args example:
        name: voc_2007_trainval_all1_1shot
        dirname: VOC2007 / VOC2012
        split: novel_10shot_split_3_trainval
    """
    use_more_base = 'ploidy' in name
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("/media/datasets/RFS/train", "split")

        if use_more_base:
            ploidy = name.split('_')[-1]
            split_id = name.split('_')[3][-1]
            split_dir = os.path.join(split_dir, ploidy, 'split{}'.format(split_id))
            shot = name.split('_')[-3].split('shot')[0]
            seed = int(name.split('_')[-2].replace('seed', ''))
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            if "seed" in name:
                shot = name.split('_')[-2].split('shot')[0]
                seed = int(name.split('_seed')[-1])
                split_dir = os.path.join(split_dir, "seed{}".format(seed))
            else:
                shot = name.split('_')[-1].split('shot')[0]

        for cls in classnames:
            with PathManager.open(os.path.join(split_dir,
                    "box_{}shot_{}_train.txt".format(shot, cls))) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [fid.split('/')[-1].split('.jpg')[0] \
                                for fid in fileids_]
                fileids[cls] = fileids_

            # fileids 确实是进入了了 3ploidy
            # if cls == 'car':
            #     import pdb; pdb.set_trace()
            #     print(os.path.join(split_dir,
            #         "box_{}shot_{}_train.txt".format(shot, cls)))
            #     print(fileids[cls])
    else:
        with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                #year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join("/media/datasets/RFS/train")
                anno_file = os.path.join(dirname, "annotations", fileid + ".txt")
                jpeg_file = os.path.join(dirname, "images", fileid + ".jpg")

                #tree = ET.parse(anno_file)

                objs, height, width = parse_rec(anno_file)

                for obj in objs:
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": height,
                        "width": width,
                    }
                    cls_ = obj['name']
                    if cls != cls_:
                        continue
                    bbox = obj['bbox']
                    # bbox = obj.find("bndbox")
                    # bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    # bbox[0] -= 1.0
                    # bbox[1] -= 1.0

                    instances = [{
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS
                    }]
                    r["annotations"] = instances
                    dicts_.append(r)

            # this make sure that dataset_dicts has *exactly* K-shot
            if use_more_base and cls in RFS_BASE_CATEGORIES[int(split_id)]:
                if len(dicts_) > int(shot) * int(ploidy[0]):
                    dicts_ = np.random.choice(dicts_, int(shot)*int(ploidy[0]), replace=False)
            else:
                if len(dicts_) > int(shot):
                    dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            anno_file = os.path.join(dirname, "annotations", fileid + ".txt")
            jpeg_file = os.path.join(dirname, "images", fileid + ".jpg")

            # tree = ET.parse(anno_file)

            objs, height, width = parse_rec(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": height,
                "width": width,
            }
            instances = []

            for obj in objs:
                cls = obj["name"]
                if not (cls in classnames):
                    continue
                bbox = obj['bbox']

                instances.append({
                    "category_id": classnames.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                })
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_rfs(
    name, metadata, dirname, split, year, keepclasses, sid):
    if keepclasses.startswith('base_novel'):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith('base'):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith('novel'):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name, lambda: load_filtered_rfs_instances(
            name, dirname, split, thing_classes)
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes, dirname=dirname, year=year, split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid]
    )
