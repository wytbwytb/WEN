# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from fsdet.structures import BoxMode
from fsdet.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_rfs"]


# fmt: off
CLASS_NAMES = [
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
        'spray_alcohol',
        'portable_charger_1',
        'utility_knife', 
        'mobile_phone',
        'metal_can',
        'drink_bottle'
    ]
# fmt: on

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

def load_rfs_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "annotations", fileid + ".txt")
        jpeg_file = os.path.join(dirname, "images", fileid + ".jpg")

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
            
            bbox = obj['bbox']

            instances.append({
                "category_id": CLASS_NAMES.index(cls),
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
            })
           
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_rfs(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_rfs_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )
