"""VOC Dataset Classes
Original author: Yonghye Kwon
https://github.com/developer0hye
"""
import cv2
import numpy as np

import os
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = (
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor")
        
def read_gt_file_for_evaluation(label_path, keep_difficult):
    label = []
    
    tree = ET.parse(label_path)
    root = tree.getroot()
    
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    
    objs = root.findall("object")
    for obj in objs:
        difficult = int(obj.find("difficult").text)
        
        if difficult == 1 and keep_difficult == False:
            continue

        class_name = obj.find("name").text.lower().strip()
        
        if not class_name in CLASSES:
            continue
        
        class_idx = CLASSES.index(class_name)

        bbox = obj.find("bndbox")
        
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        label.append([class_idx, xmin, ymin, xmax, ymax])

    label = np.array(label).reshape(-1, 5)
    return label

import os
os.makedirs("gt", exist_ok=True)

root = "./dataset/VOCdevkit"
image_sets = [("2007", "test")]
keep_difficult = False
vis = False

for (year, name) in image_sets:
    rootpath = os.path.join(root, "VOC" + year)
    with open(os.path.join(rootpath, "ImageSets", "Main", name + ".txt")) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            
            img_path = os.path.join(rootpath, "JPEGImages", line + ".jpg")
            label_path = os.path.join(rootpath, "Annotations", line + ".xml")
            
            if vis: img = cv2.imread(img_path)
            label = read_gt_file_for_evaluation(label_path, keep_difficult)
            
            eval_label_path = os.path.join("gt", Path(label_path).stem + ".txt")
            with open(eval_label_path, 'w') as f:
                for bbox in label:
                    f.write(f"{CLASSES[bbox[0]]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    if vis: cv2.rectangle(img, (bbox[1], bbox[2]), (bbox[3], bbox[4]), (0, 255, 0), 3)
            
            if vis:
                cv2.imshow("img", img)
                cv2.waitKey(0)
