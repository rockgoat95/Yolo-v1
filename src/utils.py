##utils

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import xml.etree.ElementTree as Et

def image_read(img_path, annot_path, number):
    annot_file = os.listdir(annot_path)[number]
    filename = annot_file.split(".")[0]+".jpg"
    print(filename)
    image = cv2.imread(os.path.join(img_path,filename))
    
    img = image.copy()
    xml =  open(os.path.join(annot_path, annot_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    
    # size = root.find('size')
    # width = size.find('width').text
    # height = size.find('height').text
    # channels = size.find('depth').text
    objects = root.findall("object")
    for _object in objects:
        name = _object.find('name').text
        bndbox = _object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0), 2)
    plt.figure()
    plt.imshow(img)
    plt.show()

    return image

def filtering_and_non_max_suppression(decoded_boxes,object_class , overlapThresh =0.2, score_Thresh =0.5):
    
    pick = []
    nms_labels =[]
    n = len(decoded_boxes)
    label_operator = LabelBinarizer()
    label_operator.fit(object_class)
    labels = label_operator.classes_[np.argmax(decoded_boxes[:,5:], axis = 1)]

    max_class_idx = np.argmax(decoded_boxes[:,5:], axis = 1)
    scores = [decoded_boxes[:,5:][i,max_class_idx[i]] for i in range(n)]

    scores = scores*decoded_boxes[:,0]

    for i, cl_name in enumerate(object_class):
        cls_idx = np.array([i for i in range(n) if labels[i] == cl_name and scores[i] >score_Thresh])
        if len(cls_idx)== 0 :
            continue
        Rem_idx = cls_idx
        while len(Rem_idx)>0:
            cl_score = scores[Rem_idx]
            temp_best_idx = Rem_idx[np.argmax(cl_score)]
            pick.append(temp_best_idx)
            nms_labels.append(labels[temp_best_idx])
            Rem_idx = np.setdiff1d(Rem_idx, temp_best_idx)
            for i in Rem_idx:
                if get_iou(decoded_boxes[temp_best_idx],decoded_boxes[i]) >overlapThresh:
                    Rem_idx = np.setdiff1d(Rem_idx, i)
    return pick , nms_labels
    
def get_iou(bb1, bb2):

    x_left = max(bb1[1] -bb1[3]/2, bb2[1]-bb2[3]/2)
    y_top = max(bb1[2] -bb1[4]/2, bb2[2]-bb2[4]/2)
    x_right = min(bb1[1] +bb1[3]/2, bb2[1]+bb2[3]/2)
    y_bottom = min(bb1[2] +bb1[4]/2, bb2[2]+bb2[4]/2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou