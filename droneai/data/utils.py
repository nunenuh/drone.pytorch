import json
from PIL import Image
import tifffile as tiff

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json( path):
    with open(path, 'r') as jsfile:
        data = json.load(jsfile)
    return data

def load_image(path, to_np=False):
    img = Image.open(path)
    if to_np:
        img = np.array(img)
    return img

def load_tiff(path):
    img = tiff.imread(path)
    return img

def class2idx():
    data = {
        "unlabeled": 0, "paved-area": 1, "dirt": 2,
        "grass": 3, "gravel": 4, "water": 5,
        "rocks": 6, "pool": 7, "vegetation": 8,
        "roof": 9, "wall": 10, "window": 11,
        "door": 12, "fence": 13, "fence-pole": 14,
        "person": 15, "dog": 16, "car": 17,
        "bicycle": 18, "tree": 19, "bald-tree": 20,
        "ar-marker": 21, "obstacle": 22, "conflicting": 23
    }
    return data

def idx2class():
    data = [
        'unlabeled','paved-area','dirt',
        'grass','gravel','water',
        'rocks','pool','vegetation',
        'roof','wall','window',
        'door','fence','fence-pole',
        'person','dog','car',
        'bicycle','tree','bald-tree',
        'ar-marker','obstacle','conflicting'
    ]
    return data

def visualize(image, mask, alpha_val=0.6, figsize_val=(10,10)):
    if type(image) == str:
        img = load_image(image)
    if type(mask) == str:
        mask = load_tiff(mask)
    
    print('Image Size', np.asarray(image).shape)
    print('Mask Size', np.asarray(mask).shape)

    plt.figure(figsize=figsize_val)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha_val)
    plt.title('Picture with Mask Appplied')
    plt.show()