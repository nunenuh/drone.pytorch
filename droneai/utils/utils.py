import numpy as np
import cv2 as cv

def convert_to_classes_masks(mask_img, colors):
    masks = []
    h,w,c = mask_img.shape
    for color in colors:
        # canvas = np.zeros((h,w))
        label_img = mask_img == color
        label_img = label_img.astype(np.int64)
        label_img = ((label_img[:,:,0] * 255) + (label_img[:,:,1] * 255) + (label_img[:,:,2] * 255))/3
        label_img = label_img.astype(np.uint8)
        masks.append(label_img)
    masks = np.dstack(masks)
    return masks


def load_image(path, to_np=True):
    pass


def load_json( path):
    with open(path, 'r') as jsfile:
        data = json.load(jsfile)
    return data