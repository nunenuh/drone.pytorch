import numpy as np
import cv2 as cv
import torch


class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class PairResize:
    def __init__(self, dim, interpolation=cv.INTER_AREA):
        self.dim = dim
        self.inter = interpolation
    
    def __call__(self, image:np.ndarray, mask:np.ndarray):
        image_result = cv.resize(image, self.dim, interpolation=self.inter)
        mask_result = cv.resize(mask, self.dim, interpolation=self.inter)
        return image_result, mask_result