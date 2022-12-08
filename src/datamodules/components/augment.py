import os
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import string
import numpy as np

import cv2
from PIL import Image

import augly
import augly.image as imaugs
import albumentations as A

from src.datamodules.components.aug_functional import *



class Augmentation:
    def __init__(self,
                 n_augments: int,
                 bg_image_dir: str,
                 strong_augment: bool = True,
                 ):
        
        self.n_augments = n_augments
        
        geometric = [A.HorizontalFlip(p = 1),
                     A.VerticalFlip(p = 1),
                     A.Perspective(p = 1),
                     A.ShiftScaleRotate(shift_limit = 0.3, rotate_limit = 90, p = 1),
                     A.RandomScale(p = 1),
                     A.CropAndPad(percent = [-0.95, -0.45]),
                     A.RandomCropFromBorders(crop_left = 0.01, crop_right = 0.01, crop_top = 0.01, crop_bottom = 0.01),
                     ]
        
        photometric = [ApplyIGFilter(),
                       A.ToSepia(p = 1),
                       A.ToGray(p = 1),
                       A.Solarize(p = 1),
                       A.RandomBrightnessContrast(brightness_limit = 0.5, contrast_limit = 0.5, p = 1),
                       A.HueSaturationValue(p = 1),
                       A.ColorJitter(p = 1),
                       A.RandomGamma(p = 1),
                       A.RandomSunFlare(p = 1),
                       A.InvertImg(p = 1),
                       ]
        
        overlay_over_source = [OverlayRandomText(),
                               OverlayRandomEmoji(),
                               OverlayRandomStripes(),
                               MemeRandomFormat(),
                               ]
        
        compose_with_other = [OverlayOntoRandomScreenshot(),
                              OverlayOntoRandomBackgroundImage(bg_image_dir),
                              OverlayOntoRandomForegroundImage(bg_image_dir),
                              BlendImage(bg_image_dir),
                              A.InvertImg(p = 1), # InvertImg added here for higher probabilty
                              ]
        
        pixel_transform = [A.GaussNoise(p = 1),
                           A.MultiplicativeNoise(p = 1),
                           A.ChannelDropout(p = 1),
                           A.ZoomBlur(p = 1),
                           A.MedianBlur(p = 1),
                           A.RandomToneCurve(p = 1),
                           A.Posterize(p = 1),
                           A.RGBShift(p = 1),
                           A.Defocus(p = 1),
                           A.RingingOvershoot(p = 1),
                           A.ISONoise(p = 1),
                           A.OpticalDistortion(p = 1),
                           A.Downscale(p = 1, interpolation = cv2.INTER_AREA),
                           A.JpegCompression(p = 1.0),
                           ]
        
        if strong_augment:
            self.augments = geometric + photometric + overlay_over_source + compose_with_other + pixel_transform
            
            augment_prob =  [(0.2/1)] * len(geometric) + \
                            [(0.2/1)] * len(photometric) + \
                            [(0.33/1)] * len(overlay_over_source) + \
                            [(0.33/1)] * len(compose_with_other) + \
                            [(0.2/1)] * len(pixel_transform)
                            
            self.augment_prob = np.array([(p/sum(augment_prob)) for p in augment_prob])
        else:
            self.augments = geometric
            self.augment_prob = np.array([1/len(self.augments)]*len(self.augments))
                        
    def __call__(self,image):
        transforms = np.random.choice(self.augments,
                                      size = self.n_augments,
                                      replace = False,
                                      p = self.augment_prob)

        for transform in transforms:
            try:
                image = transform(image = image)
                if type(image) == dict:
                    image = image['image']
            except:
                print(f'Unable to apply transformation: {transform}')
        return image
    
    
    
    