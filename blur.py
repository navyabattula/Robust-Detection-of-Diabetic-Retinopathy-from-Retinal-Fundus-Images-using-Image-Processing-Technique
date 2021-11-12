# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:29:12 2020

@author: Navya
"""

import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import math
from skimage import exposure,filters
from matplotlib import pyplot as plt
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import argparse
from PIL import Image
image = cv2.imread("res.jpg")
gray_blur = cv2.blur(image, (10,10), 0)
gray_blur = cv2.GaussianBlur(gray_blur, (5,5), 0)
gray_blur = cv2.medianBlur(gray_blur, 5)
gray_blur = cv2.GaussianBlur(gray_blur, (5,5), 0)
cv2.imwrite("outputn12.jpg",gray_blur)
