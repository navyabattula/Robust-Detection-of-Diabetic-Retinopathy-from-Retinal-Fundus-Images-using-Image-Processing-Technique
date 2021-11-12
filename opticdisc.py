# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:55:39 2020

@author: bsr
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

def resiz(img):
    width = 1024
    height = 720
    #####
    dsize=(width,height)
    return cv2.resize(img,dsize,interpolation = cv2.INTER_CUBIC)
    

def rgb2Red(img):
    b,g,r = cv2.split(img)
    return r

def rgb2Green(img):
    b,g,r = cv2.split(img)
    return g

def rgb2Gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def preprocess(img):
    #b,g,r = cv2.split(img)
    gray = rgb2Red(img)
    cv2.imshow('input' ,gray)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0, gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    gray = ndimage.grey_closing(gray,structure=kernel)
    gray = cv2.equalizeHist(gray)  
    return gray

def getROI(image):
    image_resized = resiz(image)
    b,g,r = cv2.split(image_resized)
    g = cv2.GaussianBlur(g,(15,15),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    g = ndimage.grey_opening(g,structure=kernel)    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

    x0 = int(maxLoc[0])-110
    y0 = int(maxLoc[1])-110
    x1 = int(maxLoc[0])+110
    y1 = int(maxLoc[1])+110
    image = image_resized[y0:y1,x0:x1]
    return [image,x0,y0]

def cann(img,sigma):
    v = np.mean(img)
    sigma = sigma
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)    
    return edged

def hough(edged,limm,limM):
    hough_radii = np.arange(limm, limM, 1)
    hough_res = hough_circle(edged, hough_radii)
    return hough_circle_peaks(hough_res, hough_radii,total_num_peaks=1)

image = cv2.imread("a5.jpg")
q = resiz(image)
cv2.imshow('input' ,q)
cv2.imwrite("q.jpg",q)
roi,x0,y0 = getROI(image)
preprocessed_roi = preprocess(roi)

edged = cann(preprocessed_roi,0.22)
kernel = np.ones((3,3),np.uint8)
edged1 = cv2.dilate(edged,kernel,iterations=3)
accums, cx, cy, radii = hough(edged,55,80)
cv2.circle(roi, (cx,cy), radii+10, (0, 0, 0), -1)

cv2.imshow('Region of Interest ',roi)
cv2.imwrite("outputs1.jpg",roi)
cv2.imshow('R ',edged)
cv2.imwrite("outputs2.jpg",edged)
cv2.imshow('R ',edged1)
cv2.imwrite("outputs3.jpg",edged1)
cv2.imshow('R ',preprocessed_roi)
cv2.imwrite("outputs4.jpg",preprocessed_roi)
im1=Image.open("q.jpg")
im2=Image.open("outputs1.jpg")
im1.paste(im2,(x0,y0))
im1.save("res.jpg")
im1 = cv2.imread("res.jpg")
cv2.imshow('output' ,im1)