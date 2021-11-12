# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:44:22 2020

@author: bsr
"""


import cv2 as cv
import numpy as np

img=cv.imread('res.jpg')
resized_image=cv.resize(img,(700,500))
green=resized_image.copy()
red=resized_image.copy()
green[:,:,0]=0
green[:,:,2]=0
ret,thresh1 = cv.threshold(green,127,255,cv.THRESH_TOZERO_INV)
red[:,:,0]=0
red[:,:,1]=0
ret,thresh2 = cv.threshold(red,230,255,cv.THRESH_TOZERO_INV)
total=thresh1+thresh2
cv.imwrite('output_image.jpg',total)
cv.imwrite('Green_tresh.jpg',thresh1)
cv.imwrite('Red_thresh.jpg',thresh2)
cv.waitKey(0)
cv.destroyAllWindows()