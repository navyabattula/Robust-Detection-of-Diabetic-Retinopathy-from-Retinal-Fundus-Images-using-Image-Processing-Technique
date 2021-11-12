# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:52:36 2020

@author: Navya
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import black_tophat,disk
def plot_comparison(filtered, filter_name):
    fig,ax1 = plt.subplots(ncols=1, figsize=(8, 4), sharex=True,sharey=True)
    ax1.imshow(filtered, cmap=plt.cm.gray)
    ax1.set_title(filter_name)
    ax1.axis('off')
counter=0

image_in_file= cv.imread('a2.jpg',0)
resized_image = cv.resize(image_in_file, (500, 500)) 
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(resized_image) 
eroded_image=cv.erode(clahe_image,None,iterations=3)
dilated_image=cv.dilate(eroded_image,None,iterations=3)
structuring_element=disk(7)
black_tophat_image = black_tophat(dilated_image, structuring_element)
plot_comparison(black_tophat_image, 'black tophat')
clahe=cv.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
black_top_hat1=clahe.apply(black_tophat_image)
#output_image="Base11_outputs_"+str(counter)+".png"
cv.imwrite('outputn.jpg',black_tophat_image)
cv.imwrite('clahe.jpg',clahe_image)
cv.imwrite('eroded.jpg',eroded_image)
cv.imwrite('dilated.jpg',dilated_image)
counter=counter+1
cv.waitKey(0)
cv.destroyAllWindows()
