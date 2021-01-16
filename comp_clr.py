from matplotlib.image import imread 
import os
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
A=imread('C:\\Users\Danish Shabbir\\Desktop\\dani.jpg')
plt.imshow(A)
img=np.mean(A,-1)
bt=np.fft.fft2(img)
btsort=np.sort(np.abs(bt.reshape(-1)))
for keep in (0.1,0.05,0.01,0.002):
    thresh=btsort[int(np.floor((1-keep)*len(btsort)))]
    sml_val=np.abs(bt)>thresh
    thres_low=bt*sml_val
    rev_frir=np.fft.ifft2(thres_low).real
    plt.figure()
    plt.imshow(rev_frir,cmap='gray')
    plt.axis('off')
    plt.title(' Total part of image in percentage='+str(keep*100)+'%')
#Image Resizing by using Different filters
img = cv2.imread("C:\\Users\Danish Shabbir\\Desktop\\dani.jpg")
img = cv2.resize(img, (0, 0), None, .25, .25)
g_Blur = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32)/9
shrpn = np.array(([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]), np.float32)/9
m_Blur = np.ones((3, 3), np.float32)/9
gaussian_Blur = cv2.filter2D(src=img, kernel=g_Blur, ddepth=-1)
mean_Blur = cv2.filter2D(src=img, kernel=m_Blur, ddepth=-1)
sharpen = cv2.filter2D(src=img, kernel=shrpn, ddepth=-1)

h_Stack = np.concatenate((img, gaussian_Blur, mean_Blur, sharpen), axis=1)

cv2.imshow("2D Convolution Example", h_Stack)

cv2.waitKey(0)
cv2.destroyAllWindows()