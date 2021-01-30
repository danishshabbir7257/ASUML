import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("C:\\Users\Danish Shabbir\\Desktop\\dani.png",0)
img_pl=plt.imread("C:\\Users\Danish Shabbir\\Desktop\\dani.png")
plt.hist(img.ravel(),256,[0,256]) 
plt.show()
print("image in Compuer form")
print(img_pl)
edges = cv2.Canny(img,40,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
