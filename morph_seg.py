import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2
#morphological processing
img = cv2.imread("C:\\Users\Danish Shabbir\\Desktop\\dani.jpg")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
mask = mask0+mask1
k=np.ones((3,3),np.uint8)
d = cv2.dilate(mask, k, iterations=2)
e = cv2.erode(mask, k, iterations=1)
g = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)
imges=[img,mask,d,e,g]
titles=["original","mask","Dilation","erosion    ","Morph_gradiant"]
for i in range(5):
    plt.subplot(1, 5, i+1), plt.imshow(imges[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
#Segmentation
img_clr=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gry=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,trsh=cv2.threshold(gry,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,trsh2=cv2.threshold(gry,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
print(ret)
plt.figure("otsu")
plt.imshow(trsh,cmap="gray")
plt.figure("Triangle")
plt.imshow(trsh2,cmap="gray")