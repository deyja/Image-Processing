import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("uzay.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img_gray,90,255,cv2.THRESH_BINARY)
cv2.imshow("Thresh img",img_thresh)


kernel = np.ones((5,5),dtype=np.uint8)
img_erode = cv2.erode(img_thresh,kernel,iterations=1)    # erozyon
img_dilate = cv2.dilate(img_thresh,kernel,iterations=1)  # genişleme
cv2.imshow("Erode img",img_erode)
cv2.imshow("Dilate img",img_dilate)


img_opening = cv2.dilate(img_erode,kernel,iterations=1)     # açma (erozyon + genişletme)
img_closing = cv2.erode(img_dilate,kernel,iterations=1)     # kapatma (genişletme + erozyon)
alternative_img_opening = cv2.morphologyEx(img_thresh,cv2.MORPH_OPEN,kernel)    # açma (erozyon + genişletme)
alternative_img_closing = cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,kernel)   # kapatma (genişletme + erozyon)
cv2.imshow("Open img",img_opening)
cv2.imshow("Close img",img_closing)


img_gradiant = img_dilate - img_erode
cv2.imshow("Gradiant img",img_gradiant)

cv2.waitKey(0)