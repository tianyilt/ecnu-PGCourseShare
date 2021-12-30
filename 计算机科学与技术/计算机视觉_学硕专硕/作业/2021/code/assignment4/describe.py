import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
# https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html

img = cv.imread('b_02.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# https://www.cnblogs.com/xingnie/p/10230278.html
# 需要根据版本装包,这里用 pip install opencv-contrib-python==3.4.2.16
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints2.jpg',img)

surf = cv.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(img,None)
img2 = cv.drawKeypoints(img,kp,None,(255,0,0))
cv.imwrite('surf_keypoints.jpg',img2)