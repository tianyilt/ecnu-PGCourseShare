import cv2
import math
img=cv2.imread("1.png",cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

retval,dst = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
img,contours,heridency =cv2.findContours(dst,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
a = cv2.moments(contours[0],binaryImage=False)
mc=[a['m10']/a['m00'],a['m01']/a['m00']]
angle= math.atan2(2*a['m11'],a['m20']-a['m02'])/2

area=a['m00']
center_axis=angle*180/math.acos(-1)#radium

print("area:{} center_axis:{} mc:{}".format(area,center_axis,mc))
cv2.drawContours(img,contours,-1,(255,0,255),3)
cv2.imshow("hehe",img)
cv2.waitKey(0)
