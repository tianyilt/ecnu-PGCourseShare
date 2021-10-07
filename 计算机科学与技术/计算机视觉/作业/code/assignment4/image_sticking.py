import cv2
import imutils
# https://docs.opencv.org/4.3.0/d1/de0/tutorial_py_feature_homography.html
# names = ['b_01.jpg', 'b_03.jpg']
# names = ['a_01.jpg','a_02.jpg', 'a_03.jpg']
names = ['1.jpg','2.jpg']
# names = ['9.jpg','10.jpg']
images = []
for name in names:
    image = cv2.imread(name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

print("status:",status)
if status==0:
    cv2.imwrite('stitch.jpg', stitched)