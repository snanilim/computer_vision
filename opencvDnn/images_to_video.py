import cv2
import numpy as np
import glob
from imutils import paths
 
img_array = []
imagePaths = list(paths.list_images("/home/nilim/Documents/programmer/backup/dataset/for_video"))

for (i, imagePath) in enumerate(imagePaths):
    img = cv2.imread(imagePath)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('/home/nilim/Documents/programmer/backup/multiple.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# stream = cv2.VideoCapture("project.avi")

# while True:
# 	(grabbed, frame) = stream.read()

# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(0) & 0xFF

# 	if key == ord("q"):
# 		break

# stream.release()