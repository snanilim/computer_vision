# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image file")
args = vars(ap.parse_args())


# load the image from disk
image = cv2.imread(args["image"])
# loop over the rotation angles
for angle in np.arange(0, 360, 15):
    # print('angle', angle)
    rotated = imutils.rotate(image, angle)
    cv2.imshow("Rotated (Problematic)", rotated)
    cv2.waitKey(0)
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(0)



# def rotate_bound(image, angle):
#     # grab the dimensions of the image and then determine the
#     # center
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#     # perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (nW, nH))