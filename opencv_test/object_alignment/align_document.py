import numpy as np
import argparse
import imutils
import cv2

from align_images import align_images

image = cv2.imread('/home/nilim/Documents/programmer/pro/Image/NIDn/6.png')
template = cv2.imread('/home/nilim/Documents/programmer/pro/Image/NIDn/7.jpg')


aligned = align_images(image, template, debug=True)

# resize both the aligned and template images so we can easily visualize them on our screen
aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)


# our first output visualization of the image alignment will be a side-by-side comparison of the output aligned image and the template
stacked = np.hstack([aligned, template])


# our second image alignment visualization will be *overlaying* the aligned image on the template, that way we can obtain an idea of how good our image alignment is
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.imshow("Image Alignment Overlay", aligned)
cv2.waitKey(0)