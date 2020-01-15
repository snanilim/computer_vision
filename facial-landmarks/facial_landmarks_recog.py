# import the necessary packages
from imutils import face_utils
import face_recognition
import numpy as np
import argparse
import imutils
import dlib
import cv2
 
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

args = {
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'face_recognition': 'dlib_face_recognition_resnet_model_v1.dat',
    'image': 'images/farhan.jpg',
    'image2': 'images/50.jpg'
}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
face_rec = dlib.face_recognition_model_v1(args["face_recognition"])


image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

image2 = cv2.imread(args["image2"])
image2 = imutils.resize(image2, width=500)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
rects2 = detector(gray2, 1)

for (i, rect2) in enumerate(rects2):
    print('rect', i)
    shape2 = predictor(gray2, rect2)
    
    face_descriptor2 = face_rec.compute_face_descriptor(image2, shape2, 1)

for (i, rect) in enumerate(rects):
    print('rect', i)
    shape = predictor(gray, rect)
    
    face_descriptor = face_rec.compute_face_descriptor(image, shape, 1)
    # face_descriptor = face_recognition.face_descriptor()
    shape = face_utils.shape_to_np(shape)

    match = list((np.linalg.norm([np.array(face_descriptor2)] - (np.array(face_descriptor)), axis=1)))
    print('match', match)
    val, idx = min((val, idx) for (idx, val) in enumerate(match))
    print(1-val, idx)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

print(args)

cv2.imshow("Output", image)
cv2.waitKey(0)