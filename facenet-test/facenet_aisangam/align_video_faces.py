from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2


def face_align(img):
	args = {
		'shape_predictor': '../model/shape_predictor_68_face_landmarks.dat'
	}

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	fa = FaceAligner(predictor, desiredFaceWidth=256)

	# image = cv2.imread(img)
	image = imutils.resize(img, width=300)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# cv2.imshow("Input", image)
	# cv2.waitKey(0)
	rects = detector(gray, 2)

	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rect)

		# print()

		return faceAligned

		# cv2.imwrite("../facial-landmarks/images/501" + ".jpg", faceAligned)

		# # display the output images
		# cv2.imshow("Original", faceOrig)
		# cv2.imshow("Aligned", faceAligned)
		# cv2.waitKey(0)
