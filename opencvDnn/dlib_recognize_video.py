from imutils.video import VideoStream
from imutils import paths
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from align_video_faces import face_align
import face_recognition

args = {
    'detector': '../model/face_detection_model', 
    'embedding_model': '../model/openface_nn4.small2.v1.t7', 
    'recognizer': 'output/recognizer.pickle', 
    'le': 'output/le.pickle', 
	'confidence': 0.5
}

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

recognizer = pickle.loads(open(args["recognizer"], "rb").read())

le = pickle.loads(open(args["le"], "rb").read())

embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# cap = cv2.VideoCapture("/home/nilim/Documents/programmer/backup/record.avi")
cap = cv2.VideoCapture(2)

if(cap.isOpened == False):
	print("Error opening video stream or file")

while(cap.isOpened):
	ret, frame = cap.read()

	if ret == True:
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=750)
		r = frame.shape[1] / float(rgb.shape[1])
		boxes = face_recognition.face_locations(rgb, model="hog")

		if len(boxes) > 0:
			for box in boxes:
				print('box', box)
				(top, right, bottom, left) = box

				vec = face_recognition.face_encodings(rgb, [box])
				preds = recognizer.predict_proba(vec)[0]
				print('preds', preds)
				j = np.argmax(preds)
				print(j)
				proba = preds[j]

				name = "unknown"
				if proba > 0.8:
					name = le.classes_[j]
					color = (0, 255, 0)

				if name == "unknown":
					color = (0, 0, 255)

				top = int(top * r)
				right = int(right * r)
				bottom = int(bottom * r)
				left = int(left * r)

				text = "{}: {:.2f}%".format(name, proba * 100)
				cv2.rectangle(frame, (left, top), (right, bottom),(color), 2)
				y = top - 15 if top - 15 > 15 else top + 15
				cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (color), 2)

		cv2.imshow("Frame", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break


cap.release()
# out.release()

cv2.destroyAllWindows()