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

cap = cv2.VideoCapture("/home/nilim/Documents/programmer/backup/record.avi")
# cap = cv2.VideoCapture(0)

if(cap.isOpened == False):
	print("Error opening video stream or file")

while(cap.isOpened):
	ret, frame = cap.read()

	if ret == True:
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		
		detector.setInput(imageBlob)
		detections = detector.forward()
		print('yah')
		count_face = 0
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=750)
		boxes = face_recognition.face_locations(rgb,model="hog")
		print('dlib', boxes)
		for i in range(0, detections.shape[2]):

			confidence = detections[0, 0, i, 2]
			# print('ok')
			if confidence > args["confidence"]:
				count_face += 1
				print('count_face', count_face)

				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				print('box', (startX, startY, endX, endY))

				face_area = [(startX, startY, endX, endY)]

				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				if fW < 20 or fH < 20:
					continue
				
				name = "unknown"
				color = (0, 0, 255)
				
				if len(boxes) > 0:
					if count_face > len(boxes):
						count_face = len(boxes)
					vec = face_recognition.face_encodings(rgb, [boxes[count_face - 1]])
					preds = recognizer.predict_proba(vec)[0]
					j = np.argmax(preds)
					proba = preds[j]

				if proba > 0.8:
					name = le.classes_[j]
					color = (50, 205, 50)
				print(preds)
				print(j)


				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
						(color), 2)
				cv2.putText(frame, text, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (color), 2)
				cv2.imshow('frame', frame)

		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
	else:
		break


cap.release()
# out.release()

cv2.destroyAllWindows()
