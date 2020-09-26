# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import numpy as np
import argparse
import imutils
import time
import cv2
import os
# from align_video_faces import face_align

userName = input("Please Write your name: ")

if not os.path.exists(f"/home/nilim/Documents/programmer/computer_vision/facenet-test/facenet_aisangam/pre_img/{userName}/"):
    os.makedirs(f"/home/nilim/Documents/programmer/computer_vision/facenet-test/facenet_aisangam/pre_img/{userName}/")

args = {
    'prototxt': 'deploy.prototxt.txt',
    'model': '../model/res10_300x300_ssd_iter_140000.caffemodel',
    'confidence': 0.5
}

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

camExitNum = 0
# loop over the frames from the video stream

# imagePaths = list(paths.list_images("/home/nilim/Documents/programmer/backup/dataset/mahfuz"))
# for (i, imagePath) in enumerate(imagePaths):
# 	frame = cv2.imread(imagePath)
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		# text = "{:.2f}%".format(confidence * 100)
		# y = startY - 10 if startY - 10 > 10 else startY + 10
		# cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		# cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		camExitNum += 1
		newdir = "/home/nilim/Documents/programmer/computer_vision/facenet-test/facenet_aisangam/pre_img/"
		image = frame
		print('frame', image)
		if image is not None:
			cv2.imwrite(f"{newdir}/{userName}/ " + str(camExitNum)+ "b.jpg", image)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()