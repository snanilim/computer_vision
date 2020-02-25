from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
from . import facenet
from . import detect_face
import os
import time
import pickle

input_video="/home/nilim/Documents/programmer/backup/record.avi"
modeldir = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/model/20170511-185253.pb'
classifier_filename = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/class/classifier.pkl'
npy='/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/npy'
train_img="/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/train_img"


def capture_img():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            frame_interval = 3
            
            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model(modeldir)


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            video_capture = cv2.VideoCapture(0)
            c = 0

            newdir = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/train_img'
            userName = 'noone'
            camExitNum = 0
            
            while True:
                if camExitNum < 50:
                    ret, frame = video_capture.read()
                    frame = cv2.resize(frame, (0,0), fx=1, fy=1)    #resize frame (optional)
                    timeF = frame_interval

                    if (c % timeF == 0):
                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Detected_FaceNum: %d' % nrof_faces)

                        if nrof_faces > 0:
                            print('face has')
                            cv2.imwrite(f"{newdir}/{userName}/ " + str(camExitNum)+ "b.jpg", frame)
                            camExitNum += 1
                        else:
                            print('Alignment Failure')
                    # c+=1
                    # frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)    #resize frame (optional)
                    # cv2.imshow('Video', frame)

                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                    frame = cv2.imencode('.jpg', frame)[1].tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.1)
                else:
                    break

            video_capture.release()
            cv2.destroyAllWindows()
