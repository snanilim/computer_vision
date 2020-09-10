from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import matplotlib.pyplot as plt

input_video="/home/nilim/Documents/programmer/backup/record.avi"
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"


def distance(emb1, emb2):
    # print('emb1', emb1)
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))


def compare_two(img_path):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 160 # important point to increase accuracy
            input_image_size = 160
            
            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            # video_capture = cv2.VideoCapture(0)
            c = 0


            print('Start Recognition')
            prevTime = 0
            # while True:
                # ret, frame = video_capture.read()
            # img_path = '/home/nilim/Downloads/NID_Front_Image/cbimage.jpg'
            # frame = cv2.imread(img_path,0)
            frame = plt.imread(img_path)

            frame = cv2.resize(frame, (0,0), fx=1, fy=1)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)
                    print('nrof_faces', nrof_faces)
                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        print('i', i)

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            # i = i + 1
                            print('Face is very close!')
                            continue
                        

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]) # boxes

                        cv2.imwrite('./ok.jpg', frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        print('cropped[i]', len(cropped))
                        print('cropped[i]', i)

                        if i == len(cropped):
                            break

                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))


                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                        print('emb_array', emb_array[0, :])
                        print('squeeze', np.squeeze(emb_array))

                        return np.squeeze(emb_array)

                else:
                    print('Alignment Failure')



img_path_1 = "/home/nilim/Downloads/EC_IMAGE/1584820232589.JPEG"
img_path_2 = "/home/nilim/Downloads/NID_Front_Image/01911780554_3.jpg"

emb1 = compare_two(img_path_1)
emb2 = compare_two(img_path_2)

dist = distance(emb1, emb2)
print('dist', dist)


# this work good with big image, and align image