from . import DetectionToolKit as dtk
from . import FaceToolKit as ftk
import time
import tensorflow as tf
import numpy as np
import cv2
from .detection.mtcnn import detect_face
import matplotlib.pyplot as plt
from scipy import misc
import os, pickle


with tf.Graph().as_default():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

default_color = (0, 255, 0)  # BGR
default_thickness = 2

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


# verify image
# import face_modules.single_face_detection.FaceToolKit as ftk

verification_threshhold = 0.9
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/models/0180204-160909/")
v.initial_input_output_tensors()


d = dtk.Detection()


def img_to_encoding(image):
    try:
        # image = plt.imread(img)
        # print('image', image)
        # cv2.imwrite("./lost.jpg", image)
        # aligned = d.align(image, False)[0]
        # cv2.imwrite("./aligned_n.jpg", aligned)
        # print('aligned', aligned)
        return v.img_to_encoding(image, image_size)
    except Exception as e:
        print(str(e))
    #     cv2.imwrite("./what2.jpg", image)
        return []


def img_to_encoding_db(img):
    image = plt.imread(img)
    print('image----', image)
    aligned = d.align(image, False)[0]
    return v.img_to_encoding(aligned, image_size)

def dataset(path):
    database = {}
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for file in files:
            fullpath = os.path.join(root, file)
            split_path = fullpath.split('/')
            split_filename = split_path[-1].split('.')
            person_name = split_filename[0]
            print(person_name)
            database[person_name] = img_to_encoding_db(fullpath)

    with open("/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/database/database.pkl","wb") as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # database = pickle.load(open("/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/database/database.pkl", "rb" ) )
    # print('database', database)
    return database


def distance(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))


def who_is_it(image, database):

    # Step 1: Compute the target "encoding" for the image. Use img_to_encoding()
    encoding = img_to_encoding(image)
    # print('encoding', encoding)
    if len(encoding) == 0:
        return 0, "what"
    # print('encoding')

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100
    min_dist = 1000
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = distance(encoding, db_enc)
        # cv2.imwrite(f"./{name}.jpg", db_enc)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        # identity = "unknown"
        if min_dist > dist:
            min_dist = dist
            identity = name

    if min_dist > verification_threshhold:
        identity = "unknown"
        print("Not in the database." + ", the distance is " + str(min_dist))
        cv2.imwrite("./unknown.jpg", image)
    else:
        # identity = name
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


def train_dataset():
    print('work')
    path = 'static/singleFace/dataset'
    dataset(path)


def single_face():
    with open('/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/database/database.pkl', 'rb') as handle:
        database = pickle.load(handle)
    print('database', database)
    font = cv2.FONT_HERSHEY_SIMPLEX
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        height, width, channels = frame.shape

        if ret == True:
            bounding_boxes = d.detect(frame, True)

            if len(bounding_boxes) > 0:

                for bounding_box in bounding_boxes:
                    pts = bounding_box[:4].astype(np.int32)

                    if pts[0] <= 0 or pts[1] <= 0 or pts[2] >= len(frame[0]) or pts[3] >= len(frame):
                        print('face is very close')
                        continue

                    face_image = frame[pts[1]:pts[3], pts[0]:pts[2], :]
                    face_image = misc.imresize(face_image, (image_size, image_size), interp='bilinear')
                    min_dist, identity = who_is_it(face_image, database)


                    pt1 = (pts[0], pts[1])
                    pt2 = (pts[2], pts[3])
                    # img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,255,255),2)
                    cv2.rectangle(frame, pt1, pt2, color=default_color, thickness=default_thickness)
                    cv2.putText(frame, str(identity), (pts[0]+5,pts[1]-5), font, 1, (255,255,255), 2)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break
