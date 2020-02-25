import matplotlib.pyplot as plt
from . import DetectionToolKit as dtk
from . import FaceToolKit as ftk
import os
import pickle



# def load_model():
verification_threshhold = 0.9
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
print('start model load')
v.load_model("/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/models/0180204-160909/")
v.initial_input_output_tensors()
print('end model load')

d = dtk.Detection()

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

    f = open("/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/database/database.pkl","wb")
    pickle.dump(dict,f)
    f.close()

    # database = pickle.load(open("/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/single_face_detection/database/database.pkl", "rb" ) )
    # print('database', database)
    return database



def train_dataset():
    print('work')
    path = 'static/singleFace/dataset'
    dataset(path)
    


