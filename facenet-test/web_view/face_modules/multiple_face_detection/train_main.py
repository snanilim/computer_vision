from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from . import classifier


def train_multiple_image():
    datadir = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/pre_img'
    modeldir = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/model/20170511-185253.pb'
    classifier_filename = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/class/classifier.pkl'
    print ("Training Start")
    obj=classifier.training(datadir,modeldir,classifier_filename)
    get_file=obj.main_train()
    print('Saved classifier model to file "%s"' % get_file)
    # sys.exit("All Done")

    return True
