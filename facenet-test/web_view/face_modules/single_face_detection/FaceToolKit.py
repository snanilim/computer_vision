# -*- coding: utf-8 -*-
'''
https://github.com/Alireza-Akhavan
'''
import tensorflow as tf
import numpy as np
from . import face
# import face_modules.single_face_detection.face

class Verification:
    """
    Match or mismatch of two faces
    """
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.images_placeholder = ''
        self.embeddings = ''
        self.phase_train_placeholder = ''
        self.embedding_size = ''
        self.session_closed = False

    def __del__(self):
        if not self.session_closed:
            self.session.close()

    def kill_session(self):
        self.session_closed = True
        self.session.close()

    def load_model(self, model):
        """
        Model loading
         This function must be called before other functions.
         The input to this function is a pre-trained model path for feature extraction.
        """
        face.load_model(model, self.session)

    def initial_input_output_tensors(self):
        """
        Creating input and output tensors from the model loaded by the function
        load_model
        """
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]


    def img_to_encoding(self, img, image_size):
        """
            Calculate the embedding of an image
            Here's a vector of 128 for each image
        """
        image = face.make_image_tensor(img, image_size)
        
        feed_dict = {self.images_placeholder: image, self.phase_train_placeholder:False }
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.session.run(self.embeddings, feed_dict=feed_dict)

        return np.squeeze(emb_array)

