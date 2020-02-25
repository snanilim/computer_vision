# computer_vision
computer_vision


# dlib
work well with face alignment
work well with big and small dataset [for small dataset some time lake]
work well with good quality picture
unknown handle well by distance
# drawback:
can not find all angle face
# run
python3 encode_faces.py
python3 recognize_faces_video.py

# dnn
work well with face alignment
very well for find a face
# drawback:
frequently give discremenative ans
very very bad with small dataset
# run
python3 extract_embeddings.py
python3 train_model.py
python3 recognize_video.py


# dlib and dnn
work well with big dataset
work well with good quality picture
unknown handle well by unwnown folder
predections percentage are so high
# drawback:
can not find all angle face
# run
python3 encode_faces.py
python3 dlib_train_model.py
python3 dlib_recognize_video.py



# command
autopep8 -i singledetector.py 