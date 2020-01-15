# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
 
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--embeddings", required=True,
# 	help="path to serialized db of facial embeddings")
# ap.add_argument("-r", "--recognizer", required=True,
# 	help="path to output model trained to recognize faces")
# ap.add_argument("-l", "--le", required=True,
# 	help="path to output label encoder")
args = {
    'embeddings': 'output/embeddings.pickle', 
    'recognizer': 'output/recognizer.pickle', 
    'le': 'output/le.pickle'
}

data = pickle.loads(open(args['embeddings'], 'rb').read())
# print(data)
le = LabelEncoder()
labels = le.fit_transform(data['names'])
# print(labels)

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# print(le)

f = open(args["recognizer"], 'wb')
f.write(pickle.dumps(recognizer))
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

