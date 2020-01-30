import FaceToolKit as ftk
import matplotlib.pyplot as plt
import numpy as np


verification_threshhold = 1.188
image_size = 160



v = ftk.Verification()

# Pre-load model for Verification
v.load_model("./models/0180204-160909")
v.initial_input_output_tensors()

img1 = plt.imread("./images/1.jpg")
img2 = plt.imread("./images/2.jpg")
img3 = plt.imread("./images/3.jpg")


# Display the images
# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()
# plt.imshow(img3)
# plt.show()



#generate embeddings
emb1 = v.img_to_encoding(img1, image_size)
emb2 = v.img_to_encoding(img2, image_size)
emb3 = v.img_to_encoding(img3, image_size)

print(emb1.shape)



def distance(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))

#distance
dist = distance(emb1, emb2)
is_same = dist < verification_threshhold
print ("distance img1 and img2 =", dist, " issame =", is_same)

dist = distance(emb1, emb3)
is_same = dist < verification_threshhold
print ("distance img1 and img3 =", dist, " issame =", is_same)



