import matplotlib.pyplot as plt
import numpy as np

import FaceToolKit as ftk
import DetectionToolKit as dtk

verification_threshhold = 0.9
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./models/0180204-160909/")
v.initial_input_output_tensors()



d = dtk.Detection()



# def img_to_encoding(img):
#     # image = plt.imread(img)
#     # image = img
#     # print('image', type(image))
#     # aligned = d.align(image, False)[0]
#     return v.img_to_encoding(img, image_size)

def img_to_encoding(img):
    image = plt.imread(img)
    print('image', image)
    aligned = d.align(image, False)[0]
    return v.img_to_encoding(aligned, image_size)

def img_to_encoding_db(img):
    image = plt.imread(img)
    print('image', image)
    aligned = d.align(image, False)[0]
    return v.img_to_encoding(aligned, image_size)

database = {}

# database["alireza"] = img_to_encoding_db("./images/alireza.jpg")
# database["ali"] = img_to_encoding_db("./images/ali.jpg")
# database["mohsen"] = img_to_encoding_db("./images/mohsen.jpg")
# database["muhammad"] = img_to_encoding_db("./images/muhammad.jpg")
# database["nilim"] = img_to_encoding_db("/home/nilim/Documents/programmer/backup/face-match/nilim/received_2085046494845352.jpg")





def distance(emb1, emb2):
    print('emb1', emb1)
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))

def verify(image_path, identity, database):
   
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding()
    encoding = img_to_encoding(image_path) 
    
    # Step 2: Compute distance with identity's image
    dist = distance(encoding, database[identity])
    
    # Step 3: Open the door if dist < verification_threshhold, else don't open
    if dist < verification_threshhold:
        print("It's " + str(identity) + ", welcome!")
    else:
        print("It's not " + str(identity) + ", please go away")
        
        
    return dist




# verify("images/1.jpg", "alireza", database)



# verify("images/ali.jpg", "ali", database)



def who_is_it(image_path, database):
   
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding()
    encoding = img_to_encoding(image_path)
    print('encoding')
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 
    min_dist = 1000
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = distance(encoding, db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if min_dist > dist:
            min_dist = dist
            identity = name

  
    if min_dist > verification_threshhold:
        print("Not in the database. the distance is " + str(min_dist))
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity


# who_is_it("images/alireza-in-part.jpg", database)

# who_is_it("images/m.jpg", database)

database["abc"] = img_to_encoding_db("/home/nilim/Downloads/EC_IMAGE/1584820232589.JPEG")
who_is_it("/home/nilim/Downloads/EC_IMAGE/processed.jpeg", database)
# image = plt.imread("/home/nilim/Downloads/processed.jpeg")
# who_is_it(image, database)