from . import preprocess 

def pre_process_image():
    input_datadir = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/train_img'
    output_datadir = '/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/pre_img'

    obj=preprocess.preprocesses(input_datadir,output_datadir)
    nrof_images_total,nrof_successfully_aligned=obj.collect_data()

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    return True