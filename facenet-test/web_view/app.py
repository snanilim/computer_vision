import time, os
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
import re, time, base64
import numpy as np
from face_modules.single_face_detection import single_face, train_dataset, compare_two_img
from face_modules.multiple_face_detection import identify_face_video
from face_modules.multiple_face_detection import data_preprocess
from face_modules.multiple_face_detection import train_main
from face_modules.multiple_face_detection import capture_img_from_video
# from face_modules.single_face_detection import train_dataset

UPLOAD_FOLDER = 'static/singleFace/dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('upload.html')
        file = request.files['file']
        person_name = request.form['person_name']
        print('name', person_name)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('upload.html')
        if file and allowed_file(file.filename):
            filename = f'{person_name}.jpg'
            filename = secure_filename(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload.html')
    return render_template('upload.html')


@app.route('/save-image', methods=['GET', 'POST'])
def save_image():
    if request.method == 'POST':
        print('save image call')
        print('body', request.get_json())
        base64_image = request.get_json()['img_path']
        person_name = request.get_json()['person_name']
        img_path = f"{UPLOAD_FOLDER}/{person_name}.jpg"

        image = data_uri_to_cv2_img(base64_image)
        cv2.imwrite(img_path, image)

    return redirect(url_for('capture'))



@app.route('/single-face')
def single():
    """Video streaming home page."""
    return render_template('single.html')



@app.route('/multiple-face')
def multiple():
    """Video streaming home page."""
    return render_template('video.html')



@app.route('/capture')
def capture():
    """Video streaming home page."""
    return render_template('capture.html')

@app.route('/multiple-capture')
def multiple_capture():
    """Video streaming home page."""
    return render_template('multiple_capture.html')
    

@app.route('/capture_video_pic')
def capture_video_pic():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(capture_img_from_video.capture_img(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/train_single_face_database')
def train_single_face_database():
    result = train_dataset()

    return render_template('train.html')

@app.route('/single_video_feed')
def single_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(single_face(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save-multiple-image', methods=['GET', 'POST'])
def save_multiple_image():
    if request.method == 'POST':
        person_name = request.get_json()['person_name']

    os.rename('/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/train_img/noone', f'/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/train_img/{person_name}')
    os.mkdir('/home/nilim/Documents/programmer/computer_vision/facenet-test/web_view/face_modules/multiple_face_detection/train_img/noone')
    return "True"


@app.route('/train_multiple_face_database')
def train_multiple_face_database():
    result = data_preprocess.pre_process_image()
    result = train_main.train_multiple_image()
    return render_template('train.html')       

@app.route('/multiple_video_feed')
def multiple_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag.""";
    return Response(identify_face_video.multiple_face_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/compare_two_pic_api', methods=['GET', 'POST'])
def compare_two_pic_api():
    if request.method == 'POST':
        UPLOAD_FOLDER_TWO_IMG = 'static/compare_two_img'

        base64_image_one = request.get_json()['img_one']
        base64_image_two = request.get_json()['img_two']

        img_path_one = f"{UPLOAD_FOLDER_TWO_IMG}/compare_img_one.jpg"
        img_path_two = f"{UPLOAD_FOLDER_TWO_IMG}/compare_img_two.jpg"

        image_one = data_uri_to_cv2_img(base64_image_one)
        image_two = data_uri_to_cv2_img(base64_image_two)

        cv2.imwrite(img_path_one, image_one)
        cv2.imwrite(img_path_two, image_two)

        distance = compare_two_img(img_path_one, img_path_two)

    return {"distance": distance}




if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='localhost', debug=True, port=3000)
