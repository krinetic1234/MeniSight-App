from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import cv2
import os

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (300, 300)
UPLOAD_FOLDER = './uploads'
detection_model = load_model('detection.h5')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('detection.html', output="File not found! Please try re-uploading.")
        f = request.files["file"]
        if f.filename == '':
            return render_template('detection.html', output="File not found! Please try re-uploading.")
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (300, 300))
            # print("Image")
            # print(image.shape)
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

            # print("Tensor")
            # print(image_tensor.shape.as_list())

            image_tensor = tf.expand_dims(image_tensor, 2)
            image_tensor = tf.expand_dims(image_tensor, 0)

            # print("After Expand")
            # print(image_tensor.shape.as_list())

            result = detection_model.predict(image_tensor)[0][0]
            result = result * 100
            if(result < 0):
                result = 0
            if(result > 100):
                result = 100
            result = round(result, 2)
            result = str(result)
            print(result)
            output = result + "% chance of a meningioma tumor"
            return render_template('detection.html', output=output)
        else:
            return render_template('detection.html', output="An unknown error occurred!")
    else:
        return render_template('detection.html', output="")


@app.route('/prognosis', methods=['GET', 'POST'])
def prognosis():
    if request.method == 'POST':
        age = float(request.form["age"])
        race = request.form["race"]
        gender = request.form["gender"]
        behavior = request.form["behav"]
        size = float(request.form["size"])
        site = request.form["site"]
        laterality = request.form["laterality"]

        prog_data = {'AGE': [age], 'LATITUDE': [
            longitude], 'MONTH': [month], 'DAY': [day], 'DEATHS': [deaths]}

        meningioma_df = pd.DataFrame(data=prog_data)

        output = ""
        return render_template('prognosis.html', output=output)
    else:
        return render_template('prognosis.html', output="")


@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js')
