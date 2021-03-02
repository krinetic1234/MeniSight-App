from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash, Response
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from sksurv.linear_model import CoxnetSurvivalAnalysis

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
import cv2
import os
import io
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from joblib import dump, load

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (300, 300)
UPLOAD_FOLDER = './uploads'
detection_model = load_model('detection.h5')
gb = load('coxnet.joblib')
surv_funcs = {}

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
            return render_template('detection.html', output=output, passed="true")
        else:
            return render_template('detection.html', output="An unknown error occurred!", passed="true")
    else:
        return render_template('detection.html', output="", passed="false")


@app.route('/prognosis', methods=['GET', 'POST'])
def prognosis():
    if request.method == 'POST':
        # Orig Data
        column_names = ['Age', 'Race', 'Sex', 'Behavior', 'Size',
                        'Site', 'Laterality', 'Status Recode', 'Survival Months']
        orig_dataf = pd.read_csv('prognosis-data.csv', na_values="NaN")
        orig_dataf.columns = column_names
        orig_dataf.drop(orig_dataf.tail(12514).index, inplace=True)
        orig_dataf['Age'] = orig_dataf['Age'].str[:2]
        orig_dataf = orig_dataf.iloc[:, :-2]

        # Grab our data
        age = float(request.form["age"])
        race = request.form["race"]
        sex = request.form["gender"]
        behavior = request.form["behav"]
        size = float(request.form["size"])
        site = float(request.form["site"])
        laterality = request.form["laterality"]

        prog_data = {'Age': [age], 'Race': [
            race], 'Sex': [sex], 'Behavior': [behavior], 'Size': [size], 'Site': [site], 'Size': [size], 'Laterality': [laterality]}

        prog_dataf = pd.DataFrame(data=prog_data)

        #   Concat dataframes
        prog_dataf = prog_dataf.append(orig_dataf)
        convert_dict = {'Race': 'category',
                        'Sex': 'category',
                        'Behavior': 'category',
                        'Site': 'category',
                        'Laterality': 'category',
                        }

        prog_dataf = prog_dataf.astype(convert_dict)

        non_dummy_cols = ['Age', 'Size']
        dummy_cols = list(set(prog_dataf.columns) - set(non_dummy_cols))
        prog_dataf = pd.get_dummies(prog_dataf, columns=dummy_cols)
        prog_dataf = prog_dataf.drop(columns=["Sex_Male", "Race_Unknown"])
        # print(prog_dataf.columns)

        data = prog_dataf.iloc[:1]
        surv_funcs[0] = gb.predict_survival_function(data)

        output = ""
        return render_template('survplot.html', output=output)
    else:
        return render_template('prognosis.html', output="")


@app.route('/survplot', methods=["GET"])
def survplot():
    return render_template('survplot.html')


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    for alpha, surv_alpha in surv_funcs.items():
        for fn in surv_alpha:
            axis.plot(fn.x, fn(fn.x))

    axis.set_ylim([0, 1])
    axis.set_title(
        'Probability vs. Time curve for Patient Overall Survival (OS)')
    axis.set_xlabel('Time (Months)')
    axis.set_ylabel('Probability of Survival')
    return fig


@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js')
