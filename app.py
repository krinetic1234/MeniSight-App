from flask import Flask, render_template, request

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        if(request.form["type"] == "0"):
            if 'image' not in request.files:
                return render_template('hurricanes.html', output="File not found! Please try re-uploading.")
            f = request.files["image"]
            if f.filename == '':
                return render_template('hurricanes.html', output="File not found! Please try re-uploading.")
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (300, 300))
                image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
                image_tensor = tf.expand_dims(image_tensor, 0)
                result = hurricane_model.predict(image_tensor)[0][0]
                print(result)
                result = result * 100
                if(result < 0):
                    result = 0
                if(result > 100):
                    result = 100
                result = round(result, 2)
                result = str(result)
                output = result + "% chance of a flood damage after a hurricane"
                return render_template('detection.html', output=output)
            return render_template('detection.html', output="An unknown error occurred!")
        else:
            return render_template('detection.html', output="Coming soon!")
    else:
        return render_template('detection.html', output="")


@app.route('/prognosis')
def prognosis():
    return render_template('prognosis.html')


@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js')
