from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection')
def features():
    return render_template('detection.html')


@app.route('/prognosis')
def pricing():
    return render_template('prognosis.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js')
