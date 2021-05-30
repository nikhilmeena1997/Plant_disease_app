from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
class_names = ['BrownSpot', 'Hispa', 'LeafBlast']
# Model saved with Keras model.save()
MODEL_PATH = 'models/Plant_Disease_Detector.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function() 
        # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def predict_from_loaded_model(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img=image.img_to_array(img)
    img=resnet50.preprocess_input(img)
    prediction=model.predict(img.reshape(1,224,224,3))
    output=np.argmax(prediction)
    return class_names[output]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_from_loaded_model(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)