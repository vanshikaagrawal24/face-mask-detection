from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2



app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

model = pickle.load(open('face.pkl','rb'))

@app.route('/', methods=['GET',"POST"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET',"POST"])
def predict():
    imageFile = request.files['imageFile']
    image_path = "./static"+imageFile.filename
    imageFile.save(image_path)
    input_image = mpimg.imread(image_path)
    plt.imshow(input_image)
    plt.show()
    input_image_resized = cv2.resize(input_image, (128,128))
    input_image_scaled = input_image_resized/255
    input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])    
    input_prediction = model.predict(input_image_reshaped)
    print(input_prediction)
    input_pred_label = np.argmax(input_prediction)
    print(input_pred_label)
    if input_pred_label == 0:
        return render_template('index.html', form="Not wearing Mask")
    elif input_pred_label == 1 :
        return render_template('index.html', form="Wearing Mask")
    

if __name__ == '__main__':
    app.run(debug=True)