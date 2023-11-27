from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import imageio.v2
import numpy as np
from matplotlib import pyplot as plt

app = Flask(__name__)

# Ruta donde se guardarán las imágenes cargadas

# Ruta de modelo previamente entranado
MODEL_PATH = './models/model_Mnist_LeNet.h5'
model = tf.keras.models.load_model(MODEL_PATH,compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' not in request.files:
        return 'No se encontró ningún archivo'
    file = request.files['file']
    if file.filename == '':
        return 'No se seleccionó ningún archivo'
 
    if file: 
    
        im = imageio.imread(file)
        
        gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
        gray /= 255
        prediction = model.predict(gray.reshape(1, 28, 28, 1))
        
        print("predicted number: ", prediction.argmax())
        print(file.filename)
        
        datos={
            'predi':prediction.argmax(),
            'img':file.filename
        }
        return render_template('predict.html',datos=datos)


if __name__ == '__main__':
    app.run(debug=True)