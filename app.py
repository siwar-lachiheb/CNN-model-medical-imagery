
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2
# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image
import matplotlib.pyplot as plt
# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os
from flask import jsonify # <- `jsonify` instead of `json`
#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files

MODEL_ARCHITECTURE = './model/model1.json'
MODEL_WEIGHTS = './model/model1.h5'

img_size =224
# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	img1 = cv2.imread(img_path)[...,::-1]
	plt.figure(figsize = (5,5))
	plt.imshow(img1)
	resized_arr1 = cv2.resize(img1, (img_size, img_size))
	print(img1.shape)
	x_train = []
	x_train.append(resized_arr1)
	print(img1.shape)
	x_train = np.array(x_train) / 255 
	#changing the order of parameters to reverse the order of channels colors -> working with numpy arrays 
	x_train.reshape(-1, img_size, img_size, 1)
	print(x_train.shape)

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	predictions = model.predict(x_train)
	classes_x=np.argmax(predictions)
	predictions1 = (model.predict(x_train) > 0.5).astype("int32")
	print("my predict:",classes_x)
	return classes_x
# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = {'TRAIN': ['NORMAL', 'PNEUMONIA']}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)
		print("file_path:/n")
		print(file_path)
		# Make a prediction
		prediction = model_predict(file_path, model)
		if int(prediction) == 0 :
			prediction1="NORMAL"
		if int(prediction) == 1 :
			prediction1="PNEUMONIA"
		result=jsonify({'pred':prediction1  ,'pred2':"prediction" })
		return result

if __name__ == '__main__':
	app.run(debug = True)
