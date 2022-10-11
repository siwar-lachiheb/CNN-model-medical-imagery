
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image
import cv2
# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

MODEL_ARCHITECTURE = './model/model1.json'
MODEL_WEIGHTS = './model/model1.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
img_size =224
# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded')


# ::: MODEL FUNCTIONS :::
def model_predict( model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	img1 = cv2.imread("./test_imgs/NORMAL2-IM-1440-0001.jpeg")[...,::-1]
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



classes = {'TRAIN': ['NORMAL', 'PNEUMONIA']}
prediction = model_predict( model)
test=prediction
predicted_class = classes['TRAIN'][test]
print('We think that is {}.'.format(predicted_class.lower()))
print(str(predicted_class).lower())


