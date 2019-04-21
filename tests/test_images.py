from keras import applications, optimizers
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Flatten, Dense
import numpy as np
import tensorflow as tf
import os

import cv2

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version: " + tf.__version__)

def triplet_loss(y_true, y_pred):
  N = 3
  beta = N
  epsilon = 1e-6

  anchor = y_pred[0::3]
  positive = y_pred[1::3]
  negative = y_pred[2::3]

  positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 0)
  negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 0)

  # -ln(-x/N+1)
  positive_distance = -tf.log(-tf.divide((positive_distance), beta) + 1 + epsilon)
  negative_distance = -tf.log(-tf.divide((N - negative_distance), beta) + 1 + epsilon)

  loss = negative_distance + positive_distance
  return loss

def metric_positive_distance(y_true, y_pred):
  N = 3
  beta = N
  epsilon = 1e-6
  anchor = y_pred[0::3]
  positive = y_pred[1::3]
  positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 0)
  positive_distance = -tf.log(-tf.divide((positive_distance), beta) + 1 + epsilon)
  return backend.mean(positive_distance)

#vgg16_model_weights_path = '../vgg16_model_weights.h5'
top_model_weights_path = '../top_model_weights.h5'

#vgg16_model = applications.VGG16(weights = vgg16_model_weights_path, include_top=False)
valid_data = np.load(open('../top_model_features_valid.npy', 'rb'))

top_model = Sequential()
top_model.add(Flatten(input_shape = valid_data.shape[1:]))
top_model.add(Dense(64, activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation = 'sigmoid', name='layer1'))
top_model.load_weights(top_model_weights_path)
top_model.compile(optimizer = optimizers.Adam(), loss = triplet_loss, metrics = [metric_positive_distance])

y_dummie = np.array([0] * 315)

print(top_model.metrics_names)
#for i in range(300, 315, 3):
#print(valid_data[i:i+3].shape)
i = 0
#results = top_model.evaluate(x = valid_data[i:i+6], y = valid_data[i:i+6],  batch_size = 3, verbose = 0)
results = top_model.predict(x = valid_data,  batch_size = 3, verbose = 0)
print(results)

'''
img = cv2.imread(test_data_dir + "/0/image-0477.png")
template = cv2.imread(test_data_dir + "/0/image-0478.png")
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# print("NN similarity:", results)
print("OpenCV normed correlation:", res[0][0])
'''

