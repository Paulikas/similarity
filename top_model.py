import tensorflow as tf
import numpy as np
from time import time

from keras import backend, applications, optimizers, losses

from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

from keras.applications import VGG16

import os.path
import argparse


tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version: " + tf.__version__)

parser = argparse.ArgumentParser(description = 'Build a top layer for the similarity training and train it.')
parser.add_argument('-r', '--train-model', dest = 'train_model', action = 'store_true')
parser.add_argument('-s', '--test-model', dest = 'test_model', action = 'store_true')

parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 3, help = 'Batch size')
parser.add_argument('-e', '--epochs', dest = 'epochs', type = int, default = 5, help = 'Number of epochs to train the model.')
parser.add_argument('-t', '--train-dir', dest = 'train_dir', default = '/opt/datasets/data/simulated_flight_1/train/', help = 'Path to dataset training directory.')
parser.add_argument('-v', '--valid-dir', dest = 'valid_dir', default = '/opt/datasets/data/simulated_flight_1/valid/', help = 'Path to dataset validation directory.')
args = parser.parse_args()

model_weights_path = 'model_weights.h5'

def triplet_loss(N = 1, epsilon = 1e-6):
  def triplet_loss(y_true, y_pred):
    beta = N

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
  return triplet_loss

def metric_positive_distance(y_true, y_pred):
  N = 1
  beta = N
  epsilon = 1e-6
  anchor = y_pred[0::3]
  positive = y_pred[1::3]
  positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 0)
  positive_distance = -tf.log(-tf.divide((positive_distance), beta) + 1 + epsilon)
  return backend.mean(positive_distance)

def metric_negative_distance(y_true, y_pred):
  N = 1
  beta = N
  epsilon = 1e-6
  anchor = y_pred[0::3]
  negative = y_pred[2::3]
  negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 0)
  negative_distance = -tf.log(-tf.divide((N - negative_distance), beta) + 1 + epsilon)
  return backend.mean(negative_distance)

def make_top_model(input_shape):
  return top_model

def make_model():
  input_tensor = Input(shape = (224, 224, 3))
  base_model = VGG16(include_top = False, weights = 'imagenet', input_tensor = input_tensor)
  x = base_model.output
  for i in range(8):  
    base_model.layers.pop()
  #model.summary()
  
  for layer in base_model.layers:
    layer.trainable = False

  #base_model = Model(inputs = vgg16_model.input, outputs = vgg16_model.get_layer('block3_pool').output)

  #model = Sequential()
  x = Flatten()(x)
  x = Dense(64, activation = 'relu')(x)
  x = Dropout(0.5)(x)
  pred = Dense(1, activation = 'sigmoid')(x)
  
  model = Model(inputs = base_model.input, outputs = pred)



  #for i, layer in enumerate(model.layers):
  #  print(i, layer.name)

  
  #model = Sequential()

  #for layer in base_model.layers[0:18]:
  #  layer.trainable = True

  #model.add(base_model)
  #model.add(top_model) 
  return model

def train_model():
  datagen = ImageDataGenerator()
  train_generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'binary', shuffle = False)
  valid_generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'binary', shuffle = False)

  model = make_model()

  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [metric_positive_distance, metric_negative_distance])


  tensorboard = TensorBoard(log_dir = "./logs/{}".format(time()))
  model.fit_generator(train_generator, nb_train_samples, epochs = args.epochs)

  #for i, layer in enumerate(model.layers):
  #  print(i, layer.name)
  for layer in model.layers[:4]:
     layer.trainable = False
  for layer in model.layers[4:]:
     layer.trainable = True
  
  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [metric_positive_distance, metric_negative_distance])
  results = model.fit_generator(train_generator, nb_train_samples, epochs = args.epochs)
  
  print(results.history)

  model.save_weights(model_weights_path)

def test_model():
  datagen = ImageDataGenerator()
  train_generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
  valid_generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)

  model = make_model()
  model.load_weights(model_weights_path)

  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [metric_positive_distance, metric_negative_distance])


  print(model.metrics_names)
  results = model.predict_generator(generator = valid_generator, steps = 100, verbose = 0)
  
 
  N = 1
  beta = N
  epsilon = 1e-6
  anchor = results[0::3]
  positive = results[1::3]
  negative = results[2::3]
  

  positive_distance = np.sum(np.square(anchor - positive), axis = 1)
  negative_distance = np.sum(np.square(anchor - negative), axis = 1)

  # -ln(-x/N+1)
  positive_distance = - np.log(- (positive_distance / beta) + 1 + epsilon)
  negative_distance = - np.log(-(N - negative_distance / beta) + 1 + epsilon)

  

  print(np.concatenate((positive_distance[np.newaxis].transpose(), negative_distance[np.newaxis].transpose()), axis = 1))
  

if __name__ == '__main__':
  if args.train_model:
    nb_train_samples = len(os.listdir(args.train_dir + "/0")) / 3
    train_model()
  elif args.test_model:
    test_model()   
