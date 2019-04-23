import tensorflow as tf
import numpy as np
from time import time

from PIL import Image

from keras import backend, applications, optimizers

from keras.models import Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

from keras.callbacks import Callback

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
parser.add_argument('-u', '--test-dir', dest = 'test_dir', default = '/opt/datasets/data/simulated_flight_1/test/', help = 'Path to dataset test directory.')

args = parser.parse_args()

model_weights_path = 'model_weights.h5'

def triplet_loss(N = 9, epsilon = 1e-6):
  def triplet_loss(y_true, y_pred):
    beta = N

    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    negative = y_pred[2::3]

    positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = 0, keepdims = True)
    negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = 0, keepdims = True)

    # -ln(-x/N+1)
    positive_distance = -tf.log(-tf.divide((positive_distance), beta) + 1 + epsilon)
    negative_distance = -tf.log(-tf.divide((N - negative_distance), beta) + 1 + epsilon)

    loss = negative_distance + positive_distance
    return loss
  return triplet_loss

def pd(N = 9, epsilon = 1e-6):
  def pd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 0)
    positive_distance = -tf.log(-tf.divide((positive_distance), beta) + 1 + epsilon)
    return backend.mean(positive_distance)
  return pd

def nd(N = 9, epsilon = 1e-06):
  def nd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    negative = y_pred[2::3]
    negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 0)
    negative_distance = -tf.log(-tf.divide((N - negative_distance), beta) + 1 + epsilon)
    return backend.mean(negative_distance)
  return nd

def make_top_model(input_shape):
  return top_model

def make_model():
  input_tensor = Input(shape = (224, 224, 3))
  base_model = applications.VGG16(include_top = False, weights = 'imagenet', input_tensor = input_tensor)
  x = base_model.output

  #for i in range(8):  
  #  base_model.layers.pop()
  
  for layer in base_model.layers:
    layer.trainable = False

  #x = Flatten()(x) # shape(1)
  x = Dense(7, activation = 'sigmoid')(x) # shape(1, 7 , 64)
  x = Dropout(0.5)(x)
  pred = Dense(64, activation = 'sigmoid')(x)
  
  model = Model(inputs = base_model.input, outputs = pred)

  return model

class EarlyStop(Callback):
  def on_batch_end(self, batch, logs = {}):
    if logs.get('loss') <= 1.0:
      self.model.stop_training = True

class EarlyStop2(Callback):
  def on_batch_end(self, batch, logs = {}):
    if logs.get('loss') <= 1.0:
      self.model.stop_training = True

def train_model():
  datagen = ImageDataGenerator()
  train_generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
  valid_generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)

  model = make_model()
  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [pd(), nd()])

  tensorboard = TensorBoard(log_dir = "./logs/{}".format(time()))

  # Keras data generator is meant to loop infinitely — it should never return or exit.
  # Keras has no ability to determine when one epoch starts and a new epoch begins.
  # Therefore, we compute the steps_per_epoch value as the total number of training data points divided by the batch size.
  # Once Keras hits this step count it knows that it’s a new epoch.

  early_stop = EarlyStop()

  model.fit_generator(generator = train_generator, steps_per_epoch = 9, epochs = args.epochs, validation_data = valid_generator, validation_steps = 6, callbacks = [tensorboard, early_stop])

  for layer in model.layers[:4]:
     layer.trainable = False
  for layer in model.layers[4:]:
     layer.trainable = True
  
  early_stop = EarlyStop2()

  model.fit_generator(generator = train_generator, steps_per_epoch = 9, epochs = args.epochs, validation_data = valid_generator, validation_steps = 6, callbacks = [tensorboard, early_stop])
  
  model.save_weights(model_weights_path)

def test_model():
  test_samples = len(os.listdir(args.test_dir + "/0"))
  datagen = ImageDataGenerator()
  test_generator = datagen.flow_from_directory(directory = args.test_dir, target_size = (224, 224), batch_size = test_samples, class_mode = 'categorical', shuffle = False)
  
  model = make_model()
  model.load_weights(model_weights_path)
  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [pd(), nd()])

  results = model.predict_generator(generator = test_generator, steps = 1, verbose = 0)
 
  N = 1
  beta = N
  epsilon = 1e-6
  anchor = results[0::3]
  positive = results[1::3]
  negative = results[2::3]
  
  I = anchor[6][0]
  img = Image.fromarray(np.array( (((I - I.min()) / (I.max() - I.min())) * 255.9), dtype=np.uint8))
  img.save('testa.png')
  I = positive[6][0]
  img = Image.fromarray(np.array( (((I - I.min()) / (I.max() - I.min())) * 255.9), dtype=np.uint8))
  img.save('testp.png')
  I = negative[6][0]
  img = Image.fromarray(np.array( (((I - I.min()) / (I.max() - I.min())) * 255.9), dtype=np.uint8))
  img.save('testn.png')


  positive_distance = np.nansum(np.square(anchor - positive), axis = 1)
  positive_distance = - np.log(- (positive_distance / beta) + 1 + epsilon)

  positive_distance2 = np.nansum(np.square(anchor - negative), axis = 1)
  positive_distance2 = - np.log(- (positive_distance2 / beta) + 1 + epsilon)
  
  for i in range(test_samples // 3):
    print(i, 'p ', np.nansum(positive_distance[i]))
    print(i, 'n ', np.nansum(positive_distance2[i]))
  
if __name__ == '__main__':
  if args.train_model:
    train_model()
  elif args.test_model:
    test_model()   
