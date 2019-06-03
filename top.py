import tensorflow as tf
import numpy as np

from time import time
from PIL import Image

from tensorflow import saved_model
from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import argparse
import json, h5py, os, shutil, sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TensorFlow version: " + tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--train-model', dest = 'train_model', action = 'store_true')
parser.add_argument('-s', '--test-model', dest = 'test_model', action = 'store_true')
parser.add_argument('-l', '--lc', dest = 'lc', type = int, default = 8)
parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 3)
parser.add_argument('-e', '--epochs', dest = 'epochs', type = int, default = 5)
parser.add_argument('-t', '--train-dir', dest = 'train_dir', default = '/opt/datasets/data/simulated_flight_1/train/')
parser.add_argument('-v', '--valid-dir', dest = 'valid_dir', default = '/opt/datasets/data/simulated_flight_1/valid/')
parser.add_argument('-u', '--test-dir', dest = 'test_dir', default = '/opt/datasets/data/simulated_flight_1/test/')

args = parser.parse_args()

checkpoint_dir = 'runtime_files/saved_model'
checkpoint_auto_dir = 'runtime_files/auto_saved_model.h5'
# argN = args.batch_size // 3
# Current output shape
# argN = 28*28*1
argN = -1

def triplet_loss(N = argN, epsilon = 1e-6):
  def triplet_loss(y_true, y_pred):
    # N = np.int(y_pred.shape[1])
    beta = N

    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    negative = y_pred[2::3]


    # positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis = 0, keepdims = True)
    # negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis = 0, keepdims = True)

    # x
    positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis=1, keepdims=True)
    negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis=1, keepdims = True)

    # -ln(-x/beta+1+epsilon)
    positive_distance = -tf.math.log(-tf.math.divide((positive_distance), beta) + 1 + epsilon)
    negative_distance = -tf.math.log(-tf.math.divide((N - negative_distance), beta) + 1 + epsilon)

    # sum(-ln(...))
    # loss = tf.reduce_mean(tf.add(negative_distance,positive_distance))
    loss = tf.reduce_sum(tf.add(negative_distance, positive_distance))
    return loss
  return triplet_loss

def pd(N = argN, epsilon = 1e-6):
  def pd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    # positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis=0)
    positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis=1)
    positive_distance = -tf.math.log(-tf.math.divide((positive_distance), beta) + 1 + epsilon)
    # return backend.mean(positive_distance)
    return tf.reduce_sum(positive_distance)
  return pd

def nd(N = argN, epsilon = 1e-06):
  def nd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    negative = y_pred[2::3]
    # negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis=0)
    negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis=1)
    negative_distance = -tf.math.log(-tf.math.divide((N - negative_distance), beta) + 1 + epsilon)
    # return backend.mean(negative_distance)
    return tf.reduce_sum(negative_distance)
  return nd

def make_model():
  #input_tensor = Input(shape = (224, 224, 3),  , name = 'input')
  base_model = applications.VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))

  model = Sequential()
  if (args.lc > 0):
    for layer in base_model.layers[:-1 * args.lc]:
      model.add(layer)
  else:
    for layer in base_model.layers:
      model.add(layer)
  
  for layer in model.layers:
    layer.trainable = False

  # Jeigu nukerpama daugiau negu 8 keiciaisi input shape, todel input shape atsisakome
  # model.add(Dense(64, activation = 'sigmoid', input_shape = (1, 28, 28, 256)))

  # Bandome su maziau sluoksniu
  # model.add(Dense(64, activation='sigmoid'))
  # model.add(Dropout(0.5))
  model.add(Dense(1, activation = 'sigmoid', name="out"))
  model.add(Flatten(name="flat_out"))
  # Update argN to dinamicaly set dimension
  global argN
  argN = int(model.layers[-1].output.shape[1])
  print(f'argN = {argN}')

  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(argN), metrics = [pd(argN), nd(argN)])

  return model

def train_model():
  model = make_model()

  datagen = ImageDataGenerator()
  train_generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
  # valid_generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
  # Validuojame vienu žingsniu
  valid_generator = datagen.flow_from_directory(directory=args.valid_dir, target_size=(224, 224),
                                                batch_size=33, class_mode='categorical', shuffle=False)

 
  cb_tensorboard = TensorBoard(log_dir = "./runtime_files/logs/{}".format(time()), histogram_freq = 2, write_graph = True, write_images = False)
  cb_checkpoint = ModelCheckpoint(checkpoint_auto_dir, save_weights_only = False, period = 100, verbose = 1)

  # model.fit_generator(generator = train_generator, steps_per_epoch = argN, epochs = args.epochs, validation_data = valid_generator, validation_steps = 3, callbacks = [cb_tensorboard, cb_checkpoint])
  model.fit_generator(generator=train_generator, epochs=args.epochs, validation_data=valid_generator, callbacks=[cb_tensorboard, cb_checkpoint], verbose=1)


  # tf.keras.experimental.export_saved_model(model, checkpoint_dir)
  # tf.saved_model.save(model, checkpoint_dir)
  model.save('runtime_files/train_model.h5')

  return model

def test_model(model = None):
  test_samples = len(os.listdir(args.test_dir + "/0"))
  datagen = ImageDataGenerator()
  test_generator = datagen.flow_from_directory(directory = args.test_dir, target_size = (224, 224), batch_size = 33, class_mode = 'categorical', shuffle = False)

  # model = tf.keras.experimental.load_from_saved_model(checkpoint_dir)
  #model = tf.saved_model.load(checkpoint_dir, tags = None)

  if model is None:
    model = make_model()
    # model.summary()
    # model.load_weights(checkpoint_auto_dir)
    model.load_weights('runtime_files/train_model.h5')
    # tf.saved_model.load(checkpoint_dir)

  results = model.predict_generator(generator = test_generator, steps = test_samples // 32, verbose = 0)

  print(f'argN = {argN}')
  N = argN
  beta = N
  epsilon = 1e-6
  anchor = results[0::3]
  positive = results[1::3]
  negative = results[2::3]

  positive_distance = np.nansum(np.square(anchor - positive), axis=1, keepdims=True)
  positive_distance = -np.log(- (positive_distance / beta) + 1 + epsilon)

  negative_distance = np.nansum(np.square(anchor - negative), axis=1, keepdims=True)
  # negative_distance = -np.log(- ((N - negative_distance) / beta) + 1 + epsilon)
  # Skaiciuojame taip pat kaip ir panasuma tarp teigiamu
  negative_distance = -np.log(- (negative_distance / beta) + 1 + epsilon)
 
  tp = 0
  fp = 0
  pneq = 0
  min_p = sys.maxsize
  max_p = 0
  min_n = sys.maxsize
  max_n = 0

  for i in range(test_samples // 3):
    pda = np.nansum(positive_distance[i])
    nda = np.nansum(negative_distance[i])
    print(pda, "\t", nda)
    if pda >= nda:
    # print(i)
      fp += 1
    else:
      tp += 1
    if pda == nda:
      pneq += 1

    if min_p > pda:
      min_p = pda
    if max_p < pda:
      max_p = pda

    if min_n > nda:
      min_n = nda
    if max_n < nda:
      max_n = nda

  print('Summary')
  print(min_p, ' - ', max_p, ', ', min_n, ' - ', max_n)
  print('accuracy: ', np.round(tp / (tp + fp) * 100, 1))
  print('equal predictions: ', pneq)
  
if __name__ == '__main__':
  # args.train_model = True
  # args.lc = 12
  # args.epochs = 1
  if args.train_model:
    shutil.rmtree(checkpoint_dir, ignore_errors = True)
    train_model()
  elif args.test_model:
    test_model()
