import tensorflow as tf
import numpy as np
from time import time

from PIL import Image

from keras import backend, applications, optimizers

from keras.models import Model, Sequential
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

parser.add_argument('-l', '--lc', dest = 'lc', type = int, default = 0)
parser.add_argument('-i', '--ic', dest = 'ic', type = int, default = 0)

parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 3, help = 'Batch size')
parser.add_argument('-e', '--epochs', dest = 'epochs', type = int, default = 5, help = 'Number of epochs to train the model.')
parser.add_argument('-t', '--train-dir', dest = 'train_dir', default = '/opt/datasets/data/simulated_flight_1/train/', help = 'Path to dataset training directory.')
parser.add_argument('-v', '--valid-dir', dest = 'valid_dir', default = '/opt/datasets/data/simulated_flight_1/valid/', help = 'Path to dataset validation directory.')
parser.add_argument('-u', '--test-dir', dest = 'test_dir', default = '/opt/datasets/data/simulated_flight_1/test/', help = 'Path to dataset test directory.')

args = parser.parse_args()

model_weights_path = 'model_weights.h5'

argN = 64

def triplet_loss(N = argN, epsilon = 1e-6):
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

def pd(N = argN, epsilon = 1e-6):
  def pd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 0)
    positive_distance = -tf.log(-tf.divide((positive_distance), beta) + 1 + epsilon)
    return backend.mean(positive_distance)
  return pd

def nd(N = argN, epsilon = 1e-06):
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
  input_tensor = Input(shape = (224, 224, 3), name = 'input')
  base_model = applications.VGG16(include_top = False, weights = 'imagenet', input_tensor = input_tensor)
  #x = base_model.output

  #for i in range(args.lc):  
  #  base_model.layers.pop()
  
  #for layer in base_model.layers:
  #  layer.trainable = False

  model = Sequential()
  if args.lc > 0:
    for layer in base_model.layers[:-1 * args.lc]:
      model.add(layer)
  else:
    for layer in base_model.layers:
      model.add(layer)
  
  for layer in model.layers:
    layer.trainable = False

  '''
  #x = Flatten()(x) # shape(1)
  x = Dense(7, activation = 'sigmoid')(x) # shape(1, 7 , 64)
  x = Dropout(0.5)(x)
  pred = Dense(64, activation = 'sigmoid')(x)
  
  model = Model(inputs = base_model.input, outputs = pred)
  '''
  #model.add(Dense(64, activation = 'sigmoid', input_shape = (224, 224, 3)))
  model.add(Dense(64, activation = 'sigmoid', input_shape = (1, 28, 28, 256)))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation = 'sigmoid', name="out"))

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
  sess = tf.Session()
  img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 224, 224, 3))
  var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 224, 224, 3))
  val = img + var
  out = tf.identity(val, name="out")

  with sess.as_default():
    sess.run(tf.global_variables_initializer())

    model = make_model()
    model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [pd(), nd()])

    model.summary()

    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
    valid_generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
 
    #tf.initialize_all_variables().run()
    tensorboard = TensorBoard(log_dir = "./logs/{}".format(time()))
    #early_stop = EarlyStop()

    g = tf.get_default_graph()
    tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=2000000)


    model.fit_generator(generator = train_generator, steps_per_epoch = argN, epochs = args.epochs, validation_data = valid_generator, validation_steps = 3, callbacks = [tensorboard])


    #for layer in model.layers[:4]:
    #   layer.trainable = False
    #for layer in model.layers[4:]:
    #   layer.trainable = True
  
    #for layer in model.layers:
    #   layer.trainable = True
  
    #early_stop = EarlyStop2()

    #model.fit_generator(generator = train_generator, steps_per_epoch = argN, epochs = args.ic, validation_data = valid_generator, validation_steps = 3, callbacks = [tensorboard])
    converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    eval_graph_file = 'graph.pb'
    checkpoint_name = 'checkpoint_a'

    with open(eval_graph_file, 'w') as f:
      f.write(str(g.as_graph_def()))
      saver = tf.train.Saver()
      saver.save(sess, checkpoint_name)

    model.save_weights(model_weights_path)

def test_model():
  test_samples = len(os.listdir(args.test_dir + "/0"))
  datagen = ImageDataGenerator()
  test_generator = datagen.flow_from_directory(directory = args.test_dir, target_size = (224, 224), batch_size = test_samples, class_mode = 'categorical', shuffle = False)
  
  model = make_model()
  model.load_weights(model_weights_path)
  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [pd(), nd()])

  results = model.predict_generator(generator = test_generator, steps = 1, verbose = 0)
 

  N = argN
  beta = N
  epsilon = 1e-6
  anchor = results[0::3]
  positive = results[1::3]
  negative = results[2::3]

  positive_distance = np.nansum(np.square(anchor - positive), axis = 1)
  positive_distance = - np.log(- (positive_distance / beta) + 1 + epsilon)

  positive_distance2 = np.nansum(np.square(anchor - negative), axis = 1)
  positive_distance2 = - np.log(- (positive_distance2 / beta) + 1 + epsilon)
 
  tp = 0
  fp = 0
  pneq = 0
  min_p = 10000
  max_p = 0
  min_n = 10000
  max_n = 0

  for i in range(test_samples // 3):
    pda = np.nansum(positive_distance[i])
    nda = np.nansum(positive_distance2[i])
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

  print(min_p, ' - ', max_p, ', ', min_n, ' - ', max_n)
  #print('accuracy: ', np.round(tp / (tp + fp) * 100, 1))
  #print('equal predictions: ', pneq)
  
  '''
  print(i, 'p ', np.nansum(positive_distance[i]))
  print(i, 'n ', np.nansum(positive_distance2[i]))
  I = anchor[i][0]
  img = Image.fromarray(np.array( (((I - I.min()) / (I.max() - I.min())) * 255.9), dtype=np.uint8))
  img.save(f'img/{i:03}testa.png')
  I = positive[i][0]
  img = Image.fromarray(np.array( (((I - I.min()) / (I.max() - I.min())) * 255.9), dtype=np.uint8))
  img.save(f'img/{i:03}testp.png')
  I = negative[i][0]
  img = Image.fromarray(np.array( (((I - I.min()) / (I.max() - I.min())) * 255.9), dtype=np.uint8))
  img.save(f'img/{i:03}testn.png')
  '''

if __name__ == '__main__':
  if args.train_model:
    train_model()
  elif args.test_model:
    test_model()   
