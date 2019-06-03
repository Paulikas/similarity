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

from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

import argparse

test_dir_a = '/opt/datasets/data/simulated_flight_1/test_a'
test_dir_p = '/opt/datasets/data/simulated_flight_1/test_p'
test_dir_n = '/opt/datasets/data/simulated_flight_1/test_n'

from top import make_model, triplet_loss, pd, nd

def get_test_triplet():
  gen = ImageDataGenerator()
  gen_a = gen.flow_from_directory(directory = test_dir_a, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
  gen_p = gen.flow_from_directory(directory = test_dir_p, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
  gen_n = gen.flow_from_directory(directory = test_dir_n, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
  an = gen_n.next()
  po = gen_a.next()
  ne = gen_p.next()
  # po = gen_n.next()
  # ne = gen_n.next()
  yield [an[0], po[0], ne[0]], an[1]


# model = make_model()
# # model = tf.keras.experimental.load_from_saved_model(checkpoint_dir)
# # model = keras.models.load_model(f'runtime_files/train_model.h5')
# model.load_weights('runtime_files/train_model.h5')
#
# # model.summary()
#
# # get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers])
#
# model.predict_generator(generator=get_test_triplet(), steps=1)
# print(layer_dict)
#
# sess = tf.Session()
# pred = model.predict_generator(generator=get_test_triplet(), steps=1)
# tp = triplet_loss(64)
# r = tp([], tf.convert_to_tensor(pred))
#
# pd1 = pd(64)
# r1 = pd1([], tf.convert_to_tensor(pred))
#
# nd1 = nd(64)
# r2 = nd1([], tf.convert_to_tensor(pred))
#
#
# print(K.sum(r), K.sum(r1), K.sum(r2))
#
# print(sess.run(r1), sess.run(r2))

from top import test_model, args

args.lc = 0
from top import  argN
print(f'argN = {argN}')
test_model()



