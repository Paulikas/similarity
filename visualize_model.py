import tensorflow as tf
import numpy as np

from time import time
from PIL import Image

from tensorflow import saved_model
from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework import ops

from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

from top import make_model, triplet_loss, nd, pd
checkpoint_dir = 'runtime_files/saved_model'

argN = 64

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def plot_filters(layer, x, y):

    filters = layer.W.get_values()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j+1)
        ax.matshow(filters[j][0], cmap = plt.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt

from top import test_dir_a, test_dir_n, test_dir_p
def get_test_triplet():
  gen = ImageDataGenerator()
  gen_a = gen.flow_from_directory(directory = test_dir_a, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
  gen_p = gen.flow_from_directory(directory = test_dir_p, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
  gen_n = gen.flow_from_directory(directory = test_dir_n, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
  an = gen_n.next()
  po = gen_p.next()
  ne = gen_a.next()
  po = gen_n.next()
  ne = gen_n.next()
  yield [an[0], po[0], ne[0]], an[1]


model = make_model()
# model = tf.keras.experimental.load_from_saved_model(checkpoint_dir)
# model = keras.models.load_model(f'runtime_files/train_model.h5')
model.load_weights('runtime_files/train_model.h5')

# model.summary()

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

model.predict_generator(generator=get_test_triplet(), steps=1)
print(layer_dict)

pred = model.predict_generator(generator=get_test_triplet(), steps=1)
tp = triplet_loss(64)
r = tp([], tf.convert_to_tensor(pred))

pd1 = pd(64)
r1 = pd1([], tf.convert_to_tensor(pred))

nd1 = nd(64)
r2 = nd1([], tf.convert_to_tensor(pred))

print(K.sum(r), K.sum(r1), K.sum(r2))