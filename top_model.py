from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from keras import losses

import numpy as np
import os.path
from sklearn.metrics import confusion_matrix, mean_squared_error

import argparse

# CONFIGURATION #
parser = argparse.ArgumentParser(description='Build a top layer for the similarity training and train it.')
parser.add_argument('-w', '--overwrite', dest="w",
                    action='store_true', help="Overwrite initial feature embeddings")
parser.add_argument('-t', '--train-dir', dest='train_dir',
                    default="/opt/datasets/data/simulated_flight_1/train/",
                    help='Path to dataset training directory')
parser.add_argument('-v', '--valid-dir', dest='valid_dir',
                    default="/opt/datasets/data/simulated_flight_1/valid/",
                    help='Path to dataset validation directory')
parser.add_argument('-e', '--epochs', dest='epochs', type=int,
                    default=5, help='Number of epochs to train the model')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                    default=3, help='Batch size')

args = parser.parse_args()


# img_width, img_height = 150, 150
# vgg16 imput size 224, 224
img_width, img_height = 224, 224
top_model_weights_path = 'top_model_weights.h5'
train_data_dir = args.train_dir
valid_data_dir = args.valid_dir
train_features_file = 'top_model_features_train.npy'
valid_features_file = 'top_model_features_valid.npy'
nb_train_samples = len(os.listdir(train_data_dir + "/0")) / 3
nb_valid_samples = len(os.listdir(valid_data_dir + "/0")) / 3
epochs = args.epochs
batch_size = args.batch_size


def save_features():
  datagen = ImageDataGenerator(rescale = 1. / 255)

  vgg16_model = applications.VGG16(include_top = False, weights = 'imagenet')
  # Attention in generator width is swapped with height
  generator = datagen.flow_from_directory(train_data_dir, target_size = (img_height, img_width), batch_size = batch_size, class_mode = None, shuffle = False)
  top_model_features_train = vgg16_model.predict_generator(generator, nb_train_samples)
  # top_model_features_train = vgg16_model.predict_generator(generator, nb_train_samples // batch_size)
  np.save(open(train_features_file, 'wb'), top_model_features_train)

  generator = datagen.flow_from_directory(valid_data_dir, target_size = (img_height, img_width), batch_size = batch_size, class_mode = None, shuffle = False)
  # Kam antra karta dalinama is triju man nesuprantama
  # top_model_features_valid = vgg16_model.predict_generator(generator, nb_valid_samples // batch_size)
  top_model_features_valid = vgg16_model.predict_generator(generator, nb_valid_samples)
  np.save(open(valid_features_file, 'wb'), top_model_features_valid)

# Nenaudojama y_true
# Klaida triplet loss funkcijoje
def triplet_loss(y_true, y_pred):
  # It's a margin should be in range (0, 1)
  alpha = 0.5
  embeddings = K.reshape(y_pred, (-1, 3))
  # positive_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 1]) ** 2, axis = -1))
  # negative_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 2]) ** 2, axis = -1))
  positive_distance = K.sum(K.square(embeddings[:, 0] - embeddings[:, 1]), axis = -1)
  negative_distance = K.sum(K.square(embeddings[:, 0] - embeddings[:, 2]), axis = -1)

  print(positive_distance)
  return K.maximum(0.0, positive_distance - negative_distance + alpha)
  #return K.sum(positive_distance - negative_distance + alpha)


def train_top_model():
  train_data = np.load(open('top_model_features_train.npy', 'rb'))
  # Klases visos nuliai tai accuracy nedaug ka gali pasakyti
  # train_labels = np.array([0, 0, 1] * (int(nb_train_samples / batch_size)))
  train_labels = np.array([1, 1, 0] * (int(nb_train_samples)))

  valid_data = np.load(open('top_model_features_valid.npy', 'rb'))
  valid_labels = np.array([1, 1, 0] * (int(nb_valid_samples)))

  top_model = Sequential()
  top_model.add(Flatten(input_shape = train_data.shape[1:]))
  top_model.add(Dense(64, activation = 'relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation = 'sigmoid'))

  # Batch size gali būti didesnis uz 3, nes gradientas apskaičiuojamas iverstinant daugiau kaimynu
  # lr reiktu didesnio negu buvo 0.00001, nes tuomet labai letai konverguos

  sgd_optimizer = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
  top_model.compile(optimizer = sgd_optimizer, loss = triplet_loss, metrics = ['accuracy'])
  # Pakeitus triplet_loss i losses.mean_squared_error modelis ismoksa taskirti pirmus du paveikslelius (1klase)
  # nuo 3 paveiklelio (0) klase. Tai reiskia kad sugeneruoti 1 klases paveiksleliai turi bendru paternu (Dense(64))
  # top_model.compile(optimizer=sgd_optimizer, loss=losses.mean_squared_error, metrics=['accuracy'])

  # top_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
  top_model.fit(train_data, train_labels, epochs = epochs, batch_size = batch_size*11, shuffle=True)

  top_model.save_weights(top_model_weights_path)

  # Su triplet_loss ismoksta viska vienai kalasiai priskirti ir tiek
  pred_valid = top_model.predict(valid_data, batch_size=32)
  print(triplet_loss(None, pred_valid))

  # Cia gali buti klaida del neteisingo shape (1251, 20, 15, 512)
  print(train_data.shape)
  print(train_labels)
  print(pred_valid[0:9], pred_valid.shape)
  print('Validation MSE', mean_squared_error(valid_labels, pred_valid))
  print(valid_data.shape)

# MAIN #
if nb_train_samples > 0 and nb_valid_samples > 0:
  if args.w or not (os.path.isfile(train_features_file) and os.path.isfile(valid_features_file)):
    print("Writing features")
    save_features()
  train_top_model()
else:
  print("Dataset images were not found")

