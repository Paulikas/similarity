import tensorflow as tf
import numpy as np

from keras import backend, applications, optimizers, losses
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, mean_squared_error

import os.path
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version: " + tf.__version__)

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

#img_width, img_height = 224, 224
top_model_weights_path = 'top_model_weights.h5'
train_features_file = 'top_model_features_train.npy'
valid_features_file = 'top_model_features_valid.npy'

def save_features():

  datagen = ImageDataGenerator()
  
  vgg16_model = applications.VGG16(include_top = False, weights = 'imagenet')

  generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = None, shuffle = False)#, save_to_dir = 'train_augmented')
  top_model_features_train = vgg16_model.predict_generator(generator, nb_train_samples)
  np.save(open(train_features_file, 'wb'), top_model_features_train)

  generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = None, shuffle = False)#, save_to_dir = 'test_augmented')
  top_model_features_valid = vgg16_model.predict_generator(generator, nb_valid_samples)
  np.save(open(valid_features_file, 'wb'), top_model_features_valid)

# Nenaudojama y_true
# Klaida triplet loss funkcijoje

def triplet_loss(layer):

  def loss(y_true, y_pred):
    # It's a margin should be in range (0, 1)
    #alpha = 0.5
    #embeddings = backend.reshape(y_pred, (-1, 3))
    embeddings = y_pred

    positive_distance = backend.sum(backend.square(embeddings[0::3] - embeddings[1::3]))

    negative_distance = backend.sum(backend.square(embeddings[0::3] - embeddings[2::3]))

    #return backend.maximum(0.0, positive_distance - negative_distance + alpha)
    return (positive_distance - negative_distance)
   
  return loss


def acc_metric(y_true, y_pred):
    return backend.mean(y_pred)

def train_top_model():
  train_data = np.load(open('top_model_features_train.npy', 'rb'))
  train_labels = np.array([1, 1, 0] * (int(nb_train_samples)))
  # Klases visos nuliai tai accuracy nedaug ka gali pasakyti
  
  valid_data = np.load(open('top_model_features_valid.npy', 'rb'))
  valid_labels = np.array([1, 1, 0] * (int(nb_valid_samples)))

  top_model = Sequential()
  top_model.add(Flatten(input_shape = train_data.shape[1:]))
  top_model.add(Dense(64, activation = 'relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation = 'sigmoid', name='layer1'))


  # Batch size gali buti didesnis uz 3, nes gradientas apskaiciuojamas iverstinant daugiau kaimynu
  # lr reiktu didesnio negu buvo 0.00001, nes tuomet labai letai konverguos

  sgd_optimizer = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
  top_model.compile(optimizer = sgd_optimizer, loss = triplet_loss(top_model.get_layer('layer1')), metrics = [acc_metric])
  
  top_model.summary()

  # Pakeitus triplet_loss i losses.mean_squared_error modelis ismoksa taskirti pirmus du paveikslelius (1klase)
  # nuo 3 paveiklelio (0) klase. Tai reiskia kad sugeneruoti 1 klases paveiksleliai turi bendru paternu (Dense(64))

  # top_model.fit(train_data, train_labels, epochs = args.epochs, batch_size = args.batch_size*11, shuffle=True) # kodel 11?
  top_model.fit(train_data, train_labels, epochs = args.epochs, batch_size = args.batch_size, shuffle=True)

  top_model.save_weights(top_model_weights_path)

  # Su triplet_loss ismoksta viska vienai kalasiai priskirti ir tiek
  pred_valid = top_model.predict(valid_data, batch_size=32)

  # Cia gali buti klaida del neteisingo shape (1251, 20, 15, 512)
  print("train data shape:")
  print(train_data.shape)

  print("validation data shape")
  print(valid_data.shape)

  print("train labels:")
  print(train_labels)

  print("predictions valid:")
  print(pred_valid[0:9], pred_valid.shape)

  print('Validation MSE', mean_squared_error(valid_labels, pred_valid))
    
if __name__ == '__main__':
  nb_train_samples = len(os.listdir(args.train_dir + "/0")) / 3
  nb_valid_samples = len(os.listdir(args.valid_dir + "/0")) / 3
  if nb_train_samples > 0 and nb_valid_samples > 0:
    if args.w or not (os.path.isfile(train_features_file) and os.path.isfile(valid_features_file)):
      print("Writing features")
      save_features()
    train_top_model()
  else:
    print("Dataset images were not found")

# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
