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

def acc_metric(y_true, y_pred):
    return backend.mean(y_pred)

def triplet_loss(layer):

  def loss(y_true, y_pred):
    embeddings = y_pred
    positive_distance = backend.sum(backend.square(embeddings[0::3] - embeddings[1::3]))
    negative_distance = backend.sum(backend.square(embeddings[0::3] - embeddings[2::3]))
    return (positive_distance - negative_distance)
  return loss

img_width, img_height = 224, 224
top_model_weights_path = '../top_model_weights.h5'
#test_data_dir = '/opt/datasets/data/simulated_flight_1/valid/'
#nb_test_samples = 12
batch_size = 32

input_tensor = Input(shape=(7, 7, 512))

#vgg16_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

valid_data = np.load(open('../top_model_features_valid.npy', 'rb'))
#valid_labels = np.array([1, 1, 0] * (int(nb_valid_samples)))

top_model = Sequential()
top_model.add(Flatten(input_shape = valid_data.shape[1:]))
top_model.add(Dense(64, activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation = 'sigmoid', name='layer1'))


top_model.load_weights(top_model_weights_path)

sgd_optimizer = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
top_model.compile(optimizer = sgd_optimizer, loss = triplet_loss(top_model.get_layer('layer1')), metrics = [acc_metric])

# test_datagen = ImageDataGenerator()
#test_datagen = ImageDataGenerator()


#test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_height, img_width), batch_size = batch_size, class_mode='binary')

results = top_model.predict(x=valid_data)

print(results)

#pred = top_model.predict(train_data, batch_size=32)

'''
img = cv2.imread(test_data_dir + "/0/image-0477.png")
template = cv2.imread(test_data_dir + "/0/image-0478.png")
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# print("NN similarity:", results)
print("OpenCV normed correlation:", res[0][0])
'''

