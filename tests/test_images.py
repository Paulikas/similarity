from keras import applications, optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Flatten, Dense
import numpy as np

import cv2


def triplet_loss(y_true, y_pred):
    alpha = 1
    embeddings = K.reshape(y_pred, (-1, 3))
    positive_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 1]) ** 2, axis=-1))
    negative_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 2]) ** 2, axis=-1))
    return K.sum(positive_distance - negative_distance + alpha)


img_width, img_height = 640, 480
top_model_weights_path = '/tmp/pycharm_project_199/top_model_weights.h5'
test_data_dir = '/opt/datasets/data/simulated_flight_1/test/'
nb_test_samples = 12
batch_size = 3
input_tensor = Input(shape=(img_width, img_height, 3))

vgg16_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

train_data = np.load(open('top_model_features_train.npy', 'rb'))

top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))


# model = Sequential()
# for l in vgg16_model.layers:
#     model.add(l)
# for l in top_model.layers:
#     model.add(l)

# model.load_weights("tuned_" + top_model_weights_path)
top_model.load_weights(top_model_weights_path)

top_model.compile(loss=triplet_loss, optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

# test_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(ImageDataGenerator(rescale = 1. / 255))

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_height, img_width),
                                                  batch_size = batch_size, class_mode = None, shuffle = False)

results = top_model.evaluate_generator(generator=test_generator)

top_model.predict()

img = cv2.imread(test_data_dir + "/0/image-0477.png")
template = cv2.imread(test_data_dir + "/0/image-0478.png")
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# print("NN similarity:", results)
print("OpenCV normed correlation:", res[0][0])


