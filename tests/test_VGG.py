from keras import applications
from keras.applications import VGG16

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import cv2

img_width = 224
img_height = 224
vgg16_model = VGG16(include_top = False, weights = 'imagenet')

# # Seems that weight are loaded when VGG16 objects is created
# weights = vgg16_model.get_weights()
# vgg16_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
# weights2 = vgg16_model.get_weights()

test_data_dir = '/opt/datasets/data/simulated_flight_1/test/'

img = cv2.imread(test_data_dir + "/0/image-0477.png")
template = cv2.imread(test_data_dir + "/0/image-0478.png")
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# plt.imshow(img)
# plt.show()

# Reshape
img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_CUBIC)
template = cv2.resize(template, dsize=(img_height, img_height), interpolation=cv2.INTER_CUBIC)

X = np.empty((2, img_height, img_width, 3))
np.append(X[0], np.asarray(img))
np.append(X[1], np.asarray(template))
# print(X[0, 0, 0, :])

print(img.shape, X.shape)

vgg16_features = vgg16_model.predict(X, verbose=True)
# On input image shape depends output shape
print("VGG output shape", vgg16_features.shape)
new_size = vgg16_features.shape[1] * vgg16_features.shape[2] * vgg16_features.shape[3]
vgg_res = np.corrcoef(vgg16_features[0].reshape((1, new_size)), vgg16_features[1].reshape((1, new_size)))

print("NN similarity:", vgg_res[0][1])
print("OpenCV normed correlation:", res[0][0])

# VGG network structure
for i, layer in enumerate(vgg16_model.layers):
    W = vgg16_model.layers[i].get_weights()
    W = np.asarray(W)
    if W.shape[0] > 0:
        print(i, layer.name, layer.output_shape, W[0].shape)
    else:
        print(i, layer.name, layer.output_shape)

print(vgg16_model.output)

# Summary
vgg16_model.summary()