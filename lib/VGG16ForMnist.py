import keras
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Input, Reshape
from keras.models import Model
from keras import optimizers, losses

from tensorflow.examples.tutorials import mnist
from keras.preprocessing.image import ImageDataGenerator

import cv2
import numpy as np

from sklearn.datasets import fetch_mldata
from keras.datasets import mnist


class VGG16ForMnist():
    # Bottom layer is VGG16
    vgg16_model = VGG16(include_top=False, weights='imagenet')
    top_model = None
    epochs = 1
    batch_size = 3
    trainable_vgg = False
    input_img_width, input_img_height = 224, 224


    """
    """
    def init_base_model(self, output_dims=10):
        # Model taken from VGG16, all middle layers will also be initialized
        # for more information please look https://www.slideshare.net/sujitpal/transfer-learning-and-fine-tuning-for-cross-domain-image-classification-with-keras
        base_model = Model(inputs=self.vgg16_model.input, outputs=self.vgg16_model.outputs)

        base_output = base_model.output
        # With standard input 224x224 output should be 25088, but it is waries depending on input size
        base_output = Reshape((25088,))(base_output)

        # Fix or unfix vgg layers for training
        for layer in base_model.layers[0:18]:
            layer.trainable = self.trainable_vgg

        # print(base_output.shape)
        # base_output_shape = base_output.shape[0]

        # Top layers
        top_fc1 = Dense(128, activation='relu')(base_output)
        top_fc2 = Dropout(0.5)(top_fc1)
        top_preds = Dense(output_dims, activation='softmax')(top_fc2)

        self.top_model = Model(inputs=base_model.input, outputs=top_preds)

        #
        # self.top_model = Sequential()
        # # self.top_model.add(Flatten(input_shape = base_output_shape[0]))
        # self.top_model.add(Dense(128, activation='relu'))
        # top_model.add(Dropout(0.5))
        # self.top_model.add(Dense(output_dims, activation='sigmoid'))

        # sgd_optimizer = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        sgd_optimizer = optimizers.adadelta()
        self.top_model.compile(optimizer=sgd_optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])

    def print_layers(self):
        # Network structure
        for i, layer in enumerate(self.top_model.layers):
            print(i, layer.name, layer.output_shape)

    def image_resize(self, img):
        new_img = cv2.resize(img, dsize=(self.input_img_width, self.input_img_height), interpolation=cv2.INTER_CUBIC)
        return new_img

    """
        Gets MNIST data set and convert to RGB images
    """
    def preprocess(self, mnist_data):
        n, d = mnist_data.shape
        images = np.empty((n, 224, 224, 3), dtype='uint8')
        for i in range(n):
            img = self.image_resize(mnist_data[i].reshape(28, 28))
            image = np.empty((224, 224, 3))
            image[:, :, 0] = img
            image[:, :, 1] = img
            image[:, :, 2] = img
            images[i, :, :, :] = image
            # if i > 0:
            #     np.append(images, image, axis=0)
            # else:
            #     images = np.asanyarray(image)
        return images


    def get_data_generator(self):
        mnist = fetch_mldata('MNIST original')
        print(mnist['data'].shape)
        # convert class vectors to binary class matrices
        # y_train = keras.utils.to_categorical(y_train, num_classes)
        X, y = self.preprocess(mnist['data'][1:5000]),keras.utils.to_categorical(mnist['target'][1:5000], 10)
        # X, y = mnist['data'][1:1000], mnist['target'][1:1000]

        datagen = ImageDataGenerator(rescale=1./225, validation_split=0.2, rotation_range=0)
        # generator = datagen.flow(X, y, shuffle=True, seed=17, batch_size=1, save_to_dir='/home/virginijus/tmp/pycharm_project_200/data/tmp/')
        generator = datagen.flow(X, y, shuffle=True, seed=17, batch_size=32*2)
        return generator


test = VGG16ForMnist()
generator =  test.get_data_generator()
test.init_base_model()
model = test.top_model
# test.print_layers()
# data = generator.next()
# print(data)
model.fit_generator(generator=generator, epochs=1, steps_per_epoch=100)

mnist = fetch_mldata('MNIST original')
x_test, y_test = test.preprocess(mnist['data'][60000:62000]),keras.utils.to_categorical(mnist['target'][60000:62000], 10)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# print(y_test[0])


