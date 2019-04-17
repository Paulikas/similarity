from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Reshape,Input
from keras.applications import VGG16
from keras import optimizers
from keras import backend as K

import tensorflow as tf
import numpy as np
import os.path


"""
WGG16 with custom top layer

This model should be dedicate to find similarities between two images.
Or in other words, it should be better than Pearson correlation, e.g invariant to rotation and shift 
"""
class VGG16WithTopLayer():
    # Bottom layer is VGG16
    vgg16_model = VGG16(include_top=False, weights='imagenet')
    top_model = None
    epochs = 1
    batch_size = 3
    trainable_vgg = False

    """
    Supposes that each third row of X is
        y
        X_true
        X_false
    """
    # def fit(self, X):
    #     y, X_true, X_false =


    def fit(self, X_true, X_false, y):
        X = self.vgg16_model.output

        vgg16_X_true = self.vgg16_model.predict(X_true)
        vgg16_X_false = self.vgg16_model.predict(X_false)
        vgg16_y = self.vgg16_model.predict(y)
        train_data = [vgg16_y, vgg16_X_true, vgg16_X_false]
        train_labels = [[0, 0, 1] * vgg16_y.shape[0]]

        self.init_model()
        self.top_model.fit(train_data, train_labels, epochs=self.epochs, batch_size=self.batch_size)
        # top_model.save_weights(top_model_weights_path)

    def init_base_model(self, output_dims=16):
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
        top_preds = Dense(output_dims, activation='selu')(top_fc2)

        self.top_model = Model(inputs=base_model.input, outputs=top_preds)

        #
        # self.top_model = Sequential()
        # # self.top_model.add(Flatten(input_shape = base_output_shape[0]))
        # self.top_model.add(Dense(128, activation='relu'))
        # top_model.add(Dropout(0.5))
        # self.top_model.add(Dense(output_dims, activation='sigmoid'))

        sgd_optimizer = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        # self.top_model.compile(optimizer=sgd_optimizer, loss=self.triplet_loss, metrics=['accuracy'])
        self.top_model.compile(optimizer=sgd_optimizer, loss=self.triplet_loss, metrics=[])

    """
    Code taken from example https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    @TODO still not working
    """
    def init_model(self):
        self.init_base_model()
        in_dims = self.top_model.input.shape
        # in_dims = (None, None, None, 3)
        in_dims = (3, 3, 3, 3, 3, 64)

        # Create the 3 inputs
        anchor_in = Input(shape=in_dims)
        pos_in = Input(shape=in_dims)
        neg_in = Input(shape=in_dims)

        # Share base network with the 3 inputs
        base_network = self.top_model
        anchor_out = base_network(anchor_in)
        pos_out = base_network(pos_in)
        neg_out = base_network(neg_in)
        merged_vector = K.concatenate([anchor_out, pos_out, neg_out], axis=1)

        # Define the trainable model
        model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)
        model.compile(optimizer=optimizers.Adam(),
                      loss=self.triplet_loss)


    """
     Code was taken from https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    """
    def triplet_loss(y_true, y_pred, alpha=0.5):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        embeddings = K.reshape(y_pred, (-1, 3))

        anchor  = embeddings[:, 0]
        positive = embeddings[:, 1]
        negative = embeddings[:, 2]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)

        # compute loss
        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)

        return loss

    def print_layers(self):
        # Network structure
        for i, layer in enumerate(self.top_model.layers):
            print(i, layer.name, layer.output_shape)


model = VGG16WithTopLayer()
model.init_base_model()
model.print_layers()
# model.init_model()
# model.top_model.fit()

# Show different statistics of the model
# model.top_model.summary()

valid_data_dir = '/opt/datasets/data/simulated_flight_1/valid/'
img_width, img_height = 224, 224

batch_size = 32

datagen = ImageDataGenerator(rescale = 1. / 255)
generator = datagen.flow_from_directory(valid_data_dir, target_size = (img_height, img_width), batch_size = 6, class_mode = 'binary', shuffle = False)

img = generator.next()
# print(generator.next())
nb_valid_samples = len(os.listdir(valid_data_dir + "/0")) / 3
valid_labels = np.array([1, 1, 0] * (int(nb_valid_samples)))

# X_valid = generator

result = model.top_model.fit_generator(generator=generator, epochs=30, steps_per_epoch=5)
# model.top_model.predict_generator()

generator = datagen.flow_from_directory(valid_data_dir, target_size = (img_height, img_width), batch_size = 1, class_mode = 'binary', shuffle = False)
pred = model.top_model.predict_generator(generator=generator, steps=10)

print(pred)
print(np.linalg.norm(pred[0]-pred[1]))
print(np.linalg.norm(pred[0]-pred[2]))
print(np.linalg.norm(pred[3]-pred[4]))
print(np.linalg.norm(pred[3]-pred[5]))
