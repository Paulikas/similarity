from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy import ndimage

valid_data_dir = '/opt/datasets/data/simulated_flight_1/valid/'

modified_valid_data_dir = '/home/virginijus/tmp/pycharm_project_200/data/valid/0/'
valid_data_file = 'images_valid.npy'
# img_width, img_height = 224, 224
img_width, img_height = 640, 480
target_img_width, target_img_height = 224, 224
batch_size = 32

data_dir = '/home/virginijus/tmp/pycharm_project_200/data/'

def rotate_image(image, angle):
  # image_center = tuple(np.array(image.shape[0:2])/2)
  image_center = tuple(np.array((img_width, img_height)) / 2)

  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  # result = cv2.warpAffine(image, rot_mat, image.shape[0:2], flags=cv2.INTER_CUBIC)
  result = cv2.warpAffine(image, rot_mat, (img_width, img_height), flags=cv2.INTER_LINEAR)
  return result

# datagen = ImageDataGenerator(rescale = 1. / 255, rotation_range=90.)
datagen = ImageDataGenerator()
# Mixed width with height
generator = datagen.flow_from_directory(valid_data_dir, target_size = (img_height, img_width), batch_size = 1,
                                        class_mode = None, shuffle = False)

generator.save_to_dir = modified_valid_data_dir

# for i in range(generator.n):
for i in range(1):
    img_set = generator.next()
    img = img_set[0,:,:,:]
    for angle in range(360):
        tmp_img = np.copy(img)
        # bitmap = cv2.fromarray(tmp_img)
        # bitmap = Image.fromarray(tmp_img, mode='RGB')
        rotated_img = rotate_image(tmp_img, angle)
        cropped_img = rotated_img[(img_width - target_img_width) // 2:(img_width + target_img_width) // 2,(img_height - target_img_height) // 2:(img_height + target_img_height) // 2]
        # rotated_img = tmp_img

        filename = '{}-{}.jpg'.format(i, angle)
        cv2.imwrite(data_dir + '/rotated/' + filename, cropped_img)

    # img.reshape((img_width, img_height, 3))
    # print(img.shape)
    # plt.imshow(img)
    # np.save(open(valid_data_file, 'ab'), img)

# images = np.load(open(valid_data_file, 'rb'))
# images = np.load(valid_data_file, 'rb)
# print(images.shape)