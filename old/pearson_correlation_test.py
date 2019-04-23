import cv2
import glob
import re

dataset_dir = '/opt/datasets/data/simulated_flight_1/valid/'
test_data_dir = dataset_dir + '0/*'

all_files = glob.glob(test_data_dir)
all_files.sort()

triplet = 0

for filename in all_files:
  if triplet == 0:
    file_a = filename
  elif triplet == 1:
    file_b = filename
  elif triplet == 2:
    file_c = filename

    template = cv2.imread(file_a)
    img_b = cv2.imread(file_b)
    img_c = cv2.imread(file_c)

    res_a = cv2.matchTemplate(img_b, template, cv2.TM_CCORR_NORMED)
    res_b = cv2.matchTemplate(img_c, template, cv2.TM_CCORR_NORMED)

    triplet = 0

    print(res_a[0][0], res_b[0][0])
  triplet += 1
