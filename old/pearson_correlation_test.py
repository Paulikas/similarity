import cv2
import glob
import re
from scipy import stats
import numpy as np

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

    res_a = stats.pearsonr(template.flatten(), img_b.flatten())
    res_b = stats.pearsonr(template.flatten(), img_c.flatten())
    print(res_a[0], res_b[0])

    res_a = np.corrcoef(template.flatten(), img_b.flatten())
    res_b = np.corrcoef(template.flatten(), img_c.flatten())
    print(res_a[0][1], res_b[0][1])
    
    res_a = cv2.matchTemplate(img_b, template, cv2.TM_CCORR_NORMED)
    res_b = cv2.matchTemplate(img_c, template, cv2.TM_CCORR_NORMED)
    print(1-res_a[0][0], 1-res_b[0][0])

    print()

    triplet = 0

  triplet += 1
