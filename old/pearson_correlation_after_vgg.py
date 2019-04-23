import cv2
import glob
import re
import numpy as np
from scipy import stats

valid_data = np.load(open('top_model_features_valid.npy', 'rb'))

triplet = 0

for i in valid_data:
  if triplet == 0:
    file_a = i
  if triplet == 1:
    file_b = i
  if triplet == 2:
    file_c = i


    res_a = stats.pearsonr(file_a.flatten(), file_c.flatten())


    res_b = stats.pearsonr(file_b.flatten(), file_c.flatten())

    triplet = 0

    print(res_a[0], res_b[0])
  triplet += 1


