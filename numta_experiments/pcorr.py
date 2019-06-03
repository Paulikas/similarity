import cv2
import glob
import re
from scipy import stats
import numpy as np
import io

# dataset_dir = '/opt/datasets/data/simulated_flight_1/valid/'
dataset_dir = '/opt/datasets/data/simulated_flight_1/original_train/'
test_data_dir = dataset_dir + '0/*'

all_files = glob.glob(test_data_dir)
all_files.sort()

triplet = 0
tp = 0
fp = 0
n = 0
for filename in all_files:
    # print(filename)
    if triplet == 0:
        file_a = filename
    elif triplet == 1:
        file_b = filename
    elif triplet == 2:
        file_c = filename

        template = cv2.imread(file_a)

        img_b = cv2.imread(file_b)
        img_c = cv2.imread(file_c)

        # template = cv2.resize(template, (224, 224))
        # img_b = cv2.resize(img_b, (224, 224))
        # img_c = cv2.resize(img_c, (224, 224))

        template = cv2.resize(template, (150, 150))
        img_b = cv2.resize(img_b, (150, 150))
        img_c = cv2.resize(img_c, (150, 150))

        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        #res_a = stats.pearsonr(template.flatten(), img_b.flatten())
        #res_b = stats.pearsonr(template.flatten(), img_c.flatten())
        #print(res_a[0], res_b[0])

        #res_a = np.corrcoef(template.flatten(), img_b.flatten())
        #res_b = np.corrcoef(template.flatten(), img_c.flatten())
        #print(res_a[0][1], res_b[0][1])

        # res_a = cv2.matchTemplate(template, img_b, cv2.TM_CCORR_NORMED)
        # res_b = cv2.matchTemplate(template, img_c, cv2.TM_CCORR_NORMED)

        res_a = cv2.matchTemplate(img_b, template, cv2.TM_CCOEFF_NORMED)
        res_b = cv2.matchTemplate(img_c, template, cv2.TM_CCOEFF_NORMED)
        print(res_a[0][0], "\t", res_b[0][0])

        if res_a[0][0] > res_b[0][0]:
            tp += 1
        else:
            fp += 1
        n += 1

        triplet = -1

    triplet += 1

print(f'Accracy: {tp / n}')