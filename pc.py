import cv2
import glob
import re
import numpy as np
import sys

dataset_dir = '/opt/datasets/data/simulated_flight_1/train/'
test_data_dir = dataset_dir + '0/*'

all_files = glob.glob(test_data_dir)
all_files.sort()

triplet = 0
rp = []
rn = []

for filename in all_files:
  if triplet == 0:
    file_a = filename
    triplet += 1
  elif triplet == 1:
    file_b = filename
    triplet += 1
  elif triplet == 2:
    file_c = filename
    triplet += 1
  #elif triplet == 3:
    template = cv2.imread(file_a)
    img_b = cv2.imread(file_b)
    img_c = cv2.imread(file_c)

    template = cv2.resize(template, (150, 150))
    img_b = cv2.resize(img_b, (150, 150))
    img_c = cv2.resize(img_c, (150, 150))


    #template = template.astype('float32')
    #img_b = img_b.astype('float32')
    #img_c = img_c.astype('float32')

    #cv2.normalize(template, template, 0, 1, cv2.NORM_MINMAX)
    #cv2.normalize(img_b, img_b, 0, 1, cv2.NORM_MINMAX)
    #cv2.normalize(img_c, img_c, 0, 1, cv2.NORM_MINMAX)

    #print(template)
    
    
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    
    
    

    
    

    res_a = cv2.matchTemplate(img_b, template, cv2.TM_CCOEFF_NORMED)
    res_b = cv2.matchTemplate(img_c, template, cv2.TM_CCOEFF_NORMED)
    

    print(res_a.flatten(),'\t',res_b.flatten())
    
    rp.append(res_a)
    rn.append(res_b)

    triplet = 0

  

nrp = np.array(rp)
nrn = np.array(rn)

pos = np.sum(nrn.flatten() < nrp.flatten())
neg = np.sum(nrn.flatten() >= nrp.flatten())


print('acc :', np.round(pos / (pos + neg) * 100, 1), '%')
