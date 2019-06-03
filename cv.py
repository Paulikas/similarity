import glob, os
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import json, h5py, shutil, sys
from keras import backend as K

from top import train_model, test_model, args, checkpoint_dir

folds_no = 4
indexes = np.arange(3756).reshape((3756 // 3, 3))

# epochs       = [1, 4, 16, 50, 100, 200, 400]
# lcs          = [0, 4, 8, 12]
# batch_sizes  = [3, 6, 12, 24, 48, 96, 192]


epochs       = [1, 4, 16, 50]
lcs          = [0, 4, 8, 12]
batch_sizes  = [3, 6, 12, 24, 48, 96]

# epochs       = [1, 4, 16]
# lcs          = [0]
# batch_sizes  = [96]

for epoch in epochs:
  for lc in lcs:
    for batch_size in batch_sizes:
      print(f'Epocs = {epoch}, lc = {lc}, bach sizes = {batch_size}')

      test_filename = f"/home/virginijus/tmp/pycharm_project_202/numta_experiments/res_l_{lc}_e_{epoch}_b_{batch_size}"
      # Remove old test file
      if os.path.isfile(test_filename):
        # os.remove(test_filename)
        continue

      fold = 1
      kf = KFold(n_splits=folds_no, shuffle=True)
      for train_index, test_index in kf.split(indexes):
        print(f'Fold = {fold} \n')

        test_files = np.concatenate(indexes[test_index])

        dataset_dir = '/opt/datasets/data/simulated_flight_1/test/'
        test_data_dir = dataset_dir + '0/*'

        # Reset to default (all test data moved to train)
        all_files = glob.glob(test_data_dir)
        all_files.sort()

        for filename in all_files:
          os.rename(filename, filename.replace('/test/', '/train/'))

        # Prepare some data for testing
        dataset_dir = '/opt/datasets/data/simulated_flight_1/train/'
        test_data_dir = dataset_dir + '0/*'

        all_files = glob.glob(test_data_dir)
        all_files.sort()

        for filename in [all_files[i] for i in test_files]:
          os.rename(filename, filename.replace('/train/', '/test/'))

        os.system(f"/home/virginijus/tensorflow/bin/python top.py -r -e {epoch } -l {lc} -b {batch_size}")

        # Testing
        print(f"Testing - {test_filename}")
        os.system(f"/home/virginijus/tensorflow/bin/python top.py -s -l {lc} >> {test_filename}")

        fold += 1
