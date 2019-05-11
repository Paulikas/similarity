import glob, os

dataset_dir = '/opt/datasets/data/simulated_flight_1/train/'
test_data_dir = dataset_dir + '0/*'

all_files = glob.glob(test_data_dir)
all_files.sort()

for filename in all_files[3456:3648]:
  os.rename(filename, filename.replace('/train/', '/test/'))
