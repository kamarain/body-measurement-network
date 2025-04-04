#####
prog_text = 'This program tests a neural network.'
#####

# Generic data flow and I/O
import argparse
import configparser
import json
import os
from pathlib import Path

# Image processing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Deep learning done using Keras on top of the TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

#
# 1. Process command line arguments (config file)

parser = argparse.ArgumentParser(description=prog_text)
parser.add_argument("-c", "--config", help="set configuration file")
parser.parse_args()

# Read arguments from the command line
args = parser.parse_args()

if args.config:
    config_file = args.config
else:
    raise Exception('You need to provide a configuration file!')


#
# 2. Process config file and set parameters
config = configparser.ConfigParser()
config.read(config_file)

# Dataset to be processed
config_dataset = config['MAIN']['Dataset']

# Check whether Male or Female
if config['MAIN']['Gender'] == 'Male':
    test_list_file = config[config_dataset]['MaleListTest']
    data_list_file = config[config_dataset]['MaleList']
else: # otherwise Female
    test_list_file = config[config_dataset]['FemaleListTest']
    data_list_file = config[config_dataset]['FemaleList']

# List of all silhouette images used for testing
with open(test_list_file, 'r') as f:
    test_img_list = f.readlines()
tot_test_count = len(test_img_list)
test_img_list = [x.strip() for x in test_img_list]

temp_dir = config['MAIN']['TempDir']

valid_measurements = config['CAESAR']['BodyMeasurements']
valid_measurements = valid_measurements.split()

config_network = config['MAIN']['Network']

#
# 3. Test data one by one

def preprocess_img(img_file):
    # Normalize values to [0,1]
    img = Image.open(img_file)
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img # (1, 224, 224, 1)

model = tf.keras.models.load_model(Path(temp_dir,config[config_network]['SaveName']))
data_dir_img = Path(temp_dir, config['MAIN']['Dataset']+"_silhouettes")
front_img_dir = Path(data_dir_img, 'front')
side_img_dir = Path(data_dir_img, 'side')

data_dir_meas = Path(temp_dir, config['MAIN']['Dataset']+"_body_measurements")

Y_pr = np.empty((len(test_img_list),len(valid_measurements)), dtype = np.float32)
Y_gt = np.empty((len(test_img_list),len(valid_measurements)), dtype = np.float32)
for i, file in enumerate(test_img_list):
    print("\r Reading test data %4d/%4d" %(i+1,len(test_img_list)), end=" ")
    # As set in body_meas_render_silhouettes (for not generated)
    body_file = os.path.splitext(os.path.basename(file))

    silh_front_file_name = Path(front_img_dir, config_dataset + "_silhouette_" + body_file[0] + ".png")
    silh_side_file_name = Path(side_img_dir, config_dataset + "_silhouette_" + body_file[0] + ".png")

    img_f = preprocess_img(silh_front_file_name)
    img_s = preprocess_img(silh_side_file_name)

    # Format network compatible input (this is a bit messy)
    if config_network == 'NET_SIMPLE_JONI':
        Y_pr[i,:] = model.predict([img_f])
    elif config_network == 'NET_CAESAR':
        Y_pr[i,:] = model.predict([img_f, img_s])
    elif config_network == 'NET_JORI':
        Y_pr[i,:] = model.predict([img_f, img_s])
    else:
        print(f"Batch input not defined for network called {config_network} !!")

        
    # As set in body_meas_compute_measurements (for not generated)
    meas_file = os.path.splitext(os.path.basename(file))
    meas_save_file = Path(data_dir_meas, config_dataset + "_BODY_MEAS_" + meas_file[0] + ".txt")

    with open(meas_save_file,'r') as f:
        meas_lines = f.readlines()
        Y_this = [None]*len(valid_measurements)
        for meas_line in meas_lines:
            meas_line = meas_line.split()
            for idx, meas in enumerate(valid_measurements):
                if meas_line[2] == meas:
                    Y_this[idx] = float(meas_line[3])
                    #Y.append(Y_this)
    Y_gt[i,:] = Y_this
print("... Done!\n")

# Per test image error (MAE: Mean Absolute Error)
avg_mae = abs(Y_gt-Y_pr)
# Average error over all test images (average MAE)
avg_mae = np.average(avg_mae, axis=0)
avg_val = np.average(Y_gt, axis=0)

#
# Print results
print('--- [Measure name]    [Avg value in mm]   [Mean error in mm] ---\n')
for m in range(len(valid_measurements)):
    print('%20s Val=%4.2f mae=%4.2f' %(valid_measurements[m],avg_val[m],avg_mae[m]))
print('----------------------------------------------------------------\n')
