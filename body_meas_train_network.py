#####
prog_text = 'This program trains a neural network that outputx body measurements for an input silhouette image.'
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
from tensorflow.keras.layers import Maximum, Input
from tensorflow.keras import Model
import threading

#
# 1. Process command line arguments (config file)

parser = argparse.ArgumentParser(description=prog_text)
parser.add_argument("-c", "--config", help="set configuration file")
parser.add_argument("-g", "--generated", help="process generated body files (true/false)",default="false")
parser.add_argument("-d", "--debug", help="produce debug information (true/false)",default="false")
parser.add_argument("-s","--imagesize",type=int,help="you can add specific image size", default=224)
parser.add_argument("-n","--network",type=int,help="you can select the used network", default=1)
parser.parse_args()

# Read arguments from the command line
args = parser.parse_args()

if args.config:
    config_file = args.config
else:
    raise Exception('You need to provide a configuration file!')

if args.debug == "true":
    debug = True
else:
    debug = False

if args.generated == "true":
    process_generated = True
else:
    process_generated = False

if args.imagesize == 224:
    image_size = 224
else:
    image_size = args.imagesize
    
if args.network == 1:
    network = 1
else:
    network = args.network
    

#
# 2. Process config file and set parameters
config = configparser.ConfigParser()
config.read(config_file)

# Dataset to be processed
config_dataset = config['MAIN']['Dataset']

# Check whether Male or Female
if config['MAIN']['Gender'] == 'Male':
    if process_generated:
        train_list_file = config[config_dataset]['MaleListTrainGenerated']
        SILH_DIR = config[config_dataset]['SilhDirMaleGenerated']
        MEAS_DIR = config[config_dataset]['MeasDirMaleGenerated']
    else:
        train_list_file = config[config_dataset]['MaleListTrain']
        SILH_DIR = Path(config['MAIN']['TempDir'], config['MAIN']['Dataset']+"_silhouettes")
        MEAS_DIR = Path(config['MAIN']['TempDir'], config['MAIN']['Dataset']+"_body_measurements")

    test_list_file = config[config_dataset]['MaleListTest']
else: # otherwise Female
    if process_generated:
        train_list_file = config[config_dataset]['FemaleListTrainGenerated']
        SILH_DIR = config[config_dataset]['SilhDirFemaleGenerated']
        MEAS_DIR = config[config_dataset]['MeasDirFemaleGenerated']
    else:
        train_list_file = config[config_dataset]['FemaleListTrain']
        SILH_DIR = Path(config['MAIN']['TempDir'], config['MAIN']['Dataset']+"_silhouettes")
        MEAS_DIR = Path(config['MAIN']['TempDir'], config['MAIN']['Dataset']+"_body_measurements")

    test_list_file = config[config_dataset]['FemaleListTest']

# List of all silhouette images used for training
with open(train_list_file, 'r') as f:
    train_img_list = f.readlines()
tot_train_count = len(train_img_list)
train_img_list = [x.strip() for x in train_img_list]


temp_dir = config['MAIN']['TempDir']

valid_measurements = config['CAESAR']['BodyMeasurements']
valid_measurements = valid_measurements.split()


#
# 3. Prepare training data

def preprocess_img(img_file):
    # Normalize values to [0,1]
    img = Image.open(img_file)
    img = np.array(img)
    img = img / 255

    return np.expand_dims(img, axis=-1)


config_network = config['MAIN']['Network']

if config[config_network]['BatchTraining'] == 'true':
    # Define data generator class
    class data_generator:
        def __init__(self, silh_list, batch_size, silh_dir, meas_dir):
            self.silh_list = silh_list
            # Check image size and number of measurements
            self.front_dir = Path(silh_dir, 'front')
            self.side_dir = Path(silh_dir, 'side')
            self.meas_dir = meas_dir

            # check image size and num of outputs
            file_base = os.path.splitext(os.path.basename(self.silh_list[0][:]))
            img_file = Path(self.front_dir, file_base[0] + "_silhouette" + ".png")
            meas_file = Path(self.meas_dir, file_base[0] + "_BODY_MEAS_" + ".txt")
            img = Image.open(img_file)
            self.img_size = img.size
            with open(meas_file,'r') as f:
                meas_lines = f.readlines()
            self.num_of_outputs = len(meas_lines)
            self.num_of_samples = len(silh_list)
            self.list_inds = range(len(silh_list))
            self.batch_size = batch_size
            self.batch_idx = 0
            self.lock = threading.Lock()
        def __iter__(self):
            return self
        def __next__(self):
            with self.lock:
                inds = range(self.batch_idx*self.batch_size, (self.batch_idx + 1)*self.batch_size)
                inds = [i % self.num_of_samples for i in inds]
                inds = [self.list_inds[i] for i in inds]
                self.batch_idx += 1

                X_f = np.empty((self.batch_size, self.img_size[1], self.img_size[0], 1), dtype = np.float32)
                X_s = np.empty((self.batch_size, self.img_size[1], self.img_size[0], 1), dtype = np.float32)
                Y = np.empty((self.batch_size, self.num_of_outputs), dtype = np.float32)
                # Read in each input, perform preprocessing and get labels
                for i, ind in enumerate(inds):
                    # img_dir = os.path.dirname(self.silh_list[0][:])
                    file_base = os.path.splitext(os.path.basename(self.silh_list[ind][:]))
                    front_img_file = Path(self.front_dir, file_base[0] + "_silhouette" + ".png")
                    side_img_file = Path(self.side_dir, file_base[0] + "_silhouette" + ".png")
                    meas_file = Path(self.meas_dir, file_base[0] + "_BODY_MEAS_" + ".txt")

                    X_f[i, ...] = preprocess_img(front_img_file)
                    X_s[i, ...] = preprocess_img(side_img_file)

                    with open(meas_file,'r') as f:
                        meas_lines = f.readlines()
                    Y_this = [None]*len(valid_measurements)
                    for meas_line in meas_lines:
                        meas_line = meas_line.split()
                        for idx, meas in enumerate(valid_measurements):
                            if meas_line[2] == meas:
                                Y_this[idx] = float(meas_line[3])
                                #Y.append(Y_this)
                        Y[i, :]   = np.asarray(Y_this)
                if config_network == 'NET_SIMPLE_JONI':
                    return [X_f], Y
                elif config_network == 'NET_CAESAR':
                        return [X_f, X_s], Y
                else:
                    print(f"Batch input not defined for network called {config_network} !!")
else: # read all data
    if process_generated:
        if config['MAIN']['Gender'] == 'Male':
            silhouettes_dir = config[config_dataset]['SilhDirMaleGenerated']
            measurements_dir = config[config_dataset]['MeasDirMaleGenerated']
        else:
            silhouettes_dir = config[config_dataset]['SilhDirFemaleGenerated']
            measurements_dir = config[config_dataset]['MeasDirFemaleGenerated']
    else:
        silhouettes_dir = Path(temp_dir, config['MAIN']['Dataset']+"_silhouettes")
        measurements_dir = Path(temp_dir, config['MAIN']['Dataset']+"_body_measurements")

    #X = np.empty((tot_train_count, config['MAIN']['SilhouetteWidth'], config['MAIN']['SilhouetteHeight'], 1), dtype = np.float32)
    #Y = np.empty((tot_train_count, len(self.classes)), dtype = np.float32)
    X_f, X_s, Y = [], [], []

    for img_ind in range(tot_train_count):
        print("\r Reading training data %4d/%4d" %(img_ind+1,len(train_img_list)), end=" ")
        file_base = os.path.splitext(os.path.basename(train_img_list[img_ind][:]))
        front_img_file = Path(silhouettes_dir, 'front', config_dataset + "_silhouette_" + file_base[0] + ".png")
        side_img_file = Path(silhouettes_dir, 'side', config_dataset + "_silhouette_" + file_base[0] + ".png")
        meas_file = Path(measurements_dir, config_dataset + "_BODY_MEAS_" + file_base[0] + ".txt")

        img_f = preprocess_img(front_img_file)
        img_s = preprocess_img(side_img_file)
        X_f.append(img_f)
        X_s.append(img_s)

        with open(meas_file,'r') as f:
            meas_lines = f.readlines()
        Y_this = [None]*len(valid_measurements)
        for meas_line in meas_lines:
            meas_line = meas_line.split()
            for idx, meas in enumerate(valid_measurements):
                if meas_line[2] == meas:
                    Y_this[idx] = float(meas_line[3])
        Y.append(Y_this)

    X_f = np.asarray(X_f)
    X_s = np.asarray(X_s)
    X_f = np.expand_dims(X_f, axis=-1)
    X_s = np.expand_dims(X_s, axis=-1)
    print(X_s.shape)
    if config_network == 'NET_SIMPLE_JONI':
        X = X_f # frontal only used
    elif config_network == 'NET_CAESAR':
        X_f = X_f # Do dummy something
    else:
        print(f"Batch input not defined for network called {config_network} !!")
    Y = np.asarray(Y)

#
# 4. Load the network
batch_size = json.loads(config[config_network]['BatchSize'])
epochs = json.loads(config[config_network]['Epochs'])

# Define networks and their parameters (could be in separate files)
if config_network == 'NET_SIMPLE_JONI':
    if (network == 1):
        model = Sequential([
            Conv2D(16, 5, padding='same', activation='relu', input_shape=(image_size, image_size, 1)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(len(valid_measurements))
        ])
    if (network == 2):
          model = Sequential([
            Conv2D(16, 5, padding='same', activation='relu', input_shape=(image_size, image_size, 1)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(len(valid_measurements))
        ])
    if (network == 3):
          model = Sequential([
            Conv2D(16, 5, padding='same', activation='relu', input_shape=(image_size, image_size, 1)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(len(valid_measurements))
        ])   
  

    num_of_epochs = json.loads(config[config_network]['Epochs'])
    learning_rate = json.loads(config[config_network]['LearningRate'])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.summary()

elif config_network == 'NET_CAESAR':
    # raise Exception('Unknown network:' + config_network)
    input_f = Input(shape=(image_size, image_size, 1))
    x = Conv2D(64, 11, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(input_f)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(192, 5, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(384, 3, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(x)
    x = Conv2D(384, 3, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    # conv_f = Model(inputs=input_f, outputs=x)

    input_s = Input(shape=(image_size, image_size, 1))
    y = Conv2D(64, 11, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(input_s)
    y = MaxPooling2D(pool_size=(3,3))(y)
    y = Conv2D(192, 5, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(y)
    y = MaxPooling2D(pool_size=(3,3))(y)
    y = Conv2D(384, 3, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(y)
    y = Conv2D(384, 3, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(y)
    y = Conv2D(256, 3, padding='same', kernel_initializer=tf.initializers.GlorotUniform(), activation='relu')(y)
    y = MaxPooling2D(pool_size=(3, 3))(y)
    # conv_s = Model(inputs=input_s, outputs=y)

    combined = Maximum()([x, y])

    z = Conv2D(4096, 5, padding='valid')(combined)
    z = Dropout(0.5)(z)
    z = Conv2D(4096, 1)(z)
    z = tf.reduce_mean(z, [1,2], keepdims=True)
    z = Conv2D(len(valid_measurements), 1)(z)
    z = tf.squeeze(z, [1, 2])

    model = Model(inputs=[input_f, input_s], outputs=z)

    num_of_epochs = json.loads(config[config_network]['Epochs'])
    learning_rate = json.loads(config[config_network]['LearningRate'])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.summary()

if config[config_network]['BatchTraining'] == 'true':
    # print(train_list_file)
    print('using BatchTraining')
    batch_size = json.loads(config[config_network]['BatchSize'])
    train_generator = data_generator(train_img_list, batch_size, SILH_DIR, MEAS_DIR)
    history = model.fit_generator(train_generator, epochs=num_of_epochs,
                    steps_per_epoch=round(tot_train_count/batch_size))
else:
    print(' not using BatchTraining')
    history = model.fit(X,Y,epochs=5)

# Save model
model.save(Path(temp_dir,config[config_network]['SaveName']))


...
## list all data in history
#print(history.history.keys())
#print(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_error'])
##plt.plot(history.history['val_accuracy'])
#plt.title('model performance')
#plt.ylabel('MAE')
#plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
#plt.legend(['train'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
#plt.legend(['train'], loc='upper left')
#plt.show()
