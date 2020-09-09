#####
prog_text = 'This program splits data to training and test sets and stores file lists under the temporary working directory.'
#####

# Basic packages
import argparse
import configparser
import json
import os
from pathlib import Path

from random import seed
from random import random

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
# 2. Process config file
config = configparser.ConfigParser()
config.read(config_file)

# Dataset to be processed
config_dataset = config['MAIN']['Dataset']

# Fixed seed guarantees the same output
if json.loads(config['MAIN']['RandSeed']): # not []
    seed(json.loads(config['MAIN']['RandSeed']))

# Temporary dir
temp_dir = config['MAIN']['TempDir']
if not os.path.exists(temp_dir):
    print("Making temporary working directory: ", temp_dir)
    os.makedirs(temp_dir)

# Check whether Male or Female
if config['MAIN']['Gender'] == 'Male':
    list_file = config[config_dataset]['MaleList']
else:
    list_file = config[config_dataset]['FemaleList']

#
# 3. Divide dataset files to training and testing
save_basename = os.path.splitext(os.path.basename(list_file))
train_file = Path(temp_dir, save_basename[0]+"_TRAIN"+".txt")
train_fid  = open(train_file,"w+")
test_file  = Path(temp_dir, save_basename[0]+"_TEST"+".txt")
test_fid   = open(test_file,"w+")

with open(list_file,'r') as file_in:
    for line in file_in:
        line = line.strip()
        if random() < json.loads(config[config_dataset]['UseForTraining']):
            train_fid.write(line + '\n')
        else:
            test_fid.write(line + '\n')
train_fid.close()
test_fid.close()
print('Training images written to ''%s''' %train_file)            
print('Test images written to ''%s''' %test_file)            
print('Done!')
