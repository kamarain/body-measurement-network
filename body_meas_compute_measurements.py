#####
prog_text = 'This program calculates body measurements on pre-defined 3D body paths.'
#####

# Basic packages
import argparse
import configparser
import json
from pathlib import Path

# Computation packages
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import numpy as np
import os

#
# 1. Process command line arguments (config file)

parser = argparse.ArgumentParser(description=prog_text)
parser.add_argument("-c", "--config", help="set configuration file")
parser.add_argument("-d", "--debug", help="produce debug information (true/false)",default="false")
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

#
# 2. Process config file and set parameters
config = configparser.ConfigParser()
config.read(config_file)

# Dataset to be processed
config_dataset = config['MAIN']['Dataset']

#
config_use_generated = json.loads(config[config_dataset]['UseGenerated'])

# Check whether Male or Female
if config['MAIN']['Gender'] == 'Male':
    data_list_file = config[config_dataset]['MaleList']
    if config_use_generated:
        gen_data_list_file = config[config_dataset]['MaleListTrainGenerated']
        gen_meas_dir       = config[config_dataset]['MeasDirMaleGenerated']
        print(" -> Processing also generated training data.")
else: # otherwise Female
    data_list_file = config[config_dataset]['FemaleList']
    if config_use_generated:
        gen_data_list_file = config[config_dataset]['FemaleListTrainGenerated']
        gen_meas_dir       = config[config_dataset]['MeasDirFemaleGenerated']
        print(" ->Processing also generated training data.")

# Dir where to store generated measurements
temp_dir = config['MAIN']['TempDir']
if not os.path.exists(temp_dir):
    print("Making temporary working directory: ", temp_dir)
    os.makedirs(temp_dir)
data_dir = Path(temp_dir, config['MAIN']['Dataset']+"_body_measurements")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if config_use_generated:
    if not os.path.exists(gen_meas_dir):
        os.makedirs(gen_meas_dir)
#
# 3. Read 3D models - compute measurements - save measurements

def comp_and_save_SCAPE_S_Dibra_body_meas(meas_conf, xyz_points, save_file, verbose):
    save_fid = open(save_file,"w+")
    for meas in meas_conf['MEASUREMENTS']:
        meas_inds = json.loads(measures_conf['MEASUREMENTS'][meas])
        meas_points = xyz_points[meas_inds,:]
        # calculate geodesic distance
        if not meas_inds: # empty -> OVERALL HEIGHT
            height = max(xyz[:,2])-min(xyz[:,2])
            save_fid.write("SCAPE-S-Dibra MaxZMinZ %s %3.2f\n" %(meas,height))
        elif meas_points.shape[0] == 2: # length measurement
            len = np.sqrt(np.sum((meas_points[0][:]-meas_points[1][:])**2))
            save_fid.write("SCAPE-S-Dibra Length %s %3.2f\n" %(meas,len))
        else: # Circumference
            circ = (meas_points[:-1][:]-meas_points[1:][:])**2
            circ = np.sum(circ,axis=1)
            circ = np.sqrt(circ)
            circ = np.sum(circ,axis=0)
            save_fid.write("SCAPE-S-Dibra Circumference %s %3.2f\n" %(meas,circ))

        if verbose:
            ax.plot3D(meas_points[:,0],meas_points[:,1],meas_points[:,2],'red')
            plt.draw()
            plt.pause(0.1)
    save_fid.close()



# Read measurement paths (vertex indices)
measures_conf = configparser.ConfigParser()
if not os.path.exists(config[config_dataset]['MeasurementFile']):
    print(f"The measurement file does not exist: {config[config_dataset]['MeasurementFile']}")
    exit()
else:
    measures_conf.read(config[config_dataset]['MeasurementFile'])

# 3.1 Process the original data file (real scan fits)
if debug:
    ax = plt.axes(projection='3d') # For visualization

line_cnt = 0
with open(data_list_file,'r') as file_in:
    for line in file_in:
        line_cnt += 1
        line = line.strip()
        model_file = Path(config[config_dataset]['DataDir'],line)
        print("\r -> Reading 3D body points %4d (%s)" %(line_cnt,model_file), end="\r")
        mat_data = loadmat(model_file)
        xyz = mat_data['points']
        if debug:
            ax.cla()
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=[0,0,0], s=0.1)
            plt.title(line)
            ax.set_xlim3d(-500,500)
            ax.set_ylim3d(-500,500)
            ax.set_zlim3d(-1000,1000)
            plt.ion()
            plt.show()
            plt.pause(0.1)

        meas_file = os.path.splitext(os.path.basename(line))
        meas_save_file = Path(data_dir, config_dataset + "_BODY_MEAS_" + meas_file[0] + ".txt")
        comp_and_save_SCAPE_S_Dibra_body_meas(measures_conf, xyz, meas_save_file, debug)
print(" ... Done!\n")

# 3.2 Process the generated bodies
if config_use_generated:
    line_cnt = 0
    with open(gen_data_list_file,'r') as file_in:
        for line in file_in:
            line_cnt += 1
            line = line.strip()
            model_file = line # Path(gen_data_dir,line)
            print("\r -> Reading generated body points %5d (%s)" %(line_cnt,model_file), end=" ")
            mat_data = loadmat(model_file)
            # There is something wrong in synth_sample generation and thus this crazy stuff
            xyz = mat_data['synth_sample']
            xyz = xyz['points']
            xyz = xyz[0]
            xyz = xyz[0]
            if debug:
                ax.cla()
                ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=[0,0,0], s=0.1)
                plt.title(line)
                ax.set_xlim3d(-500,500)
                ax.set_ylim3d(-500,500)
                ax.set_zlim3d(-1000,1000)
                plt.ion()
                plt.show()
                plt.pause(0.1)

            # meas_dir = os.path.dirname(line) # same as the mat dir
            meas_file = os.path.splitext(os.path.basename(line))
            meas_save_file = Path(gen_meas_dir, meas_file[0] + "_BODY_MEAS_" + ".txt")
            comp_and_save_SCAPE_S_Dibra_body_meas(measures_conf, xyz, meas_save_file, debug)
    print(" ... Done!\n")
