#####
prog_text = 'This program renders silhouettes from 3D body meshes.'
#####

# Basic packages
import argparse
import configparser
import json
from pathlib import Path
import os
from time import sleep

# Loading MAT files
from scipy.io import loadmat

# Graphics packages
import pyglet
from pyglet.gl import *
from pywavefront import visualization
import pywavefront

# Image processing packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

#
# 1. Process command line arguments (config file)

parser = argparse.ArgumentParser(description=prog_text)
parser.add_argument("-c", "--config", help="set configuration file")
parser.add_argument("-g", "--generated", help="process generated body files (true/false)",default="false")
parser.add_argument("-d", "--debug", help="produce debug information (true/false)",default="false")
parser.add_argument("-s","--imagesize",type=int,help="you can add specific image size", default=224)
parser.add_argument("-a1","--angle1",type=int,help="rotation of first picture",default=45)
parser.add_argument("-a2","--angle2",type=int,help="rotation of second picture",default=135)
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

if args.angle1 == 45:
    angle1 = 45
else:
    angle1 = args.angle1
    
if args.angle2 == 135:
    angle2 = 135
else:
    angle2 = args.angle2
#
# 2. Process config file and set parameters
config = configparser.ConfigParser()
config.read(config_file)

# Dataset to be processed
config_dataset = config['MAIN']['Dataset']

# Check whether Male or Female
if config['MAIN']['Gender'] == 'Male':
    if process_generated:
        data_list_file = config[config_dataset]['MaleListTrainGenerated']
    else:
        data_list_file = config[config_dataset]['MaleList']
    wavefront_template = pywavefront.Wavefront(config[config_dataset]['MaleOBJTemplate'])
else: # otherwise Female
    if process_generated:
        data_list_file = config[config_dataset]['FemaleListTrainGenerated']
    else:
        data_list_file = config[config_dataset]['FemaleList']
    wavefront_template = pywavefront.Wavefront(config[config_dataset]['FemaleOBJTemplate'])

# Dataset to be processed
config_dataset = config['MAIN']['Dataset']

# Dir where to store generated measurements
temp_dir = config['MAIN']['TempDir']
if not os.path.exists(temp_dir):
    print("Making temporary working directory: ", temp_dir)
    os.makedirs(temp_dir)
if process_generated: # file names contain full path
    if config['MAIN']['Gender'] == 'Male':
        OBJ_DIR = config[config_dataset]['ObjDirMaleGenerated']
        SILH_DIR = config[config_dataset]['SilhDirMaleGenerated']
    else:
        OBJ_DIR = config[config_dataset]['ObjDirFemaleGenerated']
        SILH_DIR = config[config_dataset]['SilhDirFemaleGenerated']
else:
    OBJ_DIR = Path(temp_dir, config['MAIN']['Dataset']+"_obj")
    SILH_DIR = Path(temp_dir, config['MAIN']['Dataset']+"_silhouettes")

# create dirs for obj and silhouette
if not os.path.exists(OBJ_DIR):
    os.makedirs(OBJ_DIR)
if not os.path.exists(SILH_DIR):
    os.makedirs(SILH_DIR)

FRONT_DIR = Path(SILH_DIR, "front")
if not os.path.exists(FRONT_DIR):
    os.makedirs(FRONT_DIR)
SIDE_DIR = Path(SILH_DIR, "side")
if not os.path.exists(SIDE_DIR):
    os.makedirs(SIDE_DIR)

#
# 3. Read 3D models and render them as silhouettes

# Read file names
with open(data_list_file) as f:
    data_list = f.readlines()
data_list = [x.strip() for x in data_list]

# Generate the drawing window (of the silhouette size)
if args.imagesize == 224:
    img_width = json.loads(config['MAIN']['SilhouetteWidth'])
    img_height = json.loads(config['MAIN']['SilhouetteHeight'])
else:
    img_width = args.imagesize
    img_height = args.imagesize
       
window = pyglet.window.Window(img_width,img_height)

data_list_ind = -1
wavefront_obj = []
# Event that updates the silhouette
def update_silhouette(dt):
    global data_list_ind, wavefront_obj
    # First time called
    if data_list_ind == -1:
        data_list_ind  = 0
    else:
        data_list_ind += 1

    if data_list_ind == len(data_list):
        exit() # All images processed

    print("\r -> Processing file %5d (%s)" %(data_list_ind,len(data_list)), end=" ")
    if process_generated:
        mat_data = loadmat(data_list[data_list_ind][:])
        # There is something wrong in synth_sample generation and thus this crazy stuff
        xyz = mat_data['synth_sample']
        xyz = xyz['points']
        xyz = xyz[0]
        xyz = xyz[0]
        obj_file = os.path.splitext(os.path.basename(data_list[data_list_ind][:]))
        body_file = os.path.splitext(os.path.basename(data_list[data_list_ind][:]))
        obj_save_file_name = Path(OBJ_DIR, obj_file[0] + "_silhouette" + ".obj")
    else:
        #
        mat_data = loadmat(Path(config[config_dataset]['DataDir'],data_list[data_list_ind][:]))
        xyz = mat_data['points']
        body_file = os.path.splitext(os.path.basename(data_list[data_list_ind][:]))
        obj_save_file_name = Path(OBJ_DIR, config_dataset + "_silhouette_" + body_file[0] + ".obj")

    filepath = config[config_dataset]['MaleOBJTemplate']

    obj_save_fid = open(obj_save_file_name,"w+")
    with open(filepath) as fp:
        vertex_count = 0
        for line in fp:
            if line[0] == 'v':
                #temp_obj_file.write("%s" %line)
                obj_save_fid.write("v %f %f %f\n" %(xyz[vertex_count][0],xyz[vertex_count][1],xyz[vertex_count][2]))
                vertex_count += 1
            else:
                obj_save_fid.write("%s" %line)
    obj_save_fid.close()
    wavefront_obj = pywavefront.Wavefront(obj_save_file_name) # loading object

# The main drawing function that draws the 3D model frontal view
@window.event
def on_draw():
    if wavefront_obj:
        body_file = os.path.splitext(os.path.basename(data_list[data_list_ind][:]))
        if process_generated:
            # silh_dir = os.path.dirname(data_list[data_list_ind][:])
            silh_file = os.path.splitext(os.path.basename(data_list[data_list_ind][:]))
            silh_save_file_name_front = Path(FRONT_DIR, silh_file[0] + "_silhouette" + ".png")
            silh_save_file_name_side = Path(SIDE_DIR, silh_file[0] + "_silhouette" + ".png")
        else:
            silh_save_file_name_front = Path(FRONT_DIR, config_dataset + "_silhouette_" + body_file[0] + ".png")
            silh_save_file_name_side = Path(SIDE_DIR, config_dataset + "_silhouette_" + body_file[0] + ".png")
    #---------------------------------------------------------------------------
    # Front View
    window.clear()
    glLoadIdentity()

    viewport_width, viewport_height = img_width,img_height
    glViewport(0,0, viewport_width, viewport_height)

    # Set the projection transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(viewport_width)/viewport_height, 1., 5000.)
    glMatrixMode(GL_MODELVIEW)

    # Set the modelview transformation
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0,50,+2000,0,0,0,0,1,0)

    glRotatef(-90.0, 1.0, 0.0, 0.0)
    glRotatef(angle1, 0.0, 0.0, 1.0)#angle1

    if wavefront_obj:
        visualization.draw(wavefront_obj)
        buffer_img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        buf_bytes = buffer_img.get_data('I', -buffer_img.width)
        img_array = np.empty([buffer_img.height, buffer_img.width])
        th_array = np.zeros([buffer_img.height, buffer_img.width], dtype=np.uint8)
        for img_h in range(buffer_img.height):
            for img_w in range(buffer_img.width):
                img_array[img_h][img_w] = buf_bytes[img_w+img_h*buffer_img.width]

        th_array[img_array == img_array.max()] = 255
        th_img = Image.fromarray(th_array)
        th_img.save(silh_save_file_name_front)
    #-----------------------------------------------------------------------
    # Side View
    window.clear()
    glLoadIdentity()

    viewport_width, viewport_height = img_width,img_height
    glViewport(0,0, viewport_width, viewport_height)

    # Set the projection transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(viewport_width)/viewport_height, 1., 5000.)
    glMatrixMode(GL_MODELVIEW)

    # Set the modelview transformation
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0,50,+2000,0,0,0,0,1,0)

    glRotatef(-90.0, 1.0, 0.0, 0.0)
    glRotatef(angle2, 0.0, 0.0, 1.0) # Side View, Face to Right

    if wavefront_obj:
        visualization.draw(wavefront_obj)
        buffer_img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        buf_bytes = buffer_img.get_data('I', -buffer_img.width)
        img_array = np.empty([buffer_img.height, buffer_img.width])
        th_array = np.zeros([buffer_img.height, buffer_img.width], dtype=np.uint8)
        for img_h in range(buffer_img.height):
            for img_w in range(buffer_img.width):
                img_array[img_h][img_w] = buf_bytes[img_w+img_h*buffer_img.width]

        th_array[img_array == img_array.max()] = 255
        th_img = Image.fromarray(th_array)
        th_img.save(silh_save_file_name_side)

# Clock event generates redraw every time
#update_silhouette(0)
pyglet.clock.schedule_interval(update_silhouette, json.loads(config['MAIN']['RenderInterval']))
pyglet.app.run()
