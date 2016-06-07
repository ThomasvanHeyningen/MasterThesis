import numpy as np
import copy
import math
from params import Params as Params
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import os
import random
import visualizer

def mask_value(slice, ellipse):
    x_max, y_max = slice.shape
    mask = np.ones(slice.shape)
    mask *= 255
    for x in range(0,x_max):
        for y in range (0,y_max):
            if ((y-ellipse['x0'])**2)/((ellipse['width']*1/2)**2) + ((x-ellipse['y0'])**2)/((ellipse['heigth']*1.0/2)**2) <= 1:
               mask[x][y] = slice[x][y]
    return mask

def segment_prostate(data, origin):
    shape = data.shape
    slice_count = shape[0]
    segmentation = []
    for i in range(0, slice_count):
        ellipse = {'x0': 0, 'y0':0, 'width':0,'heigth':0}

        x = (i - slice_count/2)

        width = -0.5*x**2 + 175
        height = -0.35*x**2 + 140

        ellipse['x0'] = data.shape[1]/2
        ellipse['y0'] = data.shape[2]/2+15

        ellipse['width'] = width
        ellipse['heigth'] = height
        segmentation.append(ellipse)

    return segmentation

def create_patches(data, annotation_list, patch_size, ellipse_data):
    hyperParameters = Params()
    shape = data.shape
    patches = []
    targets = []
    slices = []
    name = 0
    for slice in range(0, shape[0]):
        for annotation in annotation_list:
            radius_squared = (annotation['radius']**2 - ((annotation['z'] - slice)*15)**2)
            if(radius_squared > 0):
                radius = math.sqrt(radius_squared)
                for column in range(int(annotation['x']-radius), int(annotation['x']+radius)):
                    for row in range(int(annotation['y']-radius), int(annotation['y']+radius)):
                        patch = data[slice,None,column:column+patch_size,row:row+patch_size]
                        patches.append(patch)
                        targets.append(int(1))
                        visualizer.visualize_patches(column, row, 1, data[slice],name)
                        name += 1
            else:
                if(slice > 3 < data.shape[0]-3):
                    for i in range(0,5):
                        x = random.randint(int(ellipse_data[slice]['x0']-ellipse_data[slice]['width']/2),int(ellipse_data[slice]['x0']+ellipse_data[slice]['width']/2))
                        y = random.randint(int(ellipse_data[slice]['y0']-ellipse_data[slice]['heigth']/2),int(ellipse_data[slice]['y0']+ellipse_data[slice]['heigth']/2))
                        patch = data[slice,None,x:x+patch_size,y:y+patch_size]
                        patches.append(patch)
                        targets.append(0)
                        # name += 1
                        # visualizer.visualize_patches(x, y, 0, data[slice],name)

    targets = np.array(targets).astype('int32')
    patches = np.array(patches)
    return (patches, targets)

def create_annotation_list(annotations):
    annotation_list = []
    for annotation in annotations:
        circle = {'x': 0, 'y':0, 'z':0, 'radius':15}

        circle['x'] = annotation[0]
        circle['y'] = annotation[1]
        circle['z'] = annotation[2]

        annotation_list.append(circle)

    return annotation_list
