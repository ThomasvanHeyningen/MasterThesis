import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import os
from params import Params as Params
import math
import random

def visualize_segmentation(data, ellipse_data, patient_id):
    data *= (255.0/data.max())
    for slice in range(0,data.shape[0]):
        plt.clf()
        plt.imshow(data[slice], cmap='Greys_r')
        fig = plt.gcf()
        ellipse = Ellipse((ellipse_data[slice]['x0'],ellipse_data[slice]['y0']),ellipse_data[slice]['width'],ellipse_data[slice]['heigth'],color='R',fill=False)
        ax = plt.gca()
        fig.gca().add_artist(ellipse)
        if not os.path.exists('../output/segmentations/'+patient_id):
            os.makedirs('../output/segmentations/'+patient_id)
        plt.savefig('../output/segmentations/'+patient_id+'/slice_'+str(slice))

def visualize_annotations(data, annotation_list, patient_id):
    hyperParameters = Params()
    data *= (255.0/data.max())
    for slice in range(0,data.shape[0]):
        plt.clf()
        plt.imshow(data[slice], cmap='Greys_r')
        fig = plt.gcf()
        for annotation in annotation_list:
            radius_squared = (annotation['radius']**2 - ((annotation['z'] - slice)*15)**2)
            if(radius_squared > 0):
                radius = math.sqrt(radius_squared)
                circle = plt.Circle((annotation['x'],annotation['y']),radius,color='R',fill=False)
                ax = plt.gca()
                fig.gca().add_artist(circle)
        if not os.path.exists('../output/annotations/'+patient_id):
            os.makedirs('../output/annotations/'+patient_id)
        plt.savefig('../output/annotations/'+patient_id+'/slice_'+str(slice))

def create_segmented_pictures(data, ellipse_data, patient_id):
    data *= (255.0/data.max())
    for slice in range(0,data.shape[0]):
        plt.clf()
        slice_data = data[slice]
        mask = mask_value(slice_data, ellipse_data[slice])
        plt.imshow(mask, cmap='Greys_r')
        fig = plt.gcf()
        plt.axis('off')

        if not os.path.exists('../output/augmented_data/'+patient_id):
            os.makedirs('../output/augmented_data/'+patient_id)
        plt.savefig('../output/augmented_data/'+patient_id+'/slice_'+str(slice),bbox_inches='tight')

def visualize_patches(x, y, target, data, name):
    data *= (255.0/data.max())
    plt.clf()
    plt.imshow(data, cmap='Greys_r')
    fig = plt.gcf()
    if target is 0:
        colour = 'G'
    else:
        colour = 'R'
    #Since the rectangle drawing starts at the bottom left corner, x-radius, y-radius
    square = plt.Rectangle((x-16, y-16), 32, 32,alpha=1,color=colour,fill=False)
    ax = plt.gca()
    fig.gca().add_artist(square)

    if not os.path.exists('../output/patches/'):
        os.makedirs('../output/patches/')
    plt.savefig('../output/patches/'+str(name),bbox_inches='tight')
