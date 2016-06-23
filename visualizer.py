import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from params import Params as Params


def visualize_segmentation(data, ellipse_data, patient_id):
    data *= (255.0 / data.max())
    for slice in range(0, data.shape[0]):
        plt.clf()
        plt.imshow(data[slice], cmap='Greys_r')
        fig = plt.gcf()
        ellipse = Ellipse((ellipse_data[slice]['x0'], ellipse_data[slice]['y0']), ellipse_data[slice]['width'],
                          ellipse_data[slice]['heigth'], color='R', fill=False)
        ax = plt.gca()
        fig.gca().add_artist(ellipse)
        if not os.path.exists('../output/segmentations/' + patient_id):
            os.makedirs('../output/segmentations/' + patient_id)
        plt.savefig('../output/segmentations/' + patient_id + '/slice_' + str(slice))


def visualize_annotations(data, annotation_list, patient_id):
    hyperParameters = Params()
    data *= (255.0 / data.max())
    for slice in range(0, data.shape[0]):
        plt.clf()
        plt.imshow(data[slice], cmap='Greys_r')
        fig = plt.gcf()
        for annotation in annotation_list:
            if (-2 < annotation['z'] - slice < 2):
                circle = plt.Circle((annotation['x'], annotation['y']), annotation['radius'], color='R', fill=False)
                ax = plt.gca()
                fig.gca().add_artist(circle)
        if not os.path.exists('../output/annotations/' + patient_id):
            os.makedirs('../output/annotations/' + patient_id)
        plt.savefig('../output/annotations/' + patient_id + '/slice_' + str(slice))


def create_segmented_pictures(data, ellipse_data, patient_id):
    data *= (255.0 / data.max())
    for slice in range(0, data.shape[0]):
        plt.clf()
        slice_data = data[slice]
        mask = mask_value(slice_data, ellipse_data[slice])
        plt.imshow(mask, cmap='Greys_r')
        fig = plt.gcf()
        plt.axis('off')

        if not os.path.exists('../output/augmented_data/' + patient_id):
            os.makedirs('../output/augmented_data/' + patient_id)
        plt.savefig('../output/augmented_data/' + patient_id + '/slice_' + str(slice), bbox_inches='tight')


def visualize_patches(x, y, target, data, name):
    data *= (255.0 / data.max())
    plt.clf()
    plt.imshow(data, cmap='Greys_r')
    fig = plt.gcf()
    if target is 0:
        colour = 'G'
    else:
        colour = 'R'
    # Since the rectangle drawing starts at the bottom left corner, x-radius, y-radius
    square = plt.Rectangle((x, y), 32, 32, alpha=1, color=colour, fill=False)
    ax = plt.gca()
    fig.gca().add_artist(square)

    if not os.path.exists('../output/patches/'):
        os.makedirs('../output/patches/')
    plt.savefig('../output/patches/' + str(name), bbox_inches='tight')


'''
    Function that creates a plot to track Loss and Accuracy over epochs.
'''


def visualize_results(results, value_name, directory_name):
    train_results = results[0]
    validation_results = results[1]
    train_results = np.array(train_results)
    validation_results = np.array(validation_results)

    plt.clf()
    plt.plot(train_results[:, 0], train_results[:, 1], label='Training')
    plt.plot(validation_results[:, 0], validation_results[:, 1], label='Validation')
    plt.legend()
    path = directory_name + value_name
    plt.savefig(path)


def visualize_error(location, target, batch_id, output_path, data, patch):
    visual_data = data * (255.0 / data.max())
    if target == 0:
        colour = 'R'
    else:
        colour = 'G'
    plt.clf()
    slice = location[0]
    print slice
    slice_data = visual_data[slice]
    print slice_data.shape
    plt.imshow(visual_data[location[0]], cmap='Greys_r')
    square = plt.Rectangle((location[1], location[2]), 32, 32, alpha=1, color=colour, fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(square)

    path = output_path + 'batch_id' + str(batch_id) + '_slice' + str(location[0]) + '_patch_' + str(patch)
    plt.savefig(path)
