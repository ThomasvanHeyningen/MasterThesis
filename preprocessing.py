import numpy as np
from params import Params as Params
import os
import random
from random import shuffle
import loader
import visualizer

'''
    Make sure the data is more centered around zero instead of half the data.max
'''


def normalize_data(data):
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)  # normalize
    return data


def create_patch_batch(path, offset, batch_size, get_location=False):
    hyperParams = Params()
    patient_dirs = os.listdir(path)
    patch_list = None
    target_list = None
    empty_dirs = 0
    positive_patches = 0
    negative_patches = 0
    print 'Offset is: ' + str(offset)
    print 'Batch size is:' + str(batch_size)
    for directory in patient_dirs[offset:offset + batch_size]:
        study_dirs = os.listdir(path + '/' + directory)
        for study in study_dirs:
            total_path = path + '/' + directory + '/' + study + '/'
            filename = total_path + 'T2Tra.vtk'
            # Check if file exists, sometimes there is no data.
            if os.path.isfile(filename):
                data = loader.load_data_file(filename)
                tra_annotations = loader.load_annotations(total_path + 'annot.csv',
                                                                      total_path + 'T2TraTransform.csv')
                segmentation = segment_prostate(data)
                annotation_list = create_annotation_list(tra_annotations)
                patches, targets = create_patches(data, annotation_list, hyperParams.patch_size, segmentation)
                visualizer.visualize_annotations(data, annotation_list, directory)
                if patches.size == 0:
                    empty_dirs += 1
                elif patch_list is None:
                    patch_list = patches
                    target_list = targets
                else:
                    patch_list = np.concatenate((patch_list, patches), axis=0)
                    target_list = np.concatenate((target_list, targets), axis=0)

    shuffled_patches = np.empty_like(patch_list)
    shuffled_targets = np.empty_like(target_list)
    if not (patch_list is None):
        index_shuf = range(len(patch_list))
        shuffle(index_shuf)
        for i, old_i in enumerate(index_shuf):
            shuffled_patches[old_i] = patch_list[i]
            shuffled_targets[old_i] = target_list[i]

    return np.array(shuffled_patches), np.array(shuffled_targets), empty_dirs


def mask_value(slice, ellipse):
    x_max, y_max = slice.shape
    mask = np.ones(slice.shape)
    mask *= 255
    for x in range(0, x_max):
        for y in range(0, y_max):
            if ((y - ellipse['x0']) ** 2) / ((ellipse['width'] * 1 / 2) ** 2) + ((x - ellipse['y0']) ** 2) / (
                        (ellipse['heigth'] * 1.0 / 2) ** 2) <= 1:
                mask[x][y] = slice[x][y]
    return mask


def segment_prostate(data):
    shape = data.shape
    slice_count = shape[0]
    segmentation = []
    for i in range(0, slice_count):
        ellipse = {'x0': 0, 'y0': 0, 'width': 0, 'heigth': 0}

        x = (i - slice_count / 2)

        width = -0.5 * x ** 2 + 175
        height = -0.35 * x ** 2 + 140

        ellipse['x0'] = data.shape[1] / 2
        ellipse['y0'] = data.shape[2] / 2 + 15

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
    # name = 0
    for slice in range(0, shape[0]):
        for annotation in annotation_list:
            if -2 < annotation['z'] - slice < 2:
                radius = annotation['radius']
                for column in range(int(annotation['x'] - radius),
                                    int(annotation['x'] + radius - hyperParameters.patch_size)):
                    for row in range(int(annotation['y'] - radius),
                                     int(annotation['y'] + radius - hyperParameters.patch_size)):
                        patch = data[slice, None, column:column + hyperParameters.patch_size,
                                row:row + hyperParameters.patch_size]
                        if targets.count(1) < 350:
                            patches.append(patch)
                            targets.append(int(1))
                            # visualizer.visualize_patches(column, row, 1, data[slice],name)
                            # name += 1
            else:
                for i in range(0, 7):
                    x = random.randint(int(ellipse_data[slice]['x0'] - ellipse_data[slice]['width'] / 2),
                                       int(ellipse_data[slice]['x0'] + ellipse_data[slice]['width'] / 2))
                    y = random.randint(int(ellipse_data[slice]['y0'] - ellipse_data[slice]['heigth'] / 2),
                                       int(ellipse_data[slice]['y0'] + ellipse_data[slice]['heigth'] / 2))
                    patch = data[slice, None, x:x + patch_size, y:y + patch_size]
                    if targets.count(0) < 350:
                        patches.append(patch)
                        targets.append(0)
                        # name += 1
                        # visualizer.visualize_patches(x, y, 0, data[slice],name)

    targets = np.array(targets).astype('int32')
    patches = np.array(patches)
    return patches, targets


def create_test_batch(path, offset, batch_size, get_location=False):
    hyperParams = Params()
    patient_dirs = os.listdir(path)
    patch_list = None
    target_list = None
    location_list = None
    empty_dirs = 0
    positive_patches = 0
    negative_patches = 0
    data = None
    print 'Offset is: ' + str(offset)
    print 'Batch size is:' + str(batch_size)
    for directory in patient_dirs[offset:offset + batch_size]:
        study_dirs = os.listdir(path + '/' + directory)
        for study in study_dirs:
            total_path = path + '/' + directory + '/' + study + '/'
            filename = total_path + 'T2Tra.vtk'
            # Check if file exists, sometimes there is no data.
            if os.path.isfile(filename):
                data = loader.load_data_file(filename)
                tra_annotations = loader.load_annotations(total_path + 'annot.csv',
                                                                      total_path + 'T2TraTransform.csv')
                segmentation = segment_prostate(data)
                annotation_list = create_annotation_list(tra_annotations)
                patches, targets, locations = create_test_patches(data, annotation_list, hyperParams.patch_size, segmentation)
                visualizer.visualize_annotations(data, annotation_list, directory)
                if patches.size == 0:
                    empty_dirs += 1
                elif patch_list is None:
                    patch_list = patches
                    target_list = targets
                    location_list = locations
                else:
                    patch_list = np.concatenate((patch_list, patches), axis=0)
                    target_list = np.concatenate((target_list, targets), axis=0)
                    location_list = np.concatenate((location_list, locations), axis=0)

    shuffled_patches = np.empty_like(patch_list)
    shuffled_targets = np.empty_like(target_list)
    shuffled_locations = np.empty_like(location_list)
    if not (patch_list is None):
        index_shuf = range(len(patch_list))
        shuffle(index_shuf)
        for i, old_i in enumerate(index_shuf):
            shuffled_patches[old_i] = patch_list[i]
            shuffled_targets[old_i] = target_list[i]
            shuffled_locations[old_i] = location_list[i]

    return np.array(shuffled_patches), np.array(shuffled_targets), shuffled_locations, data


def create_test_patches(data, annotation_list, patch_size, ellipse_data):
    hyperParameters = Params()
    shape = data.shape
    patches = []
    targets = []
    locations = []
    slices = []
    # name = 0
    for slice in range(0, shape[0]):
        for annotation in annotation_list:
            if -2 < annotation['z'] - slice < 2:
                radius = annotation['radius']
                for column in range(int(annotation['x'] - radius),
                                    int(annotation['x'] + radius - hyperParameters.patch_size)):
                    for row in range(int(annotation['y'] - radius),
                                     int(annotation['y'] + radius - hyperParameters.patch_size)):
                        patch = data[slice, None, column:column + hyperParameters.patch_size,
                                row:row + hyperParameters.patch_size]
                        if targets.count(1) < 350:
                            patches.append(patch)
                            targets.append(int(1))
                            locations.append([slice, column, row])
                            # visualizer.visualize_patches(column, row, 1, data[slice],name)
                            # name += 1
            else:
                for i in range(0, 7):
                    x = random.randint(int(ellipse_data[slice]['x0'] - ellipse_data[slice]['width'] / 2),
                                       int(ellipse_data[slice]['x0'] + ellipse_data[slice]['width'] / 2))
                    y = random.randint(int(ellipse_data[slice]['y0'] - ellipse_data[slice]['heigth'] / 2),
                                       int(ellipse_data[slice]['y0'] + ellipse_data[slice]['heigth'] / 2))
                    patch = data[slice, None, x:x + patch_size, y:y + patch_size]
                    if targets.count(0) < 350:
                        patches.append(patch)
                        targets.append(0)
                        locations.append([slice, x, y])
                        # name += 1
                        # visualizer.visualize_patches(x, y, 0, data[slice],name)

    targets = np.array(targets).astype('int32')
    patches = np.array(patches)
    locations = np.array(locations)
    return patches, targets, locations


def create_annotation_list(annotations):
    annotation_list = []
    for annotation in annotations:
        circle = {'x': 0, 'y': 0, 'z': 0, 'radius': 20}

        circle['x'] = annotation[0]
        circle['y'] = annotation[1]
        circle['z'] = round(annotation[2])

        annotation_list.append(circle)

    return annotation_list
