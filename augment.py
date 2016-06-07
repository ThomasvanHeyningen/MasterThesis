import Numpy as np


def substract_mean_image(train_data):
    shape = np.shape(train_data)
    mean_image = np.mean(train_data[0:shape[0]], axis=0)
    train_data -= mean_image
    return train_data


def substract_mean_pixel_value(train_data):
    shape = np.shape(train_data)
    if True:
        train_data -= np.mean(train_data[0:shape[0]], axis=(2, 3))
    else:
        train_data -= np.mean(train_data[0:shape[0]], axis=(1, 2))
    return train_data


def normalize(train_data):
    train_data /= np.float32(255)
    substract_mean_pixel_value(train_data)
    return train_data


def flip(train_data):
    flipped_data = np.fliplr(train_data)
    return flipped_data


