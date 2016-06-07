import numpy as np
import cPickle
import lasagne
import csv
import vtk
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction import image
import re

def VTKtoNumpy(vol):
    exporter = vtk.vtkImageExport()
    exporter.SetInput(vol)
    dims = exporter.GetDataDimensions()
    if (exporter.GetDataScalarType() == 3):
        type = UnsignedInt8
    if (exporter.GetDataScalarType() == 5):
        type = 'Int16'
    a = np.zeros(reduce(np.multiply,dims),type)
    s = a.tostring()
    exporter.SetExportVoidPointer(s)
    exporter.Export()
    a = np.reshape(np.fromstring(s,type),(dims[2],dims[0],dims[1]))
    return a

def load_data_file(filename):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    Image = reader.GetOutput()
    array = VTKtoNumpy(Image)
    array = array.astype(float)
    return array

def read_csv_file(file):
    file.next()
    values = []
    for line in file:
        line = line[1:-4]
        split_line = re.split(" |,", line)
        split_line.append(1)
        values.append([float(i) for i in split_line])
    return np.array(values)

"""
Annotations are stored as the center coordinates and score; x y z, score
"""
def load_annotations(annotation_filename, transformation_filename):
    with open(annotation_filename, 'rb') as annotation_file:
        annotations = read_csv_file(annotation_file)
    with open(transformation_filename,'rb') as transformation_file:
        transformation = np.genfromtxt(transformation_file, delimiter=',')
    origin = np.array([[0],[0],[0],[1]])
    transposed_origin = np.transpose(np.dot(np.linalg.inv(transformation),origin))
    return (np.transpose(np.dot(np.linalg.inv(transformation),np.transpose(annotations))), transposed_origin.flatten())

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_other_data():
    xs = []
    ys = []
    for j in range(2):
      d = unpickle('../Data/cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('../Data/cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:30000], axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:1000,:,:,:]
    Y_train = y[0:1000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[1430:1440:,:,:,:]
    Y_test = y[1430:1440:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)
