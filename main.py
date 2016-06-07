import loader
import networks
import numpy as n
import theano
import theano.tensor as T
import lasagne
import sys
import os
import time
import numpy as np
from params import Params as Params
import loader
import visualizer
import preprocessing
import math

def create_patch_batch(path, offset, batch_size):
    patient_dirs = os.listdir(path)
    patch_list = None
    target_list = None
    for directory in patient_dirs[offset:offset+batch_size]:
        study_dirs = os.listdir(path+'/'+directory)
        for study in study_dirs:
            total_path = path+'/'+directory+'/'+study+'/'
            filename = total_path+'T2Tra.vtk'
            #Check if file exists, sometimes there is no data.
            if os.path.isfile(filename):
                data = loader.load_data_file(filename)
                tra_annotations, tra_origin = loader.load_annotations(total_path+'annot.csv',total_path+'T2TraTransform.csv')
                segmentation = preprocessing.segment_prostate(data, tra_origin)
                annotation_list = preprocessing.create_annotation_list(tra_annotations)
                patches, targets = preprocessing.create_patches(data, annotation_list, hyperParameters.patch_size, segmentation)
                # visualizer.visualize_annotations(data, annotation_list, directory)
                if patch_list is None:
                    patch_list = patches
                    target_list = targets
                elif patches.size != 0:
                    patch_list = np.concatenate((patch_list, patches), axis=0)
                    target_list = np.concatenate((target_list, targets), axis=0)
    return patch_list, target_list


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

hyperParameters = Params()

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
print("Building model and compiling functions...")
network = networks.build_vggnet(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=hyperParameters.learning_rate, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

num_epochs = hyperParameters.epochs
# We iterate over epochs:
for epoch in range(1):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    train_acc = 0
    start_time = time.time()
    path = '/media/resfilsp10/prostate/Archives/T2CNN'
    directory_list = os.listdir(path)
    max_batch_number = int(math.floor(len(directory_list)/hyperParameters.batch_size))

    for batch_number in range(0, 1):
        print 'Loading Data'
        patches, targets = create_patch_batch(path,batch_number*hyperParameters.batch_size,min(hyperParameters.batch_size, len(directory_list)-batch_number*10))
        x_train = patches
        y_train = targets
        x_test = patches
        y_test = targets
        print("Starting training batch " + str(batch_number))

        for batch in iterate_minibatches(x_train, y_train, 1, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            err, acc = (val_fn(inputs, targets))
            train_acc += acc
            train_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.6f}".format(train_acc / train_batches))
    if(train_acc/train_batches == 1):
        break


# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(x_test, y_test, hyperParameters.mini_batch, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))



print 'Visualizing Data'
# visualizer.visualize_segmentation(data, segmentation, patient_id)

# visualizer.visualize_annotations(data, annotation_list, patient_id)
