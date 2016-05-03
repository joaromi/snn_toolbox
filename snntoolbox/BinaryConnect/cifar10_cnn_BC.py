# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.


"""
This python script trains a CNN on CIFAR-10 with the stochastic version of
BinaryConnect. It should run for about 20 hours on a Titan Black GPU.
The final test error should be around 8.27%.
"""

from __future__ import print_function

import theano
import lasagne
from snntoolbox.BinaryConnect import binary_connect

theano.sandbox.cuda.use('gpu')


def build_network():

    import theano.tensor as T
    from snntoolbox.BinaryConnect import batch_norm
    from collections import OrderedDict

    # alpha is the exponential moving average factor
    alpha = .1
    epsilon = 1e-4

    # BinaryConnect
    binary = True
    stochastic = True
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    # W_LR_scale = 1.
    W_LR_scale = "Glorot"

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                    input_var=input_var)

    # 128C3-128C3-P2
    cnn = binary_connect.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    cnn = binary_connect.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 256C3-256C3-P2
    cnn = binary_connect.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    cnn = binary_connect.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 512C3-512C3-P2
    cnn = binary_connect.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    cnn = binary_connect.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 1024FP-1024FP-10FP
    cnn = binary_connect.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    cnn = binary_connect.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.rectify)

    cnn = binary_connect.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)

    cnn = batch_norm.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.identity)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    if binary:

        # W updates
        W = lasagne.layers.get_all_params(cnn, binary=True)
        W_grads = binary_connect.compute_grads(loss, cnn)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W,
                                       learning_rate=LR)
        updates = binary_connect.clipping_scaling(updates, cnn)

        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True,
                                               binary=False)
        updates = OrderedDict(list(updates.items()) + list(
            lasagne.updates.adam(loss_or_grads=loss, params=params,
                                 learning_rate=LR).items()))

    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params,
                                       learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1),
                            T.argmax(target, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch
    # (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input_var, target, LR], loss, updates=updates,
                               on_unused_input='ignore')

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target], [test_loss, test_err])

    return cnn, train_fn, val_fn

if __name__ == "__main__":

    import numpy as np
    from pylearn2.datasets.zca_dataset import ZCA_Dataset
    from pylearn2.utils import serial
    from six.moves import cPickle

    print('Building the CNN...')

    cnn, train_fn, val_fn = build_network()

    print("Loading CIFAR-10 dataset...")

    path = '${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/'
    preprocessor = serial.load(path+'preprocessor.pkl')
    train_set = ZCA_Dataset(preprocessed_dataset=serial.load(path+'train.pkl'),
                            preprocessor=preprocessor, start=0,
                            stop=45000)
    valid_set = ZCA_Dataset(preprocessed_dataset=serial.load(path+'train.pkl'),
                            preprocessor=preprocessor, start=45000, stop=50000)
    test_set = ZCA_Dataset(preprocessed_dataset=serial.load(path+'test.pkl'),
                           preprocessor=preprocessor)

    # bc01 format
    train_set.X = train_set.X.reshape(-1, 3, 32, 32)
    valid_set.X = valid_set.X.reshape(-1, 3, 32, 32)
    test_set.X = test_set.X.reshape(-1, 3, 32, 32)

    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)

    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])

    # for hinge loss
    train_set.y = 2 * train_set.y - 1.
    valid_set.y = 2 * valid_set.y - 1.
    test_set.y = 2 * test_set.y - 1.

    print('Training...')

    # Training parameters
    num_epochs = 1
    batch_size = 50
    # Decaying LR
    LR_start = 0.003
    LR_fin = 0.000002
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)

    binary_connect.train(train_fn, val_fn, batch_size, LR_start, LR_decay,
                         num_epochs, train_set.X, train_set.y, valid_set.X,
                         valid_set.y, test_set.X, test_set.y)

    W = lasagne.layers.get_all_layers(cnn)[1].W.get_value()

    import matplotlib.pyplot as plt
    plt.hist(W.flatten())
    plt.title("Weight distribution of first hidden convolution layer")

    # Dump the network weights to a file
    filename = '../data/cifar10/cnn/weightsBC'
    params = lasagne.layers.get_all_param_values(cnn)
    cPickle.dump(params, open(filename, 'wb'))
