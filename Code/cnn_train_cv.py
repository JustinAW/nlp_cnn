import pickle
import numpy as np
from collections import defaultdict
import sys
import re
import copy
import threading

from cnn_funcs import *

#***********************************************#
#               cnn_train_cv.py                 #
#   Authors: Justin Weigle                      #
#   Edited: 04 Aug 2020                         #
#***********************************************#

def ReLU(x):
    y = np.maximum(x, 0.0)
    return y


def initialize_parameters(rng, filter_shapes, ll_dims):
    parameters = {}

    # Convolution filters and biases
    # first size
    parameters["W_conv1"] = np.asarray( rng.uniform(
        low=-0.01, high=0.01, size = filter_shapes[0]), dtype = float)
    parameters["b_conv1"] = 0.15 * np.ones((filter_shapes[0][0],1))
    # second size
    parameters["W_conv2"] = np.asarray(rng.uniform(
        low=-0.01, high=0.01, size = filter_shapes[1]), dtype = float)
    parameters["b_conv2"] = 0.15 * np.ones((filter_shapes[1][0],1))
    # third size
    parameters["W_conv3"] = np.asarray(rng.uniform(
        low=-0.01, high=0.01, size = filter_shapes[2]), dtype = float)
    parameters["b_conv3"] = 0.15 * np.ones((filter_shapes[2][0],1))

    # Linear layer weights and biases
    parameters["W_lin"] = 0.1 * rng.normal(size = (ll_dims[0], ll_dims[1]))
    parameters["b_lin"] = 0.15 * np.ones((ll_dims[1],), dtype = float)

    return parameters


class fwdPropThreader (threading.Thread):
    def __init__ (self, thread_id, X, parameters, pool_sizes, cache, use_bias):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.X = X
        self.parameters = parameters
        self.pool_sizes = pool_sizes
        self.cache = cache
        self.use_bias = use_bias

    def run(self):
        print("running thread" + str(self.thread_id))
        # run convolution
        self.cache["conv" + str(self.thread_id)] = convolve(
                self.X, self.parameters["W_conv" + str(self.thread_id)],
                self.parameters["b_conv" + str(self.thread_id)],
                self.parameters["W_conv" + str(self.thread_id)].shape,
                self.X.shape, self.use_bias)
        # rectified linear units (activation)
        self.cache["a_conv" + str(self.thread_id)] = ReLU(
                self.cache["conv" + str(self.thread_id)])
        # max pooling
        self.cache["max_pool" + str(self.thread_id)] = max_pool(
                self.cache["a_conv" + str(self.thread_id)],
                self.pool_sizes[self.thread_id-1])
        print("exiting thread" + str(self.thread_id))


def forward_prop (X, parameters, rng, dropout_rate, pool_sizes):
    """ Performs forward propagation through a NLP CNN
    X:
        type: np.array shape(batch_size, img_h, img_w)
        param: the initial input feature vectors to the network
    parameters:
        type: dictionary
        param: contains the weights and biases of the layers of the network
    rng:
        type: np.random.RandomState()
        param: random number generator
    dropout_rate:
        type: int
        param: dropout rate for the hidden layer with dropout
    pool_sizes:
        type: list of tuples
        param: sizes to pool to when max pooling

    returns: dictionary "cache" of the intermediate outputs
    """
        # dimension tracking
    # conv_out = shape(batch_size, n_filters, sent_length - filter_h + 1)
    # max_pool_out = shape(batch_size, n_filters)
    # max_pool_concat = shape(batch_size, n_filters*3)
    # hl_out = shape(batch_size, n_output_classes)
    # lin_out = shape(batch_size, n_output_classes)
    # softmax_out = shape(batch_size, n_output_classes)
    # y_pred = shape(batch_size,)

    # store outputs in a dictionary for use on gradient calculations
    cache = {}

    # create a thread for each filter size to speed up convolution
    thread1 = fwdPropThreader(1, X, parameters, pool_sizes, cache, False)
    thread2 = fwdPropThreader(2, X, parameters, pool_sizes, cache, False)
    thread3 = fwdPropThreader(3, X, parameters, pool_sizes, cache, False)

    # run the 3 threads
    thread1.start()
    thread2.start()
    thread3.start()

    # wait for all the threads to finish
    thread1.join()
    thread2.join()
    thread3.join()

    # concatenate outputs of each max pooling into (batch_size, n_filters * 3)
    hl_inputs =  []
    hl_inputs.append(cache["max_pool1"][0])
    hl_inputs.append(cache["max_pool2"][0])
    hl_inputs.append(cache["max_pool3"][0])
    hl_input = np.concatenate(hl_inputs, 1)
    cache["max_pool_concat"] = hl_input

    # dropout and regular linear layer
    cache["drop_lin"] = dropout_linear_layer(cache["max_pool_concat"],
            parameters["W_lin"], parameters["b_lin"],
            rng, dropout_p = dropout_rate,
            activation = None, use_bias = False)
    cache["lin"] = linear_layer(cache["max_pool_concat"],
            parameters["W_lin"] * (1 - dropout_rate), parameters["b_lin"],
            activation = None, use_bias = False)

    # dropout and regular softmax
    cache["drop_softmax_out"] = softmax(cache["drop_lin"][0][1])
    cache["softmax_out"] = softmax(cache["lin"])

    # dropout and regular argmax prediction
    cache["drop_y_pred"] = pred_argmax(cache["drop_softmax_out"])
    cache["y_pred"] = pred_argmax(cache["softmax_out"])

    return cache


def backward_prop (input_vecs, y, parameters, cache):
    grads = {}

    # softmax backprop
    dX, grads["dW_lin"], grads["db_lin"] = softmax_back(
            cache["drop_softmax_out"], y,
            parameters["W_lin"], parameters["b_lin"],
            cache["max_pool_concat"])

    # split into the 3 different filter sizes
    dX_split = np.split(dX, 3, axis = 1) # becomes a list of length 3

    # max pool and convolve back 1
    dX = max_pool_back(dX_split[0],
            cache["max_pool1"][1], cache["max_pool1"][2])
#   dX = relu_back(dX)
    grads["dW_conv1"], grads["db_conv1"] = convolve_back(
            dX, input_vecs, parameters["W_conv1"], parameters["b_conv1"])

    # max pool and convolve back 2
    dX = max_pool_back(dX_split[1],
            cache["max_pool2"][1], cache["max_pool2"][2])
#   dX = relu_back(dX)
    grads["dW_conv2"], grads["db_conv2"] = convolve_back(
            dX, input_vecs, parameters["W_conv2"], parameters["b_conv2"])

    # max pool and convolve back 3
    dX = max_pool_back(dX_split[2],
            cache["max_pool3"][1], cache["max_pool3"][2])
#   dX = relu_back(dX)
    grads["dW_conv3"], grads["db_conv3"] = convolve_back(
            dX, input_vecs, parameters["W_conv3"], parameters["b_conv3"])

    return grads


def train_cnn(index_maps,
                W_vecs,
                img_w = 300,
                filter_hs = [3, 4, 5],
                activation = "relu",
                hidden_units = [100, 2],
                dropout_rate = 0.5,
                shuffle = True,
                n_epochs = 25,
                batch_size = 50,
                non_static = False):
    """
    Train a convolutional neural network
    """
    rng = np.random.RandomState(4432)
    img_h = len(index_maps[0][0])-1
    filter_w = img_w
    n_feat_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((n_feat_maps, filter_h, filter_w))
        pool_sizes.append((batch_size, hidden_units[0]))

    # Initialize Parameters and Adam update classes for them
    hidden_units[0] = n_feat_maps * len(filter_hs)
    parameters = initialize_parameters(
            rng, filter_shapes, hidden_units)
    updates = {}
    for param in parameters.keys():
        updates[param] = SGDUpdatesAdam(parameters[param], 0.01)

    np.random.seed(4432)

    if index_maps[0].shape[0] % batch_size > 0:
        # if the number of sentences is not a multiple of batch_size, randomly
        # selected data from the training set is reused to make a multiple of
        # batch_size
        amt_extra_data = batch_size - (index_maps[0].shape[0] % batch_size)
        train_set = np.random.permutation(index_maps[0])
        extra_data = train_set[:amt_extra_data]
        new_data = np.append(index_maps[0], extra_data, axis = 0)
    else:
        new_data = index_maps[0]
    # shuffle the data and determine number of batches
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0] / batch_size
    n_train_batches = int(np.round(n_batches * 0.9))
    n_val_batches = n_batches - n_train_batches

    # assign data to train and val sets
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]
    train_set_x = train_set[:,:img_h]
    train_set_y = train_set[:,-1]
    val_set_x = val_set[:,:img_h]
    val_set_y = val_set[:,-1]

    # assign test set data
    test_set_x = index_maps[1][:,:img_h]
    test_set_y = np.asarray(index_maps[1][:,-1], "int32")

    for epoch in range(n_epochs):
        print("-------------------------")
        print("| epoch: ", epoch+1, " / ", n_epochs, "\t|")
        print("-------------------------")
        curr_batch = 0
        #for index in range(n_train_batches):
        for index in np.random.permutation(range(n_train_batches)):
            print("-------------------------")
            print("| batch: ", curr_batch+1, " / ", n_train_batches, "\t|")
            print("-------------------------")
            curr_batch += 1
            cache = forward_prop(W_vecs[train_set_x[index*batch_size:(index+1)*batch_size]],
                    parameters, rng, dropout_rate, pool_sizes)

            # TEMP THING
#           cache = forward_prop(W_vecs[train_set_x[0:batch_size]],
#                   parameters, rng, dropout_rate, pool_sizes)

            grads = backward_prop(
                    W_vecs[train_set_x[index*batch_size:(index+1)*batch_size]],
                    train_set_y[index*batch_size:(index+1)*batch_size],
                    copy.deepcopy(parameters), copy.deepcopy(cache))

            # TEMP THING
#           grads = backward_prop(
#                   W_vecs[train_set_x[0:batch_size]],
#                   train_set_y[0:batch_size],
#                   copy.deepcopy(parameters), copy.deepcopy(cache))

            for param in parameters.keys():
                parameters[param] = updates[param].update(grads['d' + param])
#               parameters[param] -= (0.1 * grads['d' + param])

            print("softmax_out:\n", cache["softmax_out"])
            print("y: \n", train_set_y[index * batch_size:(index+1) * batch_size])
            print("y_pred: \n", cache["y_pred"])

            print("loss: ", xentropy_loss(cache["softmax_out"],
                train_set_y[index*batch_size:(index+1)*batch_size], batch_size))
            print("errors: ", errors(cache["y_pred"], train_set_y[index*batch_size:(index+1)*batch_size]))

            # TEMP THING
#           print("softmax_out:\n", cache["softmax_out"])
#           print("y: \n", train_set_y[0:batch_size])
#           print("y_pred: \n", cache["y_pred"])

#           print("loss: ", xentropy_loss(cache["softmax_out"],
#               train_set_y[0:batch_size], batch_size))
#           print("errors: ", errors(cache["y_pred"], train_set_y[0:batch_size]))

#           print("drop_loss: ", xentropy_loss(cache["drop_softmax_out"], 
#               train_set_y[index*batch_size:(index+1)*batch_size]))
#           print("drop_errors: ", errors(cache["drop_y_pred"], train_set_y[index*batch_size:(index+1)*batch_size]))


def get_index(sent, word_map, max_l, filter_h):
    """
    Turns sentences into zero padded list of indices
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_map:
            x.append(word_map[word])
    while len(x) < max_l + 2*pad:
        x.append(0)
    return x


def index_data_cv(data, word_map, cv, max_l, filter_h=5):
    """
    Turn sentences into 2-d matrix of word indices, and split
    based on the "split" value
    """
    train, test = [], []
    for line in data:
        sent = get_index(line["text"], word_map, max_l, filter_h)
        sent.append(line["y"])
        if line["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]


if __name__=="__main__":
    print("Loading data...")
    data = pickle.load(open("mr.p", "rb"))
    revs, W, W2, word_map, vocab = data[0], data[1], data[2], data[3], data[4]
    print("Data loaded!")

    vec_opt = input("Use word2vec vectors(1) or random vectors(2):\n")
    while(vec_opt != '1' and vec_opt != '2'):
        print("Must choose 1 or 2!")
        vec_opt = input("Use word2vec vectors(1) or random vectors(2):\n")
    if(vec_opt == '1'):
        print("Using word2vec vectors")
        W_vecs = W
    elif(vec_opt == '2'):
        print("Using random vectors")
        W_vecs = W2
    results = []
#    for i in range(0, 10):
    for i in range(0, 1):
        index_maps = index_data_cv(revs, word_map, i, max_l=56, filter_h=5)
    #   print("CV: ", i+1)
        train_cnn(index_maps,
                    W_vecs,
                    img_w = 300,
                    filter_hs = [3,4,5],
                    activation = "relu",
                    hidden_units = [100, 2],
                    dropout_rate = 0.5,
                    shuffle = True,
                    n_epochs = 25,
                    batch_size = 50)
