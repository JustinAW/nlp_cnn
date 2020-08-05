from scipy import signal
import numpy as np

#***********************************************#
#               cnn_funcs.py                    #
#   Authors: Justin Weigle                      #
#   Edited: 04 Aug 2020                         #
#***********************************************#

def _convolve (input_vecs, conv_filter):
    """ Runs a 2d convolution operation
    input_vecs:
        type: numpy.array
        param: shape (sentence length, feature vector length)
    conv_filter:
        type: numpy.array
        param: shape (filter height, filter width(aka feature vector length))
    """
    result = signal.convolve2d(input_vecs, conv_filter, mode = 'valid')
    return result.reshape(result.shape[0],)

#   # navtive python loop method, much slower than scipy convolution
#   filter_size = conv_filter.shape[0]
#   # filter must be flipped for convolution, otherwise is cross-correlation
#   filt = np.flipud(conv_filter) # flip vertically
#   filt = np.fliplr(filt) # flip horizontally
#   result = np.zeros((input_vecs.shape[1]))
#   for window in np.arange(0, input_vecs.shape[0] - filter_size + 1):
#       curr_region = input_vecs[window:window+filter_size]
#       curr_result = np.multiply(curr_region, filt)
#       conv_sum = np.sum(curr_result)
#       result[window] = conv_sum
#   final_result = result[0:input_vecs.shape[0] - filter_size + 1]

#   return final_result


def convolve (input_vecs, W, b, filter_shape, img_shape, use_bias = False):
    """ Runs 2d convolution using the function _convolve
    input_vecs: word2vec / random vectors
        type: numpy.array
        param: multidimensional array of shape img_shape
    W: the filters
        type: numpy.array
        param: multidimensional array of shape filter_shape
    b: the biases
        type: numpy.array
        param: array of shape (filter_shape[0], 1) - bias
    filter_shape: the quantity and shape of the filters
        type: tuple
        param: (num filters, filter height, filter width)
    img_shape: the quantity and shape of the input vectors that
            represent the sentence
        type: tuple
        param: (batch size, sentence length, feature vector length)
    """
    #img_shape is (50, (sent length), 300)
    #filter shape is (100, (3/4/5), 300)
    feature_maps = np.zeros(
            (img_shape[0], filter_shape[0], img_shape[1]-filter_shape[1]+1))
    # feature_maps shape is (50, 100, sent length - filter height + 1)

    for sent in range(img_shape[0]):
        for filt in range(filter_shape[0]):
            feature_maps[sent,filt] = _convolve(input_vecs[sent], W[filt, :])

    if use_bias:
        return (feature_maps + b)
    else:
        return feature_maps


def ReLU(x):
    y = np.maximum(x, 0.0)
    return y


def max_pool (conv_out_actv, poolsize):
    """ Uses max pooling to condense the output of a convolution operation
    conv_out_actv:
        type: numpy.array
        param: shape feature_map.shape (from convolve)
    poolsize:
        type: tuple
        param: (batch size, num filters)
    """

    # get the max value of each filter's convolution
    pool_out = conv_out_actv.max(axis = 2)
    # pool_out should be (50, 100)
    # make sure pool_out is the correct shape
    assert pool_out.shape == poolsize, "pool_out.shape != poolsize"

    # get max index array
    max_idx = conv_out_actv.argmax(axis = 2)

    return [pool_out, max_idx, conv_out_actv.shape[2]]


def _dropout (rng, layer, p):
    """ Use bernoulli random number distribution for dropout probability
    p is probability of dropout from layer
    """
    mask = rng.binomial(n = 1, p = 1 - p, size = layer.shape)
    return layer * mask, mask


def linear_layer (max_pool_out, W, b, activation = None, use_bias = False):
    if use_bias:
        if (activation != None):
            return activation(np.dot(max_pool_out, W) + b)
        else:
            return (np.dot(max_pool_out, W) + b)
    else:
        if (activation != None):
            return activation(np.dot(max_pool_out, W))
        else:
            return np.dot(max_pool_out, W)


def dropout_linear_layer (
        max_pool_out, W, b,
        rng, dropout_p,
        activation = None,
        use_bias = False):
    dropout_vecs = []
    mask = []

    # dropout on the inputs
    dropout_max_pool_out, dropout_mask = _dropout(rng, max_pool_out, dropout_p)
    output = linear_layer(dropout_max_pool_out, W, b, activation, use_bias)

    # store input and mask
    dropout_vecs.append(dropout_max_pool_out)   # drop_lin[0][0] input
    mask.append(dropout_mask)                   # drop_lin[1][0] input_mask
    #store output
    dropout_vecs.append(output)                 # drop_lin[0][1] output

    return [dropout_vecs, mask]


def softmax (z):
    """ Softmax Regularization function with dimension checking
    z:
        type: arraylike
        param: list of items of 1 or 2 dimensions

    General purpose: Will compute row wise regardless of dimensions
    Stability: Max number is subtracted first to avoid excessive exponentiation
    """
    scalez = z - np.max(z)
    expsz = np.exp(scalez)
    if (expsz.ndim > 1):
        return expsz / (np.sum(expsz, axis = 1, keepdims = True) + 1e-20)
    else:
        return expsz / (np.sum(expsz, axis = 0, keepdims = True) + 1e-20)


def xentropy_loss (softmax_out, y, batch_size):
    """ Computes the log loss aka crossentropy loss
    softmax_out:
        type: np.array shape(batches, possible answers)
        param: matrix of the normalized prediction confidences
    y:
        type: np.array shape(batches,)
        param: vector giving the correct label for each example

    The mean is used so that an increase in the batch size would not
    significantly change the cost
    """
    # softmax_out contains the [batch_size x output classes] matrix
    # log(softmax_out) gives the log probabilities for each
    #  example in the minibatch, call this LP
    # y.shape[0] is the size of the minibatch
    # np.arange(y.shape[0]) gives [0,1,...,n-1] vector
    # LP[np.arange(y.shape[0]), y] gives:
    # [LP[0,y[0]], LP[1,y[1]], ... , LP[n-1, y[n-1]]] Call this P
    # where for example if LP[0,y[0]] has a y of 1, it would get the log
    # probability at LP[0,1]
    # Division by the batch size to make batch size not matter
    return (-(np.sum(np.log(softmax_out + 1e-8)[np.arange(y.shape[0]), y]))
            / batch_size)


def pred_argmax(softmax_out):
    return np.argmax(softmax_out, axis = 1)


def errors (y_pred, y):
    assert y.ndim == y_pred.ndim,(
            "in errors: y should have the same shape as y_pred")

    if (str(y.dtype).startswith("int")):
        return np.mean((y_pred != y).astype(int))


##########################
""" Backprop Functions """
##########################
def convolve_back (d_conv, input_vecs, W, b):
    """ Compute the gradients (only for the weights and biases) of convolution
    d_conv:
        type: np.array
        param: d/dx convolution
        (batch_size, num_filters, sent_len - filter_h + 1)
    input_vecs:
        type: np.array
        param: original input to the network
        (batch_size, sent_len, feat_vec_length)
    W:
        type: np.array
        param: filter for the convolution operation
        (num_filters, filter_h, feat_vec_length)
    b:
        type: np.array
        param: bias for the convolution operation
        (num_filters, 1)
    """
    batch_size = input_vecs.shape[0] # scale down to account for batch size

    dW = np.zeros((W.shape))
    for example in range(d_conv.shape[0]):
        for filt in range(d_conv.shape[1]):
            dker = d_conv[example, filt].reshape(d_conv[2].shape[1], 1)
            dW[filt] += signal.convolve2d(input_vecs[example], dker, mode = 'valid')

    dW = (1/batch_size) * dW
    assert dW.shape == W.shape, "dW.shape != W.shape in convolve_back"
    db = (1/batch_size) * np.sum(d_conv, axis = (0, 2))
    db = db.reshape(b.shape[0], 1)
    assert db.shape == b.shape, "db.shape != b.shape in convolve_back"

    return dW, db


def relu_back (d_actv):
    keep = (d_actv > 0).astype(int)
    return np.multiply(d_actv, keep)


def max_pool_back (d_maxp_split, max_idx, origdim2):
    """ Max Pooling backprop is just a 'gradient router'
    d_maxp_split:
        type: np.array
        param: 1 of 3 max_pool_split derivatives
    mask:
        type: np.array
        param: mask to route gradients back to the convolution
            weights only where the max was taken
    """
    # shapes i(batch_size, 1), j(1, n_filters) for indexing
    i, j = np.indices(max_idx.shape, sparse=True)
    # zero array size of original array before max pooling
    d_actv = np.zeros(shape = (max_idx.shape[0], max_idx.shape[1], origdim2))
    # set original max locations to the values in d_maxp_split
    d_actv[i, j, max_idx] = d_maxp_split

    return d_actv #d/dx activation


def softmax_back (softmax_out, y, W, b, maxp_concat):
    """ Softmax backprop for linear layer or hidden layer
    softmax_out:
        type: np.array
        param: output of the softmax layer
        (batch_size, n_output_classes)
    y:
        type: np.array
        param: true labels of input sentences
    W:
        type: np.array
        param: Weights of the linear layer or hidden layer
        (feat_vec_length, n_output_classes)
    b:
        type: np.array
        param: bias of the linear layer or hidden layer
    maxp_concat:
        type: np.array
        param: output of the maxpooling layer concatenated
        (batch_size, feat_vec_length)
    """
    batch_size = y.shape[0] # scale down to account for batch size
    softmax_out[range(batch_size), y] -= 1
    dZ = softmax_out
    dW = (1/batch_size) * maxp_concat.T.dot(dZ)
    assert dW.shape == W.shape, "dW.shape != W.shape in softmax_back"
    db = (1/batch_size) * np.sum(dZ, axis = 0)
    assert db.shape == b.shape, "db.shape != b.shape in softmax_back"
#   dx = (1/batch_size) * dZ.dot(W.T) #d/dx maxp_concat
    dx = dZ.dot(dW.T) #d/dx maxp_concat
    assert dx.shape == maxp_concat.shape, (
            "dx.shape != maxp_concat.shape in softmax_back")
    return dx, dW, db


##########################
"""  Update Optimizer  """
##########################
class SGDUpdatesAdam (object):
    """ Adaptive Moment Estimation Gradient Descent Optimization
    Implementation of Adam optimization algorithm from the paper:
    https://arxiv.org/pdf/1412.6980.pdf

    Here, gradients are calculated prior to calling the update method, rather
    than the the Adam class calling forward/backward propagation
    """
    def __init__ (self,
            theta,
            learning_rate = 0.01,
            beta_1 = 0.9, beta_2 = 0.999,
            epsilon = 1e-8):
        self.theta = theta
        self.lr = learning_rate
        self.b1 = beta_1
        self.b2 = beta_2
        self.e = epsilon

        self.m = 0
        self.v = 0
        self.t = 0


    def update (self, dX):
        self.t += 1

        self.m = (self.b1 * self.m) + ((1 - self.b1) * dX)
        self.v = (self.b2 * self.v) + ((1 - self.b2) * dX**2)

        self.m = self.m / (1 - self.b1**self.t)
        self.v = self.v / (1 - self.b2**self.t)

        self.theta = self.theta - self.lr * self.m / (self.v**.5 + self.e)

#       self.lr = self.lr * (1 - self.b2**self.t)**.5 / (1 - self.b1**self.t)
#       self.theta = self.theta - (self.lr * self.m / (self.v**.5 + self.e))

        return self.theta
