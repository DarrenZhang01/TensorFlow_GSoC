# Draft version for TensorFlow stax API.
#
# Adapted based on "https://github.com/google/jax/blob/master/jax/experimental/stax.py"
#
# 2020.05.30

import functools
import itertools
import operator as op

from trax.tf_numpy.numpy import array_ops
from trax.tf_numpy.numpy import math_ops
from trax.tf_numpy.numpy import random

from tensorflow.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                           leaky_relu, selu, normalize)
# TODO: tf.nn.gelu

def Dense():

    pass

def GeneralConv():

    pass

def GeneralConvTranpose():

    pass

def BatchNorm():

    pass

def elementwise():

    pass

def _pooling_layer():

    pass

def _normalize_by_window_size():

    pass

def Flatten():

    pass
Flatten = Flatten()

def Identity():

    pass
Identity = Identity()

def FanOut():

    pass

def FanInSum():

    pass
FanInSum = FanInSum()

def FanInConcat():

    pass

def Dropout():

    pass

def serial():

    pass

def parallel():

    pass

def shape_dependent():

    pass
