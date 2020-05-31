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

from tensorflow.keras.initializers import GlorotNormal, RandomNormal, Ones, Zeros

def Dense(out_dim, W_init=None, bias_init=None):
  """ Dense/Fully-connected layer."""
  def init_func(rng_seed, input_shape):
    """ rng_seed is for specifying seed for the random initializers. """
    output_shape = input_shape[:-1] + (out_dim,)
    W_init = W_init or GlorotNormal(rng_seed)
    bias_init = bias_init or RandomNormal(seed=rng_seed)
    W, b = W_init((input_shape[-1], out_dim)), bias_init((out_dim,))
    return output_shape, (W, b)
  def apply_func(params, inputs, **kwargs):
    W, b = params
    return math_ops.dot(inputs, W) + b
  return init_func, apply_func

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
