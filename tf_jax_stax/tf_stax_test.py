#
#
# Adapated based on "https://github.com/google/jax/blob/master/tests/stax_test.py"

""" Tests for TF stax library. """

from absl.testing import absltest
from absl.testing import parameterized

from stax import Dense

from trax.tf_numpy.numpy import array_ops
from trax.tf_numpy.numpy import math_ops
from trax.tf_numpy.numpy import random
from tensorflow import test
import numpy as np

_RNG_SEED = 0

def _TestShapeAgreement(test, init_func, apply_func, input_shape):
  seed = _RNG_SEED
  result_shape, params = init_func(seed, input_shape)
  # TODO: Write a helper function for constructing random inputs.
  inputs = np.ones(input_shape)
  result = apply_func(params, inputs)
  test.assertEqual(result.shape, result_shape)

class TFStaxTest(test.TestCase):

  @parameterized.parameters(
    {'out_dim': 5, 'input_shape': (2, 3)},
    {'out_dim': 10, 'input_shape': (5, 6)}
  )
  def testDenseShape(self, out_dim, input_shape):
    init_func, apply_func = Dense(out_dim)
    _TestShapeAgreement(self, init_func, apply_func, input_shape)
