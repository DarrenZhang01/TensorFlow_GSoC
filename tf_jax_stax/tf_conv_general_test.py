# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
Tests for the general conv operation for TensorFlow.

Zhibo Zhang, 2020.06.07
"""
from tensorflow import nn
import tensorflow as tf
from tf_lax import ConvDimensionNumbers
from tensorflow.python.platform import test
from tf_conv_general import *
from absl.testing import parameterized
import itertools
from tensorflow.python.ops import numpy_ops as tfnp
from jax import numpy as jnp
import sys

class TFConvGeneralTest(tf.test.TestCase, parameterized.TestCase):

  # @parameterized.parameters(
  #   {"lhs_spec": (0, 1, 2), "dim": 1, "result": "NCW"},
  #   {"lhs_spec": (0, 2, 1), "dim": 1, "result": "NWC"}
  # )
  # def test_conv_dim_translator_1D(self, lhs_spec, dim, result):
  #   translation = conv_dim_translator(lhs_spec, dim)
  #   self.assertEqual(translation, result)
  #
  #
  # @parameterized.parameters(
  #   {"lhs_spec": (0, 1, 2, 3), "dim": 2, "result": "NCHW"},
  #   {"lhs_spec": (0, 3, 1, 2), "dim": 2, "result": "NHWC"}
  # )
  # def test_conv_dim_translator_2D(self, lhs_spec, dim, result):
  #   translation = conv_dim_translator(lhs_spec, dim)
  #   self.assertEqual(translation, result)
  #
  #
  # @parameterized.parameters(
  #   {"lhs_spec": (0, 1, 2, 3, 4), "dim": 3, "result": "NCDHW"},
  #   {"lhs_spec": (0, 4, 1, 2, 3), "dim": 3, "result": "NDHWC"}
  # )
  # def test_conv_dim_translator_2D(self, lhs_spec, dim, result):
  #   translation = conv_dim_translator(lhs_spec, dim)
  #   self.assertEqual(translation, result)


      # ("_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
      #  "_lhs_dilation={}_rhs_dilation={}_dims={}_dtype={}"
      #  "_padding={}"
      #  "_feature_group_count={}_batch_group_count={}"
      #  "_perms={}".format(lhs_shape, rhs_shape,
      #      strides, padding, lhs_dilation, rhs_dilation, ",".join(dim_nums),
      #      dtype, padding, feature_group_count, batch_group_count, perms),
  @parameterized.named_parameters([
      ("_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       "_lhs_dilation={}_rhs_dilation={}"
       "_feature_group_count={}_batch_group_count={}_dims={}"
       "_perms={}".format(lhs_shape, rhs_shape,
           strides, padding, lhs_dilation, rhs_dilation,
           feature_group_count, batch_group_count, ",".join(dimension_numbers), perms),
           lhs_shape, rhs_shape, strides, padding, lhs_dilation, rhs_dilation,
           feature_group_count, batch_group_count, dimension_numbers, perms)
      for batch_group_count, feature_group_count in [(1, 1)]
      for lhs_shape, rhs_shape in [
          ((b * batch_group_count, i * feature_group_count, 9, w),
           (j * feature_group_count * batch_group_count, i, 4, 5))
          for w in [0, 10]
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for strides in [(1, 1), (2, 1)]
      for padding in ['VALID', 'SAME']
      for lhs_dilation, rhs_dilation in [
        ((1, 1), None), (None, (1, 1)),
        ((1, 2), None), (None, (1, 2)),
        ((1, 4), None), (None, (1, 4))
      ]
      for dimension_numbers, perms in [
        (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
        (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0])),
        (("NCHW", "HWIO", "NHWC"), ([0, 1, 2, 3], [2, 3, 1, 0])),
      ]])
  def testConvGeneralDilated(self, lhs_shape, rhs_shape, strides,
                             padding, lhs_dilation, rhs_dilation,
                             feature_group_count, batch_group_count,
                             dimension_numbers, perms):
    tf.print("dimension_numbers: {}".format(dimension_numbers), output_stream=sys.stdout)
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    lhs_tf = tfnp.transpose(tfnp.ones(lhs_shape), lhs_perm)
    rhs_tf = tfnp.transpose(tfnp.ones(rhs_shape), rhs_perm)

    lhs_jax = jnp.transpose(jnp.ones(lhs_shape), lhs_perm)
    rhs_jax = jnp.transpose(jnp.ones(rhs_shape), rhs_perm)

    tf_conv = conv_general_dilated(lhs_tf, rhs_tf, strides, padding, lhs_dilation,
      rhs_dilation, dimension_numbers, feature_group_count, batch_group_count)

    jax_conv = conv_general_dilated(lhs_jax, rhs_jax, strides, padding, lhs_dilation,
      rhs_dilation, dimension_numbers, feature_group_count, batch_group_count)

    self.assertAllEqual(tf_conv, tfnp.asarray(jax_conv))

    # def args_maker():
    #   return [lax.transpose(rng(lhs_shape, dtype), lhs_perm),
    #           lax.transpose(rng(rhs_shape, dtype), rhs_perm)]
    #
    # def fun(lhs, rhs):
    #   return lax.conv_general_dilated(
    #       lhs, rhs, strides, padding, lhs_dilation, rhs_dilation,
    #       dimension_numbers, feature_group_count=feature_group_count,
    #       batch_group_count=batch_group_count)
    #
    # self._CompileAndCheck(fun, args_maker)


if __name__ == "__main__":
  test.main()
