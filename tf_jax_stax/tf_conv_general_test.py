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

class TFConvGeneralTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
    {"lhs_spec": (0, 1, 2), "dim": 1, "result": "NCW"},
    {"lhs_spec": (0, 2, 1), "dim": 1, "result": "NWC"}
  )
  def test_conv_dim_translator_1D(self, lhs_spec, dim, result):
    translation = conv_dim_translator(lhs_spec, dim)
    self.assertEqual(translation, result)


  @parameterized.parameters(
    {"lhs_spec": (0, 1, 2, 3), "dim": 2, "result": "NCHW"},
    {"lhs_spec": (0, 3, 1, 2), "dim": 2, "result": "NHWC"}
  )
  def test_conv_dim_translator_2D(self, lhs_spec, dim, result):
    translation = conv_dim_translator(lhs_spec, dim)
    self.assertEqual(translation, result)


  @parameterized.parameters(
    {"lhs_spec": (0, 1, 2, 3, 4), "dim": 3, "result": "NCDHW"},
    {"lhs_spec": (0, 4, 1, 2, 3), "dim": 3, "result": "NDHWC"}
  )
  def test_conv_dim_translator_2D(self, lhs_spec, dim, result):
    translation = conv_dim_translator(lhs_spec, dim)
    self.assertEqual(translation, result)




if __name__ == "__main__":
  test.main()
