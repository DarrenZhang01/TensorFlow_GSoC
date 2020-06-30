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
Tests for the general dot operation for TensorFlow.

Zhibo Zhang, 2020.06.30
"""
from tensorflow import nn
import tensorflow as tf
from tensorflow.python.platform import test
from tf_dot_general import compose_output_rep
from absl.testing import parameterized

class TFConvGeneralTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
    {"lhs": ['i', 'j'], "rhs": ['j', 'k'], "dims": (((1,), (0,)), ((), ())),
      "result": "ik"},
    {"lhs": ['a', 'i', 'j'], "rhs": ['a', 'j', 'k'], "dims": \
      (((2,), (1,)), ((0,), (0,))), "result": "aik"},
    {"lhs": ['a', 'b', 'i', 'j'], "rhs": ['a', 'b', 'j', 'k'], "dims": \
      (((3,), (2,)), ((0, 1,), (0, 1,))), "result": "abik"},
  )
  def test_compose_output_rep(self, lhs, rhs, dims, result):
    contraction, batch = dims
    lhs_contraction, rhs_contraction = contraction
    lhs_batch, rhs_batch = batch
    output_rep = compose_output_rep(lhs, rhs, lhs_contraction, rhs_contraction,
                                    lhs_batch, rhs_batch)
    self.assertEqual(output_rep, result)



if __name__ == "__main__":
  test.main()
