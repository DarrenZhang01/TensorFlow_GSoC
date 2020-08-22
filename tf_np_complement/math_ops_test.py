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
Zhibo Zhang, 2020.08.20
"""

import tensorflow.compat.v2 as tf
import numpy as onp
from tensorflow.python.ops import numpy_ops as np
from math_ops import size as np_size
from tensorflow.python.platform import test


class OpsTest(test.TestCase):

  def testSize(self):

    def run_test(arr):
      onp_arr = arr.numpy() if isinstance(arr, tf.Tensor) else arr
      print(onp_arr)
      self.assertEqual(np_size(arr), onp.size(onp_arr))

    run_test(np.array([1]))
    run_test(np.array([1, 2, 3, 4, 5]))
    run_test(np.ones((2, 3, 2)))
    run_test(np.ones((3, 2)))
    run_test(np.zeros((5, 6, 7)))
    run_test(1)
    run_test(onp.ones((3, 2, 1)))
    run_test(tf.constant(5))
    run_test(tf.constant([1, 1, 1]))


if __name__ == "__main__":
  test.main()
