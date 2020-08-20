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
Implement TF Numpy operations `pi`, `round`, `sign`, `size`, 'einsum' and linalg series.

Zhibo Zhang, 2020.06.19
"""

import tensorflow.compat.v2 as tf
import numpy as onp
from tensorflow.python.ops import numpy_ops as np


# # @utils.np_doc(np.moveaxis)
# def moveaxis(x, from, to):
#   return np.array(onp.moveaxis(onp.array(x), from, to))
#
#
# # @utils.np_doc(np.sign)
# def sign(x):
#   if isinstance(x, (float, int, onp.ndarray)):
#     return tf.convert_to_tensor(np.sign(x))
#   elif isinstance(x, (tf.Tensor, np.ndarray)):
#     x = np.sign(x.numpy())
#     return tf.convert_to_tensor(x)
#   else:
#     raise TypeError("The inputs must be one of types {int, float, numpy array"
#                     ", TensorFlow Tensor, TensorFlow ndarray} object.")

# @utils.np_doc(np.size)
def size(x):
  if isinstance(x, (int, float)):
    return 1
  elif isinstance(x, (np.ndarray, tf.Tensor)):
    return np.prod(x.shape)
  elif isinstance(x, onp.ndarray):
    return np.prod(x.shape)
  else:
    raise TypeError("The inputs must be one of types {int, float, numpy array"
                    ", TensorFlow Tensor, TensorFlow ndarray} object.")
