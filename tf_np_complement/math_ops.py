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
import numpy as np

# ##############################################################
# # Below are two type conversion utils from
# #     <https://github.com/google/trax/blob/master/trax/tf_numpy/numpy_impl/utils.py>
# # TODO(Zhibo Zhang):
# #     remove these two after integrating the TF Numpy operations into trax/TF
#
# def _to_tf_type(dtype):
#   """Converts a native python or numpy type to TF DType.
#   Args:
#     dtype: Could be a python type, a numpy type or a TF DType.
#   Returns:
#     A tensorflow `DType`.
#   """
#   return tf.as_dtype(dtype)
#
#
# def _to_numpy_type(dtype):
#   """Converts a native python or TF DType to numpy type.
#   Args:
#     dtype: Could be a python type, a numpy type or a TF DType.
#   Returns:
#     A NumPy `dtype`.
#   """
#   if isinstance(dtype, tf.DType):
#     return dtype.as_numpy_dtype
#   return np.dtype(dtype)
#
# ##############################################################


@utils.np_doc(np.sign)
def sign(x):
  x = np.sign(x.numpy())
  return tf.convert_to_tensor(x)
