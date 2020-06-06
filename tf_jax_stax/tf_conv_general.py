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
This file contains a general conv operation for TensorFlow.

Zhibo Zhang, 2020.06.06
"""
from tf_lax import ConvDimensionNumbers
from tensorflow import nn
import tensorflow

# Translate dimension numbers from number representations into string
#   representations
#
# For example, in a 1D case, "NWC" means "# batches", "spatial dimensions" and
#   "# Channels (features)", so the according number representations based on order
#   "batch dimension, feature dimension, spatial dimension" is tuple (0, 2, 1);
#
# The reverse translation is simple. Set the default mapping as follows:
#   1D: "NCW" <=> (lhs_spec[0], lhs_spec[1], lhs_spec[2])
#   2D: "NCHW" <=> (lhs_spec[0], lhs_spec[1], lhs_spec[2], lhs_spec[3])
#   3D: "NCDHW" <=> (lhs_spec[0], lhs_spec[1], lhs_spec[2], lhs_spec[3], lhs_spec[4])
# In order to get the translated string version from the number version, sort
# strings and numbers at the same time based on numbers (0 to len(lhs_spec) - 1)
def conv_dim_translator(lhs_spec, rhs_spec, out_spec):
  """ Translate ConvDimensionNumbers into string representations. """



def conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation=None,
                         rhs_dilation=None, dimension_numbers=None
                         feature_group_count=1, batch_group_count=1, precision=None):
  """ A general conv function that integrates normal conv, deconvolution,
  dilated convolution, etc."""
  dim = None
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  if len(lhs_spec) >= 6:
    raise TypeError("Current implmentation does not support 4 or higher"
                    "dimensional convolution, but got: ", len(lhs_spec) - 2)
  dim = len(lhs_spec) - 2
  if dim == 1:

  elif dim == 2:

  elif dim == 3:

















    1
