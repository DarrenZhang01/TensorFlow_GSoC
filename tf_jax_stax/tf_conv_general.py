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
from tensorflow import nn
import numpy as np
import tensorflow as tf
from jax import lax
from more_itertools import sort_together


# # Translate dimension numbers from number representations into string
# #   representations
# #
# # For example, in a 1D case, "NWC" means "# batches", "spatial dimensions" and
# #   "# Channels (features)", so the according number representations based on order
# #   "batch dimension, feature dimension, spatial dimension" is tuple (0, 2, 1);
# #
# # The reverse translation is simple. Set the default mapping as follows:
# #   1D: "NCW" <=> (lhs_spec[0], lhs_spec[1], lhs_spec[2])
# #   2D: "NCHW" <=> (lhs_spec[0], lhs_spec[1], lhs_spec[2], lhs_spec[3])
# #   3D: "NCDHW" <=> (lhs_spec[0], lhs_spec[1], lhs_spec[2], lhs_spec[3], lhs_spec[4])
# # In order to get the translated string version from the number version, sort
# # strings and numbers at the same time based on numbers (0 to len(lhs_spec) - 1)
# def conv_dim_translator(lhs, lhs_spec, dim):
#   """ Translate ConvDimensionNumbers into string representations. """
#   str_maps = {1: ['N', 'C', 'W'], 2: ['N', 'C', 'H', 'W'],
#               3: ['N', 'C', 'D', 'H', 'W']}
#   str_list = str_maps[dim]
#   sorted_str = sort_together([list(lhs_spec), str_list])[1]
#   spatial_dim_maps = {1: 'W', 2: "HW", 3: "DHW"}
#   if not ((sorted_str[0] == 'N' and sorted_str[1] == 'C') or\
#           (sorted_str[0] == 'N' and sorted_str[-1] == 'C')):
#     # raise TypeError("The batch number should always be at the first dimension"
#     #                 "and the channel/feature number should always be at the"
#     #                 "second dimension or the very last dimension, but got ",
#     #                 sorted_str)
#     # The current data format is not valid, need to swap certain axis of the
#     # inputs to make it valid.
#     lhs = np.moveaxis(lhs, (sorted_str.index('N'), sorted_str.index('C')), (0, dim + 1))
#     return 'N' + spatial_dim_maps[dim] + 'C', lhs
#   output_str = "NC" + spatial_dim_maps[dim] \
#                 if sorted_str[0] == 'N' and sorted_str[1] == 'C' else \
#                 'N' + spatial_dim_maps[dim] + 'C'
#   return output_str, lhs


# For example,
#  in the 3D case, if lhs_dilation = 2, then convert it to [2, 2, 2]
#                  if lhs_dilation = (2, 2, 2), convert it also to [2, 2, 2]
def _conv_general_param_type_converter(window_strides, lhs_dilation, rhs_dilation):
  """ Convert the inputs strides, lhs_dilation, rhs_dilation to the standard
  TF conv inputs."""
  strides = [window_strides] * dim if isinstance(window_strides, int) else \
            list(window_strides)
  if lhs_dilation:
    lhs_dilation = [lhs_dilation] * dim if isinstance(lhs_dilation, int) else \
                    list(lhs_dilation)
  if rhs_dilation:
    rhs_dilation = [rhs_dilation] * dim if isinstance(rhs_dilation, int) else \
                    list(rhs_dilation)
  return (strides, lhs_dilation, rhs_dilation)


# TODO: Support feature_group_count, batch_group_count and precision, and
#       allow lhs_dilation and rhs_dilation to happen at the same time.
def conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation=None,
                         rhs_dilation=None, dimension_numbers=None,
                         feature_group_count=1, batch_group_count=1, precision=None):
  """ A general conv function that integrates normal conv, deconvolution,
  dilated convolution, etc."""
  # raise TypeError("lhs shape: {}, rhs shape: {}".format(lhs.shape, rhs.shape))
  dim = None
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  if lhs_spec != out_spec:
    raise TypeError("Current implementation requires the `data_format` of the"
                    "inputs and outputs to be the same.")
  if len(lhs_spec) >= 6:
    raise TypeError("Current implmentation does not support 4 or higher"
                    "dimensional convolution, but got: ", len(lhs_spec) - 2)
  dim = len(lhs_spec) - 2
  if lhs_dilation and rhs_dilation:
    if lhs_dilation == (1,) * dim and rhs_dilation == (1,) * dim:
      lhs_dilation, rhs_dilation = None, None
    else:
      raise TypeError("Current implementation does not support that deconvolution"
                    "and dilation to be performed at the same time, but got"
                    " lhs_dilation: {}, rhs_dilation: {}".format(lhs_dilation,
                    rhs_dilation))
  print("the dim is: {}".format(dim))
  if padding not in ["SAME", "VALID"]:
    raise TypeError("Current implementation requires the padding parameter"
                    "to be either 'VALID' or 'SAME', but got: ", padding)
  # Convert params from int/Sequence[int] to list of ints.
  strides, lhs_dilation, rhs_dilation = _conv_general_param_type_converter(
    window_strides, lhs_dilation, rhs_dilation
  )
  # Preprocess the shapes
  dim_maps = {}
  if isinstance(lhs_spec, str):
    dim_maps['I'] = list(rhs_spec).index('I')
    dim_maps['O'] = list(rhs_spec).index('O')
    dim_maps['N'] = list(lhs_spec).index('N')
    dim_maps['C'] = list(lhs_spec).index('C')
  else:
    dim_maps['I'] = rhs_spec[1]
    dim_maps['O'] = rhs_spec[0]
    dim_maps['N'] = lhs_spec[0]
    dim_maps['C'] = lhs_spec[1]
  # data_format, lhs = conv_dim_translator(lhs, lhs_spec, dim)
  lhs = np.moveaxis(lhs, (dim_maps['N'], dim_maps['C']), (0, dim + 1))
  # Adjust the filters, put the dimension 'I' and 'O' at last.
  rhs = np.moveaxis(rhs, (dim_maps['O'], dim_maps['I']), (dim + 1, dim))
  spatial_dim_maps = {1: 'W', 2: "HW", 3: "DHW"}
  data_format = 'N' + spatial_dim_maps[dim] + 'C'
  tf_nn_APIs = {1: [nn.conv1d, nn.conv1d_transpose],
                2: [nn.conv2d, nn.conv2d_transpose],
                3: [nn.conv3d, nn.conv3d_transpose]}

  output = None
  if rhs_dilation or (lhs_dilation is None and rhs_dilation is None):
    output = tf_nn_APIs[dim][0](lhs, rhs, strides, padding, data_format, rhs_dilation)
  else:
    output = tf_nn_APIs[dim][1](lhs, rhs, strides, padding, data_format, lhs_dilation)
  output = np.moveaxis(output, (0, dim + 1), (dim_maps['N'], dim_maps['C']))
  return output
