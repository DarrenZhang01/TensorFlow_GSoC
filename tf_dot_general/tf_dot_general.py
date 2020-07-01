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
Construct an equivalent general dot operation as that in JAX -
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>

Although there is an implementation in TF XLA, avoid directly using XLA when
possible.

Zhibo Zhang, 2020.06.30
"""

import tensorflow as tf
import string
import sys


# Given lhs representation, rhs representation, contraction and batch dimensions,
# compose the output representation.
#   e.g., ij, jk, (((1,), (0,)), ((), ())) -> ik
#         aij, ajk, (((2,), (1,)), ((0,), (0,))) -> aik
def compose_output_rep(lhs_rep, rhs_rep, lhs_contraction, rhs_contraction,
                        lhs_batch, rhs_batch):
  output_rep = []
  for dim in lhs_batch:
    output_rep.append(lhs_rep[dim])
  for dim in rhs_batch:
    if rhs_rep[dim] not in output_rep:
      output_rep.append(rhs_rep[dim])

  for i in range(len(lhs_rep)):
    if i not in lhs_batch and i not in lhs_contraction:
      output_rep.append(lhs_rep[i])
  for i in range(len(rhs_rep)):
    if i not in rhs_batch and i not in rhs_contraction:
      output_rep.append(rhs_rep[i])
  return ''.join(output_rep)


# The general dot operation:
#   e.g., non-batched: ij,jk->ik
#         batched: ijk,ikl->ijl
def dot_general(lhs, rhs, dimension_numbers):

  char_list = list(string.ascii_lowercase)[8:]
  lhs_dim, rhs_dim = len(lhs.shape), len(rhs.shape)
  lhs_rep = char_list[:lhs_dim]
  rhs_rep = char_list[lhs_dim:lhs_dim+rhs_dim]
  contraction, batch = dimension_numbers
  lhs_contraction, rhs_contraction = contraction
  lhs_batch, rhs_batch = batch

  for i in range(len(lhs_contraction)):
    rhs_rep[rhs_contraction[i]] = lhs_rep[lhs_contraction[i]]
  for i in range(len(lhs_batch)):
    if i < len(rhs_batch):
      rhs_rep[rhs_batch[i]] = lhs_rep[lhs_batch[i]]

  output_rep = compose_output_rep(lhs_rep, rhs_rep, lhs_contraction,
                                  rhs_contraction, lhs_batch, rhs_batch)
  equation = ''.join(lhs_rep) + ',' + ''.join(rhs_rep) + "->" + output_rep
  return tf.einsum(equation, lhs, rhs)
