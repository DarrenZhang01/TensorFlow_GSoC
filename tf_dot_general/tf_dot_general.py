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
from tensorflow.python.ops import numpy_ops as tf_np
import string

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


# If it is the general non-batched/single-batched matrix multiplication,
# use the highly optimized kernel `tf.tensordot` to handle it.
def non_batched_matmul(lhs, rhs, lhs_contraction, rhs_contraction):
  return tf.tensordot(lhs, rhs, axes=(list(lhs_contraction), list(rhs_contraction)))


# # If the batch dimension of the two input matrices correspond with each other
# # (which includes most of the cases), then move the batch dimension to the very
# # front of the matrices, move the contraction dimensions to the end of the lhs
# # matrices and to the axes that are right after the batch dimension in rhs.
# # i.e.,
# #   lhs: (batch dim) + (other spatial dims) + (contraction dims)
# #   rhs: (batch dim) + (contraction dims) + (other spatial dims)
# #
# # Then combine the contraction dimensions into one (if several) such that the
# # matmul API can easily handle them without ambiguity.
# def batched_matmul(lhs, rhs, lhs_dim, lhs_batch, rhs_batch, lhs_contraction,
#                     rhs_contraction):
#   lhs = tf_np.moveaxis(tf_np.array(lhs), lhs_batch + lhs_contraction,
#                       (0,) + tuple(range(lhs_dim - len(lhs_contraction), lhs_dim)))
#   rhs = tf_np.moveaxis(tf_np.array(rhs), rhs_batch + rhs_contraction,
#                       (0,) + tuple(range(1, 1 + lhs_dim)))
#   lhs = lhs.reshape(lhs.shape[:lhs_dim - len(lhs_contraction)] + (-1,))
#   rhs = rhs.reshape((rhs.shape[0], -1) + rhs.shape[1 + len(lhs_contraction):])
#   return tf.linalg.matmul(lhs, rhs)


# The general dot operation:
#   e.g., non-batched: ij,jk->ik
#         batched: ijk,ikl->ijl
def tf_dot_general(lhs, rhs, dimension_numbers):

  char_list = list(string.ascii_lowercase)[8:]
  lhs_dim, rhs_dim = len(lhs.shape), len(rhs.shape)
  lhs_rep = char_list[:lhs_dim]
  rhs_rep = char_list[lhs_dim:lhs_dim+rhs_dim]
  contraction, batch = dimension_numbers
  lhs_contraction, rhs_contraction = contraction
  lhs_batch, rhs_batch = batch

  if len(lhs_batch) == 0 and len(rhs_batch) == 0:
    return non_batched_matmul(lhs, rhs, lhs_contraction, rhs_contraction)

  cond_a = lhs_dim == rhs_dim == 3
  cond_b = lhs_batch == (0,) and rhs_batch == (0,)
  cond_c = lhs_contraction == (lhs_dim - 1,) and rhs_contraction == (1,)
  if cond_a and cond_b and cond_c:
    return tf.linalg.matmul(lhs, rhs)

  for i in range(len(lhs_contraction)):
    rhs_rep[rhs_contraction[i]] = lhs_rep[lhs_contraction[i]]
  for i in range(len(lhs_batch)):
    if i < len(rhs_batch):
      rhs_rep[rhs_batch[i]] = lhs_rep[lhs_batch[i]]

  output_rep = compose_output_rep(lhs_rep, rhs_rep, lhs_contraction,
                                  rhs_contraction, lhs_batch, rhs_batch)
  equation = ''.join(lhs_rep) + ',' + ''.join(rhs_rep) + "->" + output_rep
  return tf.einsum(equation, lhs, rhs)
