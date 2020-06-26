# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reconstruct the JAX test utils
  {_default_tolerance, device_under_test, JaxTestCase, cases_from_list}
such that they are under TF support.

Adapted based on <https://github.com/google/jax/blob/master/jax/test_util.py>,

Zhibo Zhang, 2020.06.26
"""
import numpy.random as npr
import zlib
import re
import tensorflow as tf
from tensorflow.nest import map_structure
from trax.tf_numpy.extensions import jit


# TODO(Zhibo Zhang): Find a way to replace the functionality
#   `xla.xla_primitive_callable.cached_info().misses`

def _dtype(x):
  return (getattr(x, 'dtype', None) or
          np.dtype(dtypes.python_scalar_dtypes.get(type(x), None)) or
          np.asarray(x).dtype)

def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True

def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {np.dtype(key): value for key, value in tol.items()}
  dtype = dtypes.canonicalize_dtype(np.dtype(dtype))
  return tol.get(dtype, default_tolerance()[dtype])

def _assert_numpy_allclose(a, b, atol=None, rtol=None):
  a = a.astype(np.float32) if a.dtype == dtypes.bfloat16 else a
  b = b.astype(np.float32) if b.dtype == dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  np.testing.assert_allclose(a, b, **kw)
