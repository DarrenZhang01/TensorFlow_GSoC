# Copyright 2019 Google LLC
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

"""General-purpose internal utilities."""

from collections import namedtuple
import functools
import inspect
import operator
import types
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Sized, Tuple, Union
from .typing import Axes, PyTree
from . import dataclasses
# from jax import lax
from jax.lib import xla_bridge
from jax.tree_util import tree_all, tree_map
from .kernel import Kernel
import numpy as onp

import sys
from tf_dot_general import tf_dot_general
from tensorflow.python.ops import numpy_ops as np
import tensorflow as tf


def canonicalize_get(get):
  if get is None:
    return True, get

  if not get:
    # NOTE(schsam): It seems slightly nicer to not support the empty-tuple
    # case. Happy to add support later, if there's a use-case.
    raise ValueError('"get" must be non-empty.')

  get_is_not_tuple = isinstance(get, str)
  if get_is_not_tuple:
    get = (get,)

  get = tuple(s.lower() for s in get)
  if len(set(get)) < len(get):
    raise ValueError('All entries in "get" must be unique. Got {}'.format(get))
  return get_is_not_tuple, get


_KERNEL_NAMED_TUPLE_CACHE: Dict[Any, Any] = {}


def named_tuple_factory(name, get):
  key = (name, get)
  if key in _KERNEL_NAMED_TUPLE_CACHE:
    return _KERNEL_NAMED_TUPLE_CACHE[key]
  else:
    _KERNEL_NAMED_TUPLE_CACHE[key] = namedtuple(name, get)
    return named_tuple_factory(name, get)


def _output_to_dict(output):
  if isinstance(output, dict):
    return output

  if isinstance(output, Kernel):
    # Avoid directly converting the Kernel object to the directionary since
    # TF Tensor does not allow deep copy.
    out_dict = {
      "nngp": output.nngp,
      "ntk": output.ntk,
      "cov1": output.cov1,
      "cov2": output.cov2,
      "x1_is_x2": output.x1_is_x2,
      "is_gaussian": output.is_gaussian,
      "is_reversed": output.is_reversed,
      "is_input": output.is_input,
      "diagonal_batch": output.diagonal_batch,
      "diagonal_spatial": output.diagonal_spatial,
      "shape1": output.shape1,
      "shape2": output.shape2,
      "batch_axis": output.batch_axis,
      "channel_axis": output.channel_axis,
      "mask1": output.mask1,
      "mask2": output.mask2
    }
    return out_dict

  if hasattr(output, '_asdict'):
    return output._asdict()

  if isinstance(output, types.GeneratorType):
    return (_output_to_dict(out) for out in output)

  raise ValueError(type(output))


def wraps(f):
  def wrapper(g):
    @functools.wraps(f)
    def h(*args, **kwargs):
      return g(*args, **kwargs)

    h.__signature__ = inspect.signature(f)
    return h
  return wrapper


def get_namedtuple(name):
  def getter_decorator(fn):
    try:
      argspec = inspect.getfullargspec(fn)
      get_index = argspec.args.index('get')
      defaults = argspec.defaults
    except:
      raise ValueError('`get_namedtuple` functions must have a `get` argument.')

    @wraps(fn)
    def getter_fn(*args, **kwargs):
      canonicalized_args = list(args)

      if 'get' in kwargs:
        get_is_not_tuple, get = canonicalize_get(kwargs['get'])
        kwargs['get'] = get
      elif get_index < len(args):
        get_is_not_tuple, get = canonicalize_get(args[get_index])
        canonicalized_args[get_index] = get
      elif defaults is None:
        raise ValueError(
            '`get_namedtuple` function must have a `get` argument provided or'
            'set by default.')
      else:
        get_is_not_tuple, get = canonicalize_get(defaults[get_index -
                                                          len(args)])

      fn_out = fn(*canonicalized_args, **kwargs)

      if get is None:
        if isinstance(fn_out, dict):
          ReturnType = named_tuple_factory(name, tuple(fn_out.keys()))
          fn_out = ReturnType(*fn_out.values())
        return fn_out

      fn_out = _output_to_dict(fn_out)

      if get_is_not_tuple:
        if isinstance(fn_out, types.GeneratorType):
          return (output[get[0]] for output in fn_out)
        else:
          return fn_out[get[0]]

      ReturnType = named_tuple_factory(name, get)
      if isinstance(fn_out, types.GeneratorType):
        return (ReturnType(*tuple(output[g] for g in get)) for output in fn_out)
      else:
        return ReturnType(*tuple(fn_out[g] for g in get))

    return getter_fn

  return getter_decorator


def x1_is_x2(x1: np.ndarray,
             x2: np.ndarray = None,
             eps: float = 1e-12) -> Union[bool, np.ndarray]:
  if not isinstance(x1, np.ndarray):
    raise TypeError('`x1` must be an ndarray. A {} is found.'.format(type(x1)))

  if x2 is None:
    return True

  if x1 is x2:
    return True

  if x1.shape != x2.shape:
    return False

  if xla_bridge.get_backend().platform == 'tpu':
    eps = 1e-4

  return np.all(np.abs(x1 - x2) < eps)


def canonicalize_axis(axis: Axes,
                      x: Union[int, Sized, np.ndarray]) -> List[int]:
  """Converts axis into a sorted non-negative list.

  Args:
    axis: input axis.
    x: array / shape / number of dimensions.

  Returns:
    A sorted list of integer axes.
  """
  axis = [axis] if isinstance(axis, int) else list(axis)
  if hasattr(x, 'ndim'):
    ndim = x.ndim
  elif isinstance(x, tf.TensorShape) or isinstance(x, tuple):
    ndim = len(x)
  elif hasattr(x, '__len__'):
    ndim = len(x.shape)
  elif isinstance(x, int):
    ndim = x
  else:
    raise TypeError(x, type(x))
  return list(set(onp.arange(ndim)[axis]))


def zip_axes(x: np.ndarray,
             start_axis: int = 0,
             end_axis: int = -1) -> np.ndarray:
  """Zip (interleave) axes starting from `start_axis`.

  Changes the shape as follows:
  `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip (interleave).
    end_axis: `int`, number of axis until which to zip (interleave).

  Returns:
    A `np.ndarray` with a new shape.
  """
  return _zip_axes(x, start_axis, end_axis, unzip=False)


def unzip_axes(x: np.ndarray,
               start_axis: int = 0,
               end_axis: int = -1) -> np.ndarray:
  """Unzip (de-interleave) axes starting from `start_axis`.

  Changes the shape as follows:
  `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to unzip (de-interleave).
    end_axis: `int`, number of axis until which to unzip (de-interleave).

  Returns:
    A `np.ndarray` with a new shape.
  """
  return _zip_axes(x, start_axis, end_axis, unzip=True)


def _zip_axes(x: np.ndarray,
              start_axis: int = 0,
              end_axis: int = -1,
              unzip: bool = False) -> np.ndarray:
  """Zip/unzip (interleave/de-interleave) axes starting from `start_axis`.

  Changes the shape as follows:
    If `unzip == True`:
    `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ..]`
    If `unzip == False`:
    `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ..]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip/unzip.
    end_axis: `int`, number of axis until which to zip/unzip.
    unzip: `bool`, set to `True` to unzip instead of zip.

  Returns:
    A `np.ndarray` with a new shape.
  """
  if end_axis == -1:
    end_axis = len(x.shape)
  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to zip, got {end_axis - start_axis}.')

  odd_axes = range(start_axis + 1, end_axis, 2)
  last_axes = range(end_axis - half_ndim, end_axis)

  if unzip:
    x = np.moveaxis(x, odd_axes, last_axes)
  else:
    x = np.moveaxis(x, last_axes, odd_axes)
  return x


def transpose_zipped(x: np.ndarray) -> np.ndarray:
  return np.moveaxis(x, range(1, x.ndim, 2), range(0, x.ndim, 2))


def diagonal_between(x: np.ndarray,
                     start_axis: int = 0,
                     end_axis: int = -1) -> np.ndarray:
  """Returns the diagonal along all dimensions between start and end axes."""
  if end_axis == -1:
    end_axis = x.ndim
  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to flatten, got {end_axis - start_axis}.')
  if half_ndim == 0:
    return x

  side_shape = x.shape[start_axis:start_axis + half_ndim]
  side_size = size_at(side_shape)

  shape_2d = x.shape[:start_axis] + (side_size, side_size) + x.shape[end_axis:]
  shape_result = x.shape[:start_axis] + side_shape + x.shape[end_axis:]

  x = np.diagonal(x.reshape(shape_2d), axis1=start_axis, axis2=start_axis+1)
  x = np.moveaxis(x, -1, start_axis)
  return x.reshape(shape_result)


def zip_flat(x, y):
  return tuple(c for xy in zip(x, y) for c in xy)


def interleave_ones(x, start_axis, end_axis, x_first):
  x_axes = x.shape[start_axis:end_axis]
  ones = (1,) * (end_axis - start_axis)
  shape = x.shape[:start_axis]
  if x_first:
    shape += zip_flat(x_axes, ones)
  else:
    shape += zip_flat(ones, x_axes)
  shape += x.shape[end_axis:]
  return x.reshape(shape)


def outer_prod(x, y, start_axis, end_axis, prod_op):
  if y is None:
    y = x
  x = interleave_ones(x, start_axis, end_axis, True)
  y = interleave_ones(y, start_axis, end_axis, False)
  if x.ndim <= 5:
    return prod_op(x, y)
  elif x.ndim == 6:
    x = np.tile(x, (1, x.shape[0], 1, 1, 1, 1))
    y = np.tile(y, (y.shape[1], 1, 1, 1, 1, 1))
    z = np.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])
    k = np.reshape(y, (y.shape[0] * y.shape[1],) + y.shape[2:])
    result = prod_op(z, k)
    result = np.reshape(result, (x.shape[0], x.shape[1],) + result.shape[1:])
    return result
  else:
    raise ValueError("Current setting does not support matrices of rank beyond 6")


ArrayOrList = Union[Optional[np.ndarray], List[Optional[np.ndarray]]]


@dataclasses.dataclass
class MaskedArray:
  masked_value: ArrayOrList
  mask: ArrayOrList

  astuple = ...  # type: Callable[[], Tuple[ArrayOrList, ArrayOrList]]


def get_masked_array(x: ArrayOrList,
                     mask_constant: float = None) -> MaskedArray:
  """Return `x` with entries equal to `mask_constant` zeroed-out, and the mask.

  The mask returned is a boolean `np.ndarray` with masked indices having `True`.

  Args:
    x: `np.ndarray` to mask. If `x` is a `MaskedInput`, treat it as
      `(masked_x, mask)` and pass it through.
    mask_constant: an optional `float`, the value in inputs to be considered as
      masked (e.g. padding in a batch of sentences). `None` means no masking.
      Can also be `np.nan`, `np.inf` etc.

  Returns:
    A `MaskedArray` of `(masked_x, boolean_mask)`.
  """
  if isinstance(x, list):
    x_array = []
    mask_array = []
    for x_ in x:
      masked_array = get_masked_array(x_, mask_constant)
      x_array.append(masked_array.masked_value)
      mask_array.append(masked_array.mask)
    # fields = zip(*(get_masked_array(_x, mask_constant).astuple() for _x in x))
    # return MaskedArray(*(list(f) for f in fields))
    return MaskedArray(x_array, mask_array)

  if x is None:
    mask = None

  if isinstance(x, MaskedArray):
    masked_value = x.masked_value
    mask = x.mask
    x = masked_value

  elif isinstance(x, np.ndarray):
    if mask_constant is None:
      mask = None
    else:
      choice_a = lambda: np.array(tf.math.is_nan(x))
      choice_b = lambda: x == mask_constant
      # mask = choice_a(x) if math.isnan(mask_constant) else choice_b(x)
      mask = tf.cond(tf.math.is_nan(mask_constant), choice_a, choice_b)
  else:
    raise TypeError(x, type(x))

  if mask is not None:
    x = np.where(mask, np.zeros((), x.dtype), x)

  return MaskedArray(x, mask)  # pytype: disable=wrong-arg-count


def size_at(x: Union[np.ndarray, Sequence[int]],
            axes: Iterable[int] = None) -> int:
  if hasattr(x, 'shape'):
    x = x.shape

  if axes is None:
    axes = range(len(x))

  return functools.reduce(operator.mul, [x[a] for a in axes], 1)


def get_res_batch_dims(contracting_dims: List[int],
                       batch_dims: List[int]) -> List[int]:
  res_batch_dims = [2 * b - i for i, b in enumerate(batch_dims)]
  for i, b in enumerate(batch_dims):
    for c in contracting_dims:
      if b > c:
        res_batch_dims[i] -= 2
  return res_batch_dims


def dot_general(lhs: np.ndarray,
                rhs: np.ndarray,
                contracting_dims: Axes,
                batch_dims: Axes,
                precision=None) -> np.ndarray:
  """`dot_general` with preserved dims order and shared lhs / rhs dims.

  Precisely, returns `dot_general(lhs, rhs, dimension_numbers)` where
  `dimension_numbers == ((contracting_dims, contracting_dims),
                         (batch_dims, batch_dims))`,
  but allows arbitrary dimension order and preserves it in the output. See XLA's
   `DotGeneral<https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`.

  Args:
    lhs: np.ndarray.
    rhs: np.ndarray.
    contracting_dims: contracting dimensions.
    batch_dims: batch dimensions.
    precision: Optional. Either `None`, which means the default precision for
      the backend, or a `Precision` enum value.

  Returns:
    Dot product result with preserved dimension order.
  """
  contracting_dims = canonicalize_axis(contracting_dims, lhs)
  batch_dims = canonicalize_axis(batch_dims, lhs)

  n_batch_dims = len(batch_dims)
  leading_batch_dims = range(n_batch_dims)

  dimension_numbers = ((contracting_dims, contracting_dims),
                       (leading_batch_dims, leading_batch_dims))

  lhs = np.moveaxis(lhs, batch_dims, leading_batch_dims)
  if rhs is None:
    rhs = lhs
  else:
    rhs = np.moveaxis(rhs, batch_dims, leading_batch_dims)

  prod = tf_dot_general(lhs, rhs, dimension_numbers)
  prod = zip_axes(prod, n_batch_dims)

  res_batch_dims = get_res_batch_dims(contracting_dims, batch_dims)
  prod = np.moveaxis(prod, leading_batch_dims, res_batch_dims)
  return prod


def make_2d(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
  if mat is None:
    return mat

  if mat.ndim % 2 == 1:
    raise ValueError('Expected an even-dimensional matrix. Please file a bug at'
                     'https://github.com/google/neural-tangents/issues/new')

  mat = unzip_axes(mat)
  mat = mat.reshape((size_at(mat.shape[:mat.ndim // 2]),
                     size_at(mat.shape[mat.ndim // 2:])))
  return mat


def is_on_cpu(x: PyTree) -> bool:
  def _arr_is_on_cpu(x: np.ndarray) -> bool:
    # TODO(romann): revisit when https://github.com/google/jax/issues/1431 and
    # https://github.com/google/jax/issues/1432 are fixed.
    if hasattr(x, 'device_buffer'):
      return 'cpu' in str(x.device_buffer.device()).lower()

    if isinstance(x, np.ndarray):
      return True

    raise NotImplementedError(type(x))

  return tree_all(tree_map(_arr_is_on_cpu, x))
