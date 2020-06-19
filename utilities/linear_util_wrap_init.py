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
Apply transformations on functions, from
<https://github.com/google/jax/blob/master/jax/linear_util.py>
"""


class WrappedFun(object):
  """Represents a function `f` to which `transforms` are to be applied.
  Arguments:
    f: the function to be transformed.
    transforms: a list of `(gen, gen_static_args)` tuples representing
      transformations to apply to `f.` Here `gen` is a generator function
      and `gen_static_args` is a tuple of static arguments for the generator. See
      description at the start of this module for the expected behavior of the
      generator.
    stores: a list of out_store for the auxiliary output of the `transforms`.
    params: extra parameters to pass as keyword arguments to `f`, along with the
      transformed keyword arguments.
  """
  __slots__ = ("f", "transforms", "stores", "params")

  def __init__(self, f, transforms, stores, params):
    self.f = f
    self.transforms = transforms
    self.stores = stores
    self.params = params

  @property
  def __name__(self):
    return getattr(self.f, '__name__', '<unnamed wrapped function>')

  def wrap(self, gen, gen_static_args, out_store) -> 'WrappedFun':
    """Add another transform and its store."""
    return WrappedFun(self.f, ((gen, gen_static_args),) + self.transforms,
                      (out_store,) + self.stores, self.params)

  def populate_stores(self, stores):
    """Copy the values from the `stores` into `self.stores`."""
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    """Calls the underlying function, applying the transforms.
    The positional `args` and keyword `kwargs` are passed to the first
    transformation generator.
    """
    stack = []
    for (gen, gen_static_args), out_store in zip(self.transforms, self.stores):
      gen = gen(*(gen_static_args + tuple(args)), **kwargs)
      args, kwargs = next(gen)
      stack.append((gen, out_store))
    gen = None

    ans = self.f(*args, **dict(self.params, **kwargs))
    del args
    while stack:
      gen, out_store = stack.pop()
      ans = gen.send(ans)
      if out_store is not None:
        ans, side = ans
        out_store.store(side)

    return ans

  def __repr__(self):
    def transform_to_str(x):
      i, (gen, args) = x
      return "{}   : {}   {}".format(i, fun_name(gen), fun_name(args))
    transformation_stack = map(transform_to_str, enumerate(self.transforms))
    return "Wrapped function:\n" + '\n'.join(transformation_stack) + '\nCore: ' + fun_name(self.f) + '\n'

  def __hash__(self):
    return hash((self.f, self.transforms, self.params))

  def __eq__(self, other):
    return (self.f == other.f and self.transforms == other.transforms and
            self.params == other.params)


def wrap_init(f, params={}) -> WrappedFun:
  """Wraps function `f` as a `WrappedFun`, suitable for transformation."""
  return WrappedFun(f, (), (), tuple(sorted(params.items())))
