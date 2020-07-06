### This markdown file contains a list of JAX symbols used for the current Neural Tangents.

### JAX:

1. `jax.lax`
    - [x] [`jax.lax.add`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html#jax.lax.add) (`trax.tf_numpy.numpy.add`)
    - [x] [`jax.lax.cond`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) (`tf.cond`)
    - [x] [`jax.lax.conv_general_dilated`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated) ([TF `conv_general_dilated`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_jax_stax/tf_conv_general.py))
    - [x] [`jax.lax.dot_general`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html#jax.lax.dot_general) ([TF `dot_general`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_dot_general/tf_dot_general.py))
    - [x] [`jax.lax.padtype_to_pads`](https://jax.readthedocs.io/en/latest/_modules/jax/lax/lax.html) ([TF `lax`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_jax_stax/tf_lax.py))
    - [x] [`jax.lax.reduce_window`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html#jax.lax.reduce_window) ([TF `reduce_window`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_jax_stax/tf_reduce_window.py))
    - [x] [`jax.lax.reduce_window_shape_tuple`](https://jax.readthedocs.io/en/latest/_modules/jax/lax/lax.html) ([TF `lax`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_jax_stax/tf_lax.py))
    - [ ] [`jax.lax._reduce_window_sum`](https://jax.readthedocs.io/en/latest/_modules/jax/lax/lax.html)

2. `jax.linear_util`
    - [x] [`jax.linear_util.wrap_init`](https://github.com/google/jax/blob/master/jax/linear_util.py) ([here](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/utilities))

3. `jax.numpy` (`trax.tf_numpy.numpy`, `tf.math`, `tf.linalg`, `tf`)
    - [x] [`jax.numpy.abs`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html)
    - [x] [`jax.numpy.all`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.all.html)
    - [x] [`jax.numpy.any`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.any.html#jax.numpy.any)
    - [x] [`jax.numpy.arccos`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arccos.html)
    - [x] [`jax.numpy.arcsin`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arcsin.html)
    - [x] [`jax.numpy.asarray`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.asarray.html)
    - [x] [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)
    - [x] [`jax.numpy.arange`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arange.html)
    - [ ] `jax.numpy.bool_`
    - [x] [`jax.numpy.broadcast_to`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.broadcast_to.html)
    - [x] [`jax.numpy.clip`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.clip.html)
    - [x] [`jax.numpy.concatenate`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.concatenate.html)
    - [x] [`jax.numpy.count_nonzero`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.count_nonzero.html)
    - [x] [`jax.numpy.cos`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cos.html)
    - [x] [`jax.numpy.dot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html)
    - [x] [`jax.numpy.expand_dims`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expand_dims.html)
    - [x] [`jax.numpy.diag`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.diag.html)
    - [x] [`jax.numpy.diagonal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.diagonal.html)
    - [x] [`jax.numpy.diag_indices`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.diag_indices.html)
    - [ ] [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html) (`tf.einsum`, may need a TF Numpy wrapper later on)
    - [x] [`jax.numpy.eye`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html#jax.numpy.eye)
    - [x] [`jax.numpy.expm1`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expm1.html#jax.numpy.expm1)
    - [x] [`jax.numpy.expand_dims`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expand_dims.html)
    - [x] [`jax.numpy.full`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.full.html)
    - [x] `jax.numpy.float64`
    - [x] `jax.numpy.inf`
    - [x] [`jax.numpy.isnan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isnan.html)
    * [`jax.numpy.linalg.eigh`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh) (`tf.linalg.eigh`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.linalg.eigvalsh`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigvalsh.html#jax.numpy.linalg.eigvalsh) (`tf.linalg.eigvalsh`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.linalg.norm`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.norm.html#jax.numpy.linalg.norm) (`tf.norm`, may need a TF Numpy wrapper later on)
    - [x] [`jax.numpy.logspace`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logspace.html#jax.numpy.logspace)
    - [x] `jax.numpy.minimum`
    - [x] `jax.numpy.maximum`
    - [x] `jax.numpy.max`
    - [x] [`jax.numpy.matmul`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html)
    - [x] [`jax.numpy.mean`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.mean.html)
    - [x] [`jax.numpy.moveaxis`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.moveaxis.html)
    - [x] `jax.numpy.ndarray`
    - [x] `jax.numpy.nan`
    - [x] [`jax.numpy.ones`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ones.html)
    - [x] [`jax.numpy.ones_like`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.outer.html)
    - [x] [`jax.numpy.outer`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.outer.html)
    - [x] [`jax.numpy.pad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html)
    - [x] [`jax.numpy.prod`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.prod.html)
    - [ ] `jax.numpy.pi` (TF nightly version; `tf.constant(math.pi)`)
    - [x] [`jax.numpy.reshape`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html)
    - [x] [`jax.numpy.round`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.round.html) (`tf.math.round`, may need a TF Numpy wrapper later on)
    - [x] [`jax.numpy.squeeze`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html)
    - [ ] `jax.numpy.sign` (`tf.math.sign`, may need a TF Numpy wrapper later on)
    - [ ] `jax.numpy.size` (`tf.size`, may need a TF Numpy wrapper later on)
    - [x] [`jax.numpy.sort`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html#jax.numpy.sort)
    - [x] [`jax.numpy.stack`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html)
    - [x] [`jax.numpy.split`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.split.html)
    - [x] [`jax.numpy.sqrt`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sqrt.html#jax.numpy.sqrt)
    - [x] [`jax.numpy.sum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sum.html)
    - [x] [`jax.numpy.tensordot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tensordot.html)
    - [x] [`jax.numpy.trace`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.trace.html)
    - [x] [`jax.numpy.transpose`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html#jax.numpy.transpose)
    - [x] [`jax.numpy.take`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.take.html)
    - [x] `jax.numpy.uint32`
    - [x] [`jax.numpy.var`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.var.html)
    - [x] [`jax.numpy.where`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html)
    - [x] [`jax.numpy.zeros_like`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.zeros_like.html)

4. `jax.ops`
    - [ ] [`jax.ops.index_mul`](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_mul.html#jax.ops.index_mul)
    - [ ] [`jax.ops.index_update`](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_update.html)
    - [ ] [`jax.ops.index`](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index.html#jax.ops.index)
5. [`jax.random`](https://jax.readthedocs.io/en/latest/jax.random.html) (TF random)
    - [x] `jax.random.split`
    - [x] `jax.random.normal`
    - [x] `jax.random.uniform`
    - [x] `jax.random.bernoulli`
    - [ ] `jax.random.PRNGKey`
6. - [ ] [`jax.abstract_arrays.ShapedArray`](https://github.com/google/jax/blob/master/jax/abstract_arrays.py) (Avoid direct access)
7. - [ ] [`jax.api_util.flatten_fun`](https://github.com/google/jax/blob/master/jax/api_util.py)
8. [`jax.experimental.stax`](https://jax.readthedocs.io/en/latest/jax.experimental.stax.html) ([TF equivalent functionalities](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/tf_jax_stax))
    - [x] `jax.experimental.stax.serial` (line 305 in stax.py)
    - [x] `jax.experimental.stax.parallel` (line 334 in stax.py)
    - [x] `jax.experimental.stax.GeneralConv` (line 592 in stax.py)
    - [x] `jax.experimental.stax.FanOut` (line 738 in stax.py)
    - [x] `jax.experimental.stax.FanInSum` (line 752 in stax.py)
    - [x] `jax.experimental.stax.FanInConcat` (line 771 in stax.py)
    - [x] `jax.experimental.stax.AvgPool` (line 873 in stax.py)
    - [x] `jax.experimental.stax.SumPool` (line 875 in stax.py)
    - [x] `jax.experimental.stax._pooling_layer` (line 908 in stax.py)
    - [x] `jax.experimental.stax.Identity` (line 1155 in stax.py)
    - [x] `jax.experimental.stax.softmax` (line 1349 in stax.py)
    - [x] `jax.experimental.stax.Dropout` (line 1559 in stax.py)
    - [x] `jax.experimental.stax.elementwise` (line 2048 in stax.py)
9. - [ ] [`jax.interpreters.partial_eval.abstract_eval_fun`](https://github.com/google/jax/blob/master/jax/interpreters/partial_eval.py) (avoid direct access)
10. - [ ] [`jax.lib.xla_bridge.get_backend`](https://jax.readthedocs.io/en/latest/_modules/jax/lib/xla_bridge.html) (avoid direct access)
11. - [ ] [`jax.scipy.special.erf`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.special.erf.html#jax.scipy.special.erf) (`tf.math.erf`, may need a TF Numpy wrapper)
12. - [ ] [`jax.scipy.linalg.solve`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.solve.html) (`tf.linalg.solve`, may need a TF Numpy wrapper later on)
13. - [x] [`jax.tree_util`](https://jax.readthedocs.io/en/latest/jax.tree_util.html)
    * `jax.tree_util.tree_map`
    * `jax.tree_util.tree_flatten`
    * `jax.tree_util.tree_unflatten`
    * `jax.tree_util.register_pytree_node`
    * `jax.tree_util.tree_reduce`
    * `jax.tree_util.tree_multimap`
14. - [x] [`jax.api.grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad) ([`grad` in TF Numpy extensions](https://github.com/google/trax/blob/master/trax/tf_numpy/extensions/extensions.py#L219))
15. - [x] [`jax.api.jit`](https://jax.readthedocs.io/en/latest/_modules/jax/api.html) ([`jit` in TF Numpy extensions](https://github.com/google/trax/blob/master/trax/tf_numpy/extensions/extensions.py#L282))
16. - [x] [`jax.api.pmap`](https://jax.readthedocs.io/en/latest/_modules/jax/api.html) ([`pmap` in TF Numpy extensions](](https://github.com/google/trax/blob/master/trax/tf_numpy/extensions/extensions.py#L1020)))
17. - [x] [`jax.api.device_get`](https://jax.readthedocs.io/en/latest/_modules/jax/api.html) (`with tf.device("CPU"): np.array(x)`)
18. - [x] [`jax.api.jacobian`](https://github.com/google/jax/blob/master/jax/api.py#L600) ([`tf.GradientTape.jacobian`](https://www.tensorflow.org/api_docs/python/tf/GradientTape#jacobian))
19. - [ ] `jax.api.jvp` (Need a direct jvp interface based on the current forward-mode in TF)
20. - [x] `jax.api.vjp` (`vjp` in TF Numpy extensions)
21. - [x] [`jax.api.vmap`] ([`tf.vectorized_map`](https://www.tensorflow.org/api_docs/python/tf/vectorized_map))
22. - [x] `jax.api.eval_shape` (`eval_on_shapes` in TF Numpy extensions)
23. - [ ] [`Config` object in `jax.config`](https://github.com/google/jax/blob/master/jax/config.py#L39)
24. - [ ] [`jax.interpreters.pxla.ShardedDeviceArray`](https://jax.readthedocs.io/en/latest/_modules/jax/interpreters/pxla.html) (avoid direct access)
25. - [x] [`jax.test_util`](https://github.com/google/jax/blob/master/jax/test_util.py)
    * `jax.test_util._default_tolerance`
    * `jax.test_util.device_under_test`
    * `jax.test_util.JaxTestCase`
    * `jax.test_util.cases_from_list`
26. - [x] `jax.experimental.optimizers.optimizer`
27. - [x] `jax.experimental.optimizers.momentum`
28. - [ ] `jax.experimental.optimizers.make_schedule`
29. - [x] `jax.experimental.optimizers.sgd`
