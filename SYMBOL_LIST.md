### This markdown file contains a list of JAX symbols used for the current Neural Tangents.

### JAX:

1. `jax.lax`
    * [`jax.lax.add`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html#jax.lax.add) (`trax.tf_numpy.numpy.add`)
    * [`jax.lax.cond`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) (`tf.cond`)
    * [`jax.lax.conv_general_dilated`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated) ([TF `conv_general_dilated`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_jax_stax/tf_conv_general.py))
    * [`jax.lax.dot_general`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html#jax.lax.dot_general)
    * `jax.lax.dot_general_dilated`
    * [`jax.lax.padtype_to_pads`](https://jax.readthedocs.io/en/latest/_modules/jax/lax/lax.html)
    * [`jax.lax.reduce_window`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html#jax.lax.reduce_window) ([TF `reduce_window`](https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/tf_jax_stax/tf_reduce_window.py))
    * [`jax.lax.reduce_window_shape_tuple`](https://jax.readthedocs.io/en/latest/_modules/jax/lax/lax.html)
    * [`jax.lax._reduce_window_sum`](https://jax.readthedocs.io/en/latest/_modules/jax/lax/lax.html)

2. `jax.linear_util`
    * [`jax.linear_util.wrap_init`](https://github.com/google/jax/blob/master/jax/linear_util.py) ([here](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/utilities))

3. `jax.numpy` (`trax.tf_numpy.numpy`, `tf.math`, `tf.linalg`, `tf`)
    * [`jax.numpy.abs`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html)
    * [`jax.numpy.all`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.all.html)
    * [`jax.numpy.any`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.any.html#jax.numpy.any)
    * [`jax.numpy.arccos`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arccos.html)
    * [`jax.numpy.arcsin`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arcsin.html)
    * [`jax.numpy.asarray`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.asarray.html)
    * [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)
    * [`jax.numpy.arange`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arange.html)
    * `jax.numpy.bool_`
    * [`jax.numpy.broadcast_to`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.broadcast_to.html)
    * [`jax.numpy.clip`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.clip.html)
    * [`jax.numpy.concatenate`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.concatenate.html)
    * [`jax.numpy.count_nonzero`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.count_nonzero.html)
    * [`jax.numpy.cos`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cos.html)
    * [`jax.numpy.dot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html)
    * [`jax.numpy.expand_dims`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expand_dims.html)
    * [`jax.numpy.diag`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.diag.html)
    * [`jax.numpy.diagonal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.diagonal.html)
    * [`jax.numpy.diag_indices`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.diag_indices.html)
    * [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html) (`tf.einsum`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.eye`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html#jax.numpy.eye)
    * [`jax.numpy.expm1`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expm1.html#jax.numpy.expm1)
    * [`jax.numpy.expand_dims`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expand_dims.html)
    * [`jax.numpy.full`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.full.html)
    * `jax.numpy.float64`
    * `jax.numpy.inf`
    * [`jax.numpy.isnan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isnan.html)
    * [`jax.numpy.linalg.eigh`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh) (`tf.linalg.eigh`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.linalg.eigvalsh`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigvalsh.html#jax.numpy.linalg.eigvalsh) (`tf.linalg.eigvalsh`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.linalg.norm`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.norm.html#jax.numpy.linalg.norm) (`tf.norm`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.logspace`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logspace.html#jax.numpy.logspace)
    * `jax.numpy.minimum`
    * `jax.numpy.maximum`
    * `jax.numpy.max`
    * [`jax.numpy.matmul`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html)
    * [`jax.numpy.mean`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.mean.html)
    * [`jax.numpy.moveaxis`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.moveaxis.html)
    * `jax.numpy.ndarray`
    * `jax.numpy.nan`
    * [`jax.numpy.ones`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ones.html)
    * [`jax.numpy.ones_like`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.outer.html)
    * [`jax.numpy.outer`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.outer.html)
    * [`jax.numpy.pad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html)
    * [`jax.numpy.prod`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.prod.html)
    * \*`jax.numpy.pi` (`tf.constant(math.pi)`?)
    * [`jax.numpy.reshape`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html)
    * [`jax.numpy.round`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.round.html) (`tf.math.round`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.squeeze`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html)
    * `jax.numpy.sign` (`tf.math.sign`, may need a TF Numpy wrapper later on)
    * `jax.numpy.size` (`tf.size`, may need a TF Numpy wrapper later on)
    * [`jax.numpy.sort`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html#jax.numpy.sort)
    * [`jax.numpy.stack`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html)
    * [`jax.numpy.split`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.split.html)
    * [`jax.numpy.sqrt`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sqrt.html#jax.numpy.sqrt)
    * [`jax.numpy.sum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sum.html)
    * [`jax.numpy.tensordot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tensordot.html)
    * [`jax.numpy.trace`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.trace.html)
    * [`jax.numpy.transpose`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html#jax.numpy.transpose)
    * [`jax.numpy.take`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.take.html)
    * `jax.numpy.uint32`
    * [`jax.numpy.var`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.var.html)
    * [`jax.numpy.where`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html)
    * [`jax.numpy.zeros_like`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.zeros_like.html)

4. \*`jax.ops`
    * [`jax.ops.index_mul`](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_mul.html#jax.ops.index_mul)
    * [`jax.ops.index_update`](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_update.html)
    * [`jax.ops.index`](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index.html#jax.ops.index)
5. [`jax.random`](https://jax.readthedocs.io/en/latest/jax.random.html) (TF random)
    * `jax.random.split`
    * `jax.random.normal`
    * `jax.random.uniform`
    * `jax.random.bernoulli`
    * `jax.random.PRNGKey`
6. \*[`jax.abstract_arrays.ShapedArray`](https://github.com/google/jax/blob/master/jax/abstract_arrays.py)(Only appeared [once](https://github.com/google/neural-tangents/search?q=shapedarray&unscoped_q=shapedarray) in Neural Tangents, and TF ndarray can be an equivalence)
7. \*[`jax.api_util.flatten_fun`](https://github.com/google/jax/blob/master/jax/api_util.py) (based on `tree_flatten` in JAX tree utils - may need an equivalent set of tree utils in TF)
8. [`jax.experimental.stax`](https://jax.readthedocs.io/en/latest/jax.experimental.stax.html) ([TF equivalent functionalities](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/tf_jax_stax))
    * `jax.experimental.stax.serial` (line 305 in stax.py)
    * `jax.experimental.stax.parallel` (line 334 in stax.py)
    * `jax.experimental.stax.GeneralConv` (line 592 in stax.py)
    * `jax.experimental.stax.FanOut` (line 738 in stax.py)
    * `jax.experimental.stax.FanInSum` (line 752 in stax.py)
    * `jax.experimental.stax.FanInConcat` (line 771 in stax.py)
    * `jax.experimental.stax.AvgPool` (line 873 in stax.py)
    * `jax.experimental.stax.SumPool` (line 875 in stax.py)
    * `jax.experimental.stax._pooling_layer` (line 908 in stax.py)
    * `jax.experimental.stax.Identity` (line 1155 in stax.py)
    * `jax.experimental.stax.softmax` (line 1349 in stax.py)
    * `jax.experimental.stax.Dropout` (line 1559 in stax.py)
    * `jax.experimental.stax.elementwise` (line 2048 in stax.py)
9. \*[`jax.interpreters.partial_eval.abstract_eval_fun`](https://github.com/google/jax/blob/master/jax/interpreters/partial_eval.py)
10. \*[`jax.lib.xla_bridge.get_backend`](https://jax.readthedocs.io/en/latest/_modules/jax/lib/xla_bridge.html)
11. \*[`jax.scipy.special.erf`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.special.erf.html#jax.scipy.special.erf) (`tf.math.erf`, may need a TF Numpy wrapper)
12. [`jax.scipy.linalg.solve`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.solve.html) (`tf.linalg.solve`, may need a TF Numpy wrapper later on)
13. \*[`jax.tree_util`](https://jax.readthedocs.io/en/latest/jax.tree_util.html)
    * `jax.tree_util.tree_map`
    * `jax.tree_util.tree_flatten`
    * `jax.tree_util.tree_unflatten`
    * `jax.tree_util.register_pytree_node`
    * `jax.tree_util.tree_reduce`
    * `jax.tree_util.tree_multimap`
14. [`jax.api.grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad)
15. [`jax.api.jit`](https://jax.readthedocs.io/en/latest/_modules/jax/api.html)
16. [`jax.api.pmap`](https://jax.readthedocs.io/en/latest/_modules/jax/api.html)
17. [`jax.api.device_get`](https://jax.readthedocs.io/en/latest/_modules/jax/api.html)
18. `jax.api.jacobian`
19. `jax.api.jvp`
20. `jax.api.vjp` (`vjp` in trax extensions)
21. `jax.api.vmap`
22. `jax.api.eval_shape`
23. `jax.config.config`
24. `jax.config.config.parse_flags_with_absl`
24. [`jax.interpreters.pxla.ShardedDeviceArray`](https://jax.readthedocs.io/en/latest/_modules/jax/interpreters/pxla.html)
25. [`jax.test_util`](https://github.com/google/jax/blob/master/jax/test_util.py)
    * `jax.test_util._default_tolerance`
    * `jax.test_util.device_under_test`
    * `jax.test_util.JaxTestCase`
    * `jax.test_util.parameterized.named_parameters`
    * `jax.test_util.parameterized.parameters`
    * `jax.test_util.cases_from_list`
    * `jax.absltest.main`
26. `jax.experimental.optimizers.optimizer`
27. `jax.experimental.optimizers.momentum`
28. `jax.experimental.optimizers.make_schedule`
29. `jax.experimental.optimizers.sgd`
