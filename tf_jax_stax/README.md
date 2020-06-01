## TensorFlow 2.x. stax APIs, equivalent to [`jax.experimental.stax`](https://jax.readthedocs.io/en/latest/jax.experimental.stax.html)

### Implementation Adjustments (compared to `jax.experimental.stax`):
1. As the parameter for the function `init_func`, instead of sending in the PRNGKey
as that in `jax.experimental.jax`, a rng seed is sent in. The reason is that there
are minor differences between the TF Initializer/Generator APIs and those in JAX (
i.e., when creating an initializer object in TF, it requires a seed rather than a
real prng key). It is possible to use `tf.random.Generator` as a walkaround,
however, the obstacle is there is no Glorol Normal distribution method for
`tf.random.Generator` objects.  
