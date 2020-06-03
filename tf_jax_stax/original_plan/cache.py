def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None,
                bias_init=RandomNormal(seed=1e-6)):
  """ General Convolutional layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or GlorotNormal()
